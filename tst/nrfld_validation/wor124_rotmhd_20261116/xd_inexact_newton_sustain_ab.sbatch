#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor124_inexact_sustain
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_floor_trial"
reader="$base/scripts/read_restart.py"
rst="$base/high_amr_long/combined_B32uG_floor_amr11_3node_job_341015/output/combined_B32uG_floor_amr11_3node.00162.rst"
job="$base/inexact_sustain/job_${SLURM_JOB_ID}"
mkdir -p "$job"

module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm OMP_NUM_THREADS=1
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-6
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2

if [[ $(python3 "$reader" "$rst" --field cycle) != 1580 ]]; then
  echo 'FAIL: expected cycle 1580 checkpoint' > "$job/status.txt"
  exit 1
fi
cd "$source_dir"
{
  date -Is
  sha256sum "$rst" "$source_dir/bin/athena" "$source_dir/src/eos/general/general_mhd.cpp"
  printf 'source_dir=%s ranks=%s start_cycle=1580 target_cycle=1600 cfl=0.3 numlevel=11 fgmres=160/600 MG3\n' \
    "$source_dir" "$SLURM_NTASKS"
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"

printf 'case,fgmres_threshold,exit_code,cycle_limit,has_fatal,elapsed_seconds\n' \
  > "$job/status.csv"
for spec in strict:2e-8 loose:1e-6; do
  label=${spec%%:*}
  threshold=${spec#*:}
  out="$job/$label"
  mkdir -p "$out/output"
  started=$(date +%s)
  set +e
  /usr/bin/time -v -o "$out/resources.txt" \
    mpirun -np "$SLURM_NTASKS" "$source_dir/bin/athena" \
      -r "$rst" -d "$out/output" \
      job/problem_id="wor124_inexact_sustain_${label}" \
      time/nlim=1600 time/ncycle_out=20 time/cfl_number=0.3 \
      mesh/numlevel=11 \
      output2/dcycle=20 output2/variable=cons,prim,uov output4/dcycle=20 \
      nrfld/diagnostic_verbosity=0 nrfld/jacobian_check=false \
      nrfld/jacobian_check_each_cycle=false \
      nrfld/fgmres_threshold="$threshold" \
      nrfld/fgmres_restart=160 nrfld/fgmres_max_iterations=600 \
      nrfld/fgmres_mg_vcycles=3 \
      > "$out/run.log" 2>&1
  code=$?
  set -e
  limit=0
  fatal=0
  if grep -q 'Terminating on cycle limit' "$out/run.log"; then limit=1; fi
  if grep -q 'FATAL ERROR' "$out/run.log"; then fatal=1; fi
  printf '%s,%s,%s,%s,%s,%s\n' "$label" "$threshold" "$code" "$limit" \
    "$fatal" "$(($(date +%s)-started))" >> "$job/status.csv"
done
date -Is > "$job/completed.txt"
