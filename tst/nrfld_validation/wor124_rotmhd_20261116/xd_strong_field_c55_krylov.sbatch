#!/bin/bash -l
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor124_B129_krylov
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_floor_trial"
reader="$base/scripts/read_restart.py"
rst="$base/continuation/combined_B129_c50to56_cap1e5_64r_job_340985/output/combined_B129_c50to56_cap1e5_64r.00008.rst"
job="$base/strong_field_c55_krylov/job_${SLURM_JOB_ID}"
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

if [[ $(python3 "$reader" "$rst" --field cycle) != 55 ]]; then
  echo 'FAIL: expected cycle 55 checkpoint' > "$job/status.txt"
  exit 1
fi

cd "$source_dir"
{
  date -Is
  sha256sum "$rst" "$source_dir/bin/athena"
  printf 'source_dir=%s ranks=%s start_cycle=55 target=56 cfl=0.3 threshold=2e-8\n' \
    "$source_dir" "$SLURM_NTASKS"
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"

printf 'case,restart,max_iterations,mg_vcycles,exit_code,cycle_limit,has_fatal,elapsed_seconds\n' \
  > "$job/status.csv"

# Preserve the strict residual target and vary only Krylov subspace width,
# preconditioner work, and (last case) the iteration budget.
for spec in r600_mg3_n600:600:600:3 r320_mg5_n600:320:600:5 r320_mg5_n1000:320:1000:5; do
  IFS=: read -r label restart maxit mg <<< "$spec"
  out="$job/$label"
  mkdir -p "$out/output"
  started=$(date +%s)
  set +e
  /usr/bin/time -v -o "$out/resources.txt" \
    mpirun -np "$SLURM_NTASKS" "$source_dir/bin/athena" \
      -r "$rst" -d "$out/output" \
      job/problem_id="wor124_B129_${label}" \
      time/nlim=56 time/ncycle_out=1 time/cfl_number=0.3 \
      mesh/numlevel=11 \
      output2/dcycle=1 output2/variable=cons,prim,uov output4/dcycle=1 \
      nrfld/diagnostic_verbosity=0 nrfld/jacobian_check=false \
      nrfld/jacobian_check_each_cycle=false \
      nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart="$restart" \
      nrfld/fgmres_max_iterations="$maxit" nrfld/fgmres_mg_vcycles="$mg" \
      > "$out/run.log" 2>&1
  code=$?
  set -e
  limit=0
  fatal=0
  if grep -q 'Terminating on cycle limit' "$out/run.log"; then limit=1; fi
  if grep -q 'FATAL ERROR' "$out/run.log"; then fatal=1; fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$label" "$restart" "$maxit" "$mg" "$code" "$limit" "$fatal" \
    "$(($(date +%s)-started))" >> "$job/status.csv"
done

date -Is > "$job/completed.txt"
