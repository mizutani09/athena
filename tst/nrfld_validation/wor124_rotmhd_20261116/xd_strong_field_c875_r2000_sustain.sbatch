#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=5
#SBATCH --ntasks=320
#SBATCH --ntasks-per-node=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor124_B129_c875_r2k
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_floor_trial"
reader="$base/scripts/read_restart.py"
parent="$base/strong_field_c874_r2000_mg5/job_342092"
rst="$parent/output/wor124_B129_c874_r2000_mg5.final.rst"
job="$base/strong_field_c875_r2000_sustain/job_${SLURM_JOB_ID}"
mkdir -p "$job/output"

module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm:ofi OMP_NUM_THREADS=1
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

if [[ ! -s "$rst" ]] || [[ $(python3 "$reader" "$rst" --field cycle) != 875 ]]; then
  printf 'FAIL_INVALID_CHECKPOINT path=%s\n' "$rst" > "$job/status.txt"
  exit 1
fi

cd "$source_dir"
{
  date -Is
  printf 'source_dir=%s ranks=%s start_cycle=875 target=880 cfl=0.3 numlevel=12 fgmres=2000/2500 MG5 threshold=2e-8\n' \
    "$source_dir" "$SLURM_NTASKS"
  sha256sum "$rst" "$source_dir/bin/athena" \
    "$source_dir/src/eos/general/general_mhd.cpp"
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"

started=$(date +%s)
set +e
/usr/bin/time -v -o "$job/resources.txt" \
  mpirun -np "$SLURM_NTASKS" "$source_dir/bin/athena" \
    -r "$rst" -d "$job/output" \
    job/problem_id=wor124_B129_c875_r2000_sustain \
    time/nlim=880 time/ncycle_out=1 time/cfl_number=0.3 \
    mesh/numlevel=12 \
    output2/dcycle=5 output2/variable=cons,prim,uov output4/dcycle=1 \
    nrfld/diagnostic_verbosity=0 nrfld/jacobian_check=false \
    nrfld/jacobian_check_each_cycle=false \
    nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart=2000 \
    nrfld/fgmres_max_iterations=2500 nrfld/fgmres_mg_vcycles=5 \
    > "$job/run.log" 2>&1
code=$?
set -e
elapsed=$(($(date +%s)-started))
limit=0
fatal=0
if grep -q 'Terminating on cycle limit' "$job/run.log"; then limit=1; fi
if grep -q 'FATAL ERROR' "$job/run.log"; then fatal=1; fi
printf 'case,exit_code,cycle_limit,has_fatal,elapsed_seconds\n' > "$job/status.csv"
printf 'r2000_max2500_mg5_sustain,%s,%s,%s,%s\n' "$code" "$limit" "$fatal" "$elapsed" \
  >> "$job/status.csv"
if (( code != 0 || limit != 1 || fatal != 0 )); then exit 1; fi

sha256sum "$job/output"/*.final.rst > "$job/final_restart.sha256"
date -Is > "$job/completed.txt"
