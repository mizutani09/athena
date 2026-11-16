#!/bin/bash -l
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor124_c4_diag
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source"
rst="$base/pdebug/job_340725/combined/output/combined.final.rst"
job="$base/c4_diag/job_${SLURM_JOB_ID}"
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
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
cd "$source_dir"
{
  date -Is
  sha256sum "$rst" bin/athena
  printf 'source_commit=%s\n' "$(cat "$base/source_commit.txt")"
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"
printf 'case,exit_code,cycle_limit,has_fatal,elapsed_seconds\n' > "$job/status.csv"

failed=0
for case in cfl03 cfl01 mg5 cfl01_mg5; do
  cfl=0.3; mg=3
  case "$case" in
    cfl01|cfl01_mg5) cfl=0.1 ;;
  esac
  case "$case" in
    mg5|cfl01_mg5) mg=5 ;;
  esac
  dir="$job/$case"
  mkdir -p "$dir/output"
  started=$(date +%s)
  set +e
  /usr/bin/time -v -o "$dir/resources.txt" \
    mpirun -np "$SLURM_NTASKS" "$source_dir/bin/athena" \
      -r "$rst" -d "$dir/output" \
      job/problem_id="combined_$case" \
      time/nlim=4 time/ncycle_out=1 time/cfl_number="$cfl" \
      output2/dcycle=1 output2/variable=cons,prim,uov output4/dcycle=1 \
      nrfld/diagnostic_verbosity=0 nrfld/jacobian_check=false \
      nrfld/jacobian_check_each_cycle=false \
      nrfld/fgmres_restart=80 nrfld/fgmres_max_iterations=600 \
      nrfld/fgmres_mg_vcycles="$mg" \
      > "$dir/run.log" 2>&1
  code=$?
  set -e
  ended=$(date +%s)
  limit=0; fatal=0
  if grep -q 'Terminating on cycle limit' "$dir/run.log"; then limit=1; fi
  if grep -q 'FATAL ERROR' "$dir/run.log"; then fatal=1; fi
  printf '%s,%d,%d,%d,%d\n' "$case" "$code" "$limit" "$fatal" "$((ended-started))" \
    | tee -a "$job/status.csv"
  if (( code != 0 || limit != 1 || fatal != 0 )); then failed=1; fi
done
exit "$failed"
