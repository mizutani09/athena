#!/bin/bash -l
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor124_continue
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source"
rst="${WOR124_RESTART:?set WOR124_RESTART}"
label="${WOR124_LABEL:?set WOR124_LABEL}"
target="${WOR124_TARGET:-40}"
cfl="${WOR124_CFL:-0.1}"
cap="${WOR124_INEXACT_CAP:-1e-6}"
restart="${WOR124_FGMRES_RESTART:-80}"
max_iterations="${WOR124_FGMRES_MAX:-600}"
mg_vcycles="${WOR124_MG_VCYCLES:-3}"
variable_limiter="${WOR124_VARIABLE_LIMITER:-1}"
numlevel="${WOR124_NUMLEVEL:?set WOR124_NUMLEVEL explicitly}"
job="$base/continuation/${label}_job_${SLURM_JOB_ID}"
mkdir -p "$job/output"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
if (( SLURM_JOB_NUM_NODES > 1 )); then
  export I_MPI_FABRICS=shm:ofi
else
  export I_MPI_FABRICS=shm
fi
export OMP_NUM_THREADS=1
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER="$variable_limiter"
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_INEXACT_TRUE_CAP="$cap"
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
cd "$source_dir"
{
  date -Is
  sha256sum "$rst" bin/athena
  printf 'source_commit=%s\n' "$(cat "$base/source_commit.txt")"
  printf 'target=%s cfl=%s cap=%s restart=%s max=%s mg=%s limiter=%s numlevel=%s\n' \
    "$target" "$cfl" "$cap" "$restart" "$max_iterations" "$mg_vcycles" "$variable_limiter" "$numlevel"
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"
started=$(date +%s)
set +e
/usr/bin/time -v -o "$job/resources.txt" \
  mpirun -np "$SLURM_NTASKS" "$source_dir/bin/athena" \
    -r "$rst" -d "$job/output" \
    job/problem_id="$label" \
    time/nlim="$target" time/ncycle_out=1 time/cfl_number="$cfl" \
    mesh/numlevel="$numlevel" \
    output2/dcycle=5 output2/variable=cons,prim,uov output4/dcycle=5 \
    nrfld/diagnostic_verbosity=0 nrfld/jacobian_check=false \
    nrfld/jacobian_check_each_cycle=false \
    nrfld/fgmres_restart="$restart" nrfld/fgmres_max_iterations="$max_iterations" \
    nrfld/fgmres_mg_vcycles="$mg_vcycles" \
    > "$job/run.log" 2>&1
code=$?
set -e
ended=$(date +%s)
limit=0; fatal=0
if grep -q 'Terminating on cycle limit' "$job/run.log"; then limit=1; fi
if grep -q 'FATAL ERROR' "$job/run.log"; then fatal=1; fi
printf 'case,exit_code,cycle_limit,has_fatal,elapsed_seconds\n%s,%d,%d,%d,%d\n' \
  "$label" "$code" "$limit" "$fatal" "$((ended-started))" > "$job/status.csv"
if (( code != 0 || limit != 1 || fatal != 0 )); then exit 1; fi
