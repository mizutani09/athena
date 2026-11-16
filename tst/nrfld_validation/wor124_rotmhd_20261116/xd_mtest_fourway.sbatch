#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor124_4way
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="${WOR124_SOURCE_DIR:-$base/source}"
job="$base/${WOR124_JOB_SUBDIR:-mtest}/job_${SLURM_JOB_ID}"
mkdir -p "$job"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm
export OMP_NUM_THREADS=1
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_INEXACT_TRUE_CAP="${WOR124_INEXACT_CAP:-1e-7}"
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
if [[ "${WOR124_ABS_DIAG:-0}" == 1 ]]; then
  export ATHENA_WOR124_ABS_NORM_DIAGNOSTICS=1
fi

cd "$source_dir"
{
  date -Is
  printf 'source_commit=%s\n' "$(cat "$base/source_commit.txt")"
  printf 'source_tree=%s\n' "$(cat "$base/source_tree.txt")"
  printf 'binary_sha256='; sha256sum bin/athena
  printf 'input_sha256='; sha256sum tst/nrfld_validation/wor124_rotmhd_20261116/input_table_mhd_pilot.in
  sha256sum data/eos/protostellar_tomida2015_material.tab data/opacity/protostellar_semenov2003_nrm_c_a.h5
  module list 2>&1
  env | grep '^ATHENA_' | sort
  env | grep '^WOR124_' | sort
} > "$job/provenance.txt"
printf 'case,exit_code,cycle_limit,has_fatal,elapsed_seconds\n' > "$job/status.csv"

failed=0
cases=${WOR124_CASES:-"base rotation magnetic combined"}
cycles=${WOR124_CYCLES:-5}
for case in $cases; do
  rotation=false; magnetic=false
  case "$case" in
    rotation) rotation=true ;;
    magnetic) magnetic=true ;;
    combined) rotation=true; magnetic=true ;;
  esac
  dir="$job/$case"
  mkdir -p "$dir/output"
  started=$(date +%s)
  set +e
  /usr/bin/time -v -o "$dir/resources.txt" \
    mpirun -np "$SLURM_NTASKS" "$source_dir/bin/athena" \
      -i "$source_dir/tst/nrfld_validation/wor124_rotmhd_20261116/input_table_mhd_pilot.in" \
      -d "$dir/output" \
      job/problem_id="$case" \
      time/nlim="$cycles" time/ncycle_out="$cycles" \
      time/cfl_number="${WOR124_CFL:-0.3}" \
      mesh/numlevel="${WOR124_NUMLEVEL:-2}" \
      output2/dcycle="${WOR124_OUTPUT_CYCLES:-$cycles}" output2/variable=cons,prim,uov \
      output4/dcycle="${WOR124_OUTPUT_CYCLES:-$cycles}" \
      nrfld/diagnostic_verbosity=0 nrfld/jacobian_check=false \
      nrfld/jacobian_check_each_cycle=false \
      nrfld/fgmres_threshold="${WOR124_FGMRES_THRESHOLD:-2e-8}" \
      nrfld/fgmres_restart="${WOR124_FGMRES_RESTART:-80}" \
      nrfld/fgmres_max_iterations="${WOR124_FGMRES_MAX:-600}" \
      nrfld/fgmres_mg_vcycles="${WOR124_MG_VCYCLES:-3}" \
      problem/enable_rotation="$rotation" \
      problem/enable_magnetic_field="$magnetic" \
      problem/enable_tabular_mhd_pilot="$magnetic" \
      problem/B0="${WOR124_B0:-1.29e-4}" \
      fld/radiation_pressure_in_riemann="${WOR124_RAD_PRESSURE_RIEMANN:-true}" \
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
