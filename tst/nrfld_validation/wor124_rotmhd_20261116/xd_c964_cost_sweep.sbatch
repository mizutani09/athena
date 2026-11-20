#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=5
#SBATCH --ntasks=320
#SBATCH --ntasks-per-node=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor124_cost_sweep
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_solver_freeze_tuning_20261120"
table_dir="$base/source_full_solve_early_retry_20261119"
reader="$base/scripts/read_restart.py"
audit="$base/scripts/collapse_opacity_domain_audit.py"
rst="$base/rootfix_fgmres_refinement/job_343340/output/wor124_refine_c960_cfl015.00120.rst"
eos_table="$table_dir/data/eos/protostellar_tomida2015_material.tab"
opacity_table="$table_dir/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
job="$base/solver_freeze_cost_sweep/job_${SLURM_JOB_ID}"
mkdir -p "$job"

module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm:ofi OMP_NUM_THREADS=1
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_TANGENTIAL_FULL_SOLVE_RETRY=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
unset ATHENA_WOR92_INEXACT_TRUE_CAP
unset ATHENA_WOR92_INEXACT_STAGNATION_RATIO
unset ATHENA_NRFLD_FGMRES_REFINEMENT_STEPS
unset ATHENA_NRFLD_FGMRES_REFINEMENT_ETA_TRIGGER
unset ATHENA_NRFLD_TANGENTIAL_POSITIVITY_FALLBACK

if [[ ! -s "$rst" ]] || [[ $(python3 "$reader" "$rst" --field cycle) != 964 ]]; then
  printf 'FAIL_INVALID_CHECKPOINT path=%s\n' "$rst" > "$job/status.txt"
  exit 1
fi
if [[ ! -s "$eos_table" ]] || [[ ! -s "$opacity_table" ]]; then
  printf 'FAIL_MISSING_TABLE eos=%s opacity=%s\n' "$eos_table" "$opacity_table" \
    > "$job/status.txt"
  exit 1
fi
mkdir -p "$source_dir/data/eos" "$source_dir/data/opacity"
ln -sfn "$eos_table" "$source_dir/data/eos/protostellar_tomida2015_material.tab"
ln -sfn "$opacity_table" "$source_dir/data/opacity/protostellar_semenov2003_nrm_c_a.h5"

cd "$source_dir"
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table -b --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -hdf5 -mpi --ccmd=icpx --mpiccmd=mpiicpx --cflag=-O3 \
  > "$job/configure.log" 2>&1
make -j8 > "$job/build.log" 2>&1

{
  date -Is
  printf 'source_branch=release/protostellar-collapse-solver-v2-20261120 source_commit=67b2d3246 ranks=%s restart_cycle=964 target=965 cfl=0.15 full_solve_retry=1\n' \
    "$SLURM_NTASKS"
  sha256sum "$rst" "$eos_table" "$opacity_table" "$source_dir/bin/athena" \
    "$source_dir/src/newton_raphson/Newton_Raphson_driver.cpp"
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"

printf 'case,restart,max_iterations,mg_vcycles,exit_code,cycle_limit,has_fatal,retry_converged,direction_rejected,elapsed_seconds,fgmres_solves,fgmres_iterations,max_fgmres_iterations\n' \
  > "$job/status.csv"

run_case() {
  local name=$1 restart=$2 max_iterations=$3 mg_vcycles=$4
  local case_dir="$job/$name"
  mkdir -p "$case_dir/output"
  local started code elapsed limit=0 fatal=0 retry=0 direction=0
  started=$(date +%s)
  set +e
  /usr/bin/time -v -o "$case_dir/resources.txt" \
    mpirun -np "$SLURM_NTASKS" "$source_dir/bin/athena" \
      -r "$rst" -d "$case_dir/output" \
      job/problem_id="wor124_c964_${name}" \
      time/nlim=965 time/ncycle_out=1 time/cfl_number=0.15 time/xorder=2 \
      mesh/numlevel=12 output2/dcycle=1 output4/dcycle=1 \
      nrfld/diagnostic_verbosity=1 nrfld/nr_max_iterations=8 \
      nrfld/jacobian_check=false nrfld/jacobian_check_each_cycle=false \
      nrfld/backward_error_convergence=false \
      nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart="$restart" \
      nrfld/fgmres_max_iterations="$max_iterations" \
      nrfld/fgmres_mg_vcycles="$mg_vcycles" \
      > "$case_dir/run.log" 2>&1
  code=$?
  set -e
  elapsed=$(($(date +%s)-started))
  if grep -q 'Terminating on cycle limit' "$case_dir/run.log"; then limit=1; fi
  if grep -q 'FATAL ERROR' "$case_dir/run.log"; then fatal=1; fi
  if grep -q '^\[NR_TANGENTIAL_FULL_RETRY\].*retry_status=converged' "$case_dir/run.log"; then retry=1; fi
  if grep -q '^\[NR_TANGENTIAL_DIRECTION_REJECTED\]' "$case_dir/run.log"; then direction=1; fi
  local stats
  stats=$(awk '/^\[LINEAR_FGMRES\]/{count++; for(i=1;i<=NF;i++) if($i ~ /^iterations=/){split($i,a,"="); total+=a[2]; if(a[2]>maximum) maximum=a[2]}} END{printf "%d,%d,%d",count,total,maximum}' "$case_dir/run.log")
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$name" "$restart" "$max_iterations" "$mg_vcycles" "$code" "$limit" \
    "$fatal" "$retry" "$direction" "$elapsed" "$stats" >> "$job/status.csv"
  if (( code == 0 && limit == 1 && fatal == 0 && retry == 1 )); then
    local latest_hdf
    latest_hdf=$(find "$case_dir/output" -maxdepth 1 -type f -name '*.prim.*.athdf' \
      -printf '%p\n' | sort | tail -n 1)
    python3 "$audit" "$latest_hdf" "$opacity_table" --backend h5dump \
      > "$case_dir/opacity_domain_audit.json"
    sha256sum "$case_dir/output"/*.final.rst > "$case_dir/final_restart.sha256"
  fi
}

# The first-core collaboration input currently uses restart 40 / max 300 /
# MG2.  Compare one cheaper MG application and a wider restart without changing
# the strict residual target.  The validated high-cost c964 setup is recorded
# separately as job 343472 and is not repeated here.
run_case r40_mg1 40 300 1
run_case r40_mg2 40 300 2
run_case r160_mg2 160 300 2

date -Is > "$job/completed.txt"
if awk -F, 'NR>1 && ($5 != 0 || $6 != 1 || $7 != 0 || $8 != 1){bad=1} END{exit bad}' \
    "$job/status.csv"; then
  exit 0
fi
exit 1
