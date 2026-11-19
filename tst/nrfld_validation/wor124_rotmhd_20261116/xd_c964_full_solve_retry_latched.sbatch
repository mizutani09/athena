#!/bin/bash -l
#SBATCH --partition=M-large-a
#SBATCH --nodes=5
#SBATCH --ntasks=320
#SBATCH --ntasks-per-node=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor124_retry_c970
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_full_solve_retry_20261119"
reader="$base/scripts/read_restart.py"
audit="$base/scripts/collapse_opacity_domain_audit.py"
rst="$base/rootfix_fgmres_refinement/job_343340/output/wor124_refine_c960_cfl015.00120.rst"
eos_table="$source_dir/data/eos/protostellar_tomida2015_material.tab"
opacity_table="$source_dir/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
job="$base/rootfix_full_solve_retry_latched/job_${SLURM_JOB_ID}"
mkdir -p "$job/output"

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
unset ATHENA_WOR92_INEXACT_TRUE_CAP
unset ATHENA_WOR92_INEXACT_STAGNATION_RATIO
unset ATHENA_NRFLD_FGMRES_REFINEMENT_STEPS
unset ATHENA_NRFLD_FGMRES_REFINEMENT_ETA_TRIGGER
unset ATHENA_NRFLD_TANGENTIAL_POSITIVITY_FALLBACK
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2

if [[ ! -s "$rst" ]] || [[ $(python3 "$reader" "$rst" --field cycle) != 964 ]]; then
  printf 'FAIL_INVALID_CHECKPOINT path=%s\n' "$rst" > "$job/status.txt"
  exit 1
fi
if [[ ! -s "$eos_table" ]] || [[ ! -s "$opacity_table" ]]; then
  printf 'FAIL_MISSING_TABLE eos=%s opacity=%s\n' "$eos_table" "$opacity_table" \
    > "$job/status.txt"
  exit 1
fi

cd "$source_dir"
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table -b --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -hdf5 -mpi --ccmd=icpx --mpiccmd=mpiicpx --cflag=-O3 \
  > "$job/configure.log" 2>&1
make -j8 > "$job/build.log" 2>&1

{
  date -Is
  printf 'git_commit=%s ranks=%s restart_cycle=964 target=970 cfl=0.15 tangential_reconstruction=1 full_solve_retry=1 latch_after_success=1 nr_max_iterations=8\n' \
    1a5ef599c "$SLURM_NTASKS"
  sha256sum "$rst" "$eos_table" "$opacity_table" "$source_dir/bin/athena" \
    "$source_dir/src/utils/nrfld_prototype_flags.hpp" \
    "$source_dir/src/newton_raphson/Newton_Raphson_driver.cpp"
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"

started=$(date +%s)
set +e
/usr/bin/time -v -o "$job/resources.txt" \
  mpirun -np "$SLURM_NTASKS" "$source_dir/bin/athena" \
    -r "$rst" -d "$job/output" \
    job/problem_id=wor124_c964_full_solve_retry_latched \
    time/nlim=970 time/ncycle_out=1 time/cfl_number=0.15 time/xorder=2 \
    mesh/numlevel=12 output2/dcycle=1 output4/dcycle=1 \
    nrfld/diagnostic_verbosity=1 nrfld/nr_max_iterations=8 \
    nrfld/jacobian_check=false nrfld/jacobian_check_each_cycle=false \
    nrfld/backward_error_convergence=false \
    nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart=2000 \
    nrfld/fgmres_max_iterations=2500 nrfld/fgmres_mg_vcycles=5 \
    > "$job/run.log" 2>&1
code=$?
set -e
elapsed=$(($(date +%s)-started))
limit=0
fatal=0
retry_count=$(grep -c '^\[NR_TANGENTIAL_FULL_RETRY\].*retry_status=converged.*two_point_latched=1' "$job/run.log" || true)
if grep -q 'Terminating on cycle limit' "$job/run.log"; then limit=1; fi
if grep -q 'FATAL ERROR' "$job/run.log"; then fatal=1; fi
printf 'case,exit_code,cycle_limit,has_fatal,retry_success_count,elapsed_seconds\n' > "$job/status.csv"
printf 'c964_c970_full_solve_retry_latched,%s,%s,%s,%s,%s\n' \
  "$code" "$limit" "$fatal" "$retry_count" "$elapsed" >> "$job/status.csv"
if (( code != 0 || limit != 1 || fatal != 0 || retry_count != 1 )); then exit 1; fi

latest_hdf=$(find "$job/output" -maxdepth 1 -type f -name '*.prim.*.athdf' \
  -printf '%p\n' | sort | tail -n 1)
python3 "$audit" "$latest_hdf" "$opacity_table" --backend h5dump \
  > "$job/opacity_domain_audit.json"
sha256sum "$job/output"/*.final.rst > "$job/final_restart.sha256"
grep -E '^\[NR_TANGENTIAL_FULL_RETRY\]|^\[NR\] status=|^\[LINEAR_FGMRES\]' \
  "$job/run.log" > "$job/iteration_summary.log" || true
date -Is > "$job/completed.txt"
