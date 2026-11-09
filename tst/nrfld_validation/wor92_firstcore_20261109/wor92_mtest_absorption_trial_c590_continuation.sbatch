#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=320
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor92_absorption_c590
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/long/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/long/job_${SLURM_JOB_ID}"
mkdir -p "$case_dir/output"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm:ofi
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
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
trial="$base/debug/semenov_planck_absorption_trial.h5"
if [[ $(sha256sum "$trial" | cut -d' ' -f1) != 347baa6e7b0d4303eca1abf762b2fde5042137e42739ef65961eeb68b88dad05 ]]; then
  printf 'FAIL_INVALID_ABSORPTION_TRIAL_TABLE\n' > "$case_dir/status.txt"
  exit 1
fi
initial_rst="$base/test/job_338150/output/cfl030_flux_precision16_local.final.rst"
hst="$case_dir/output/cfl030_flux_precision16_local.hst"
reader="$base/scripts/read_athena_restart_metadata_py36.py"
if [[ ! -s "$initial_rst" ]] ||
   [[ $(python3 "$reader" "$initial_rst" --field cycle) != 590 ]]; then
  printf 'FAIL_INVALID_C590_CHECKPOINT path=%s\n' "$initial_rst" > "$case_dir/status.txt"
  exit 1
fi
start_epoch=$(date +%s)
stop_epoch=$((start_epoch + 27*60))
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_BASE_COMMIT=%s\n' f9dc679330bf76dcef0addcbc83b02d5d9121107
  printf 'JOB_ID=%s\n' "$SLURM_JOB_ID"
  printf 'PREVIOUS_JOB=338150\n'
  printf 'OPACITY_SENSITIVITY_BRANCH=semenov_planck_absorption_trial\n'
  sha256sum "$trial"
  sha256sum "$base/source_newton_converged_trial/src/newton_raphson/Newton_Raphson_driver.cpp"
  printf 'INITIAL_RESTART_REALPATH=%s\n' "$(readlink -f "$initial_rst")"
  printf 'NTASKS=%s\n' "$SLURM_NTASKS"
  printf 'STOP_POLICY=first_core_density_1e-10_or_jeans_below_16_or_opacity_domain_or_tlim_or_wall_after_each_10_cycle_segment\n'
  module list 2>&1
  sha256sum "$initial_rst"
  sha256sum "$base/source_newton_converged_trial/bin/athena" "$base/bundle/wor92_core_symmetry_cycle_output_gravity_20261107.in" "$base/bundle/data/eos/protostellar_tomida2015_material.tab" "$base/bundle/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
  sha256sum "$base/source_newton_converged_trial/src/main.cpp" "$base/source_newton_converged_trial/src/gravity/mg_gravity.cpp" "$base/source_newton_converged_trial/src/gravity/mg_gravity.hpp" "$base/source_newton_converged_trial/src/mesh/mesh.cpp"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
printf 'segment,start_cycle,end_cycle,sim_time,jeans_min,rho_max,tgas_max,gas_mass,erad_total,level_max,exit_code,reason\n' > "$case_dir/progress.csv"
cd "$base/bundle"
segment=0
while true; do
  if [[ -f "$rst" ]]; then
    current=$(python3 "$reader" "$rst" --field cycle)
    input_args=(-r "$rst")
  else
    current=$(python3 "$reader" "$initial_rst" --field cycle)
    input_args=(-r "$initial_rst")
  fi
  remaining=$((stop_epoch - $(date +%s)))
  if (( remaining < 180 )); then
    printf 'STOP_WALL_BUDGET cycle=%s\n' "$current" | tee -a "$case_dir/status.txt"
    break
  fi
  runtime=$((remaining - 120))
  if (( runtime > 1200 )); then runtime=1200; fi
  printf -v wallarg '%02d:%02d:%02d' $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
  target=$((current + 10))
  printf 'SEGMENT=%d START_CYCLE=%d TARGET_CYCLE=%d WALL=%s\n' "$segment" "$current" "$target" "$wallarg" | tee -a "$case_dir/status.txt"
  set +e
  mpirun -np "$SLURM_NTASKS" "$base/source_newton_converged_trial/bin/athena" \
    "${input_args[@]}" -d "$case_dir/output" -t "$wallarg" \
    mesh/numlevel=12 time/cfl_number=0.3 fld/opacity_table_file="$trial" nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
    time/tlim=10000.0 time/nlim="$target" \
    output2/dcycle=10 output4/dcycle=10 \
    > "$case_dir/segment_$(printf '%03d' "$segment").log" 2>&1
  code=$?
  set -e
  if [[ ! -f "$rst" ]]; then
    printf 'FAIL_NO_FINAL_RESTART segment=%d exit=%d\n' "$segment" "$code" | tee -a "$case_dir/status.txt"
    exit 1
  fi
  after=$(python3 "$reader" "$rst" --field cycle)
  read -r hsttime jeans rho level tgas gasmass erad < <(awk 'END {print $1,$35,$16,$36,$20,$23,$34}' "$hst")
  simtime=$(python3 "$reader" "$rst" --field time)
  reason=CONTINUE
  if (( code != 0 )) || grep -q 'FATAL ERROR' "$case_dir/segment_$(printf '%03d' "$segment").log"; then
    reason=FAIL_SOLVER
  elif (( after <= current )); then
    reason=FAIL_NO_PROGRESS
  elif [[ -z "${jeans:-}" || -z "${simtime:-}" || -z "${rho:-}" || -z "${tgas:-}" ]]; then
    reason=FAIL_HISTORY
  elif ! grep -q '^FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0 negative_results=0 zero_results=0 nonfinite_results=0$' "$case_dir/segment_$(printf '%03d' "$segment").log"; then
    reason=STOP_OPACITY_DOMAIN
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1e-7 or float(sys.argv[2]) >= 9900 else 1)' "$rho" "$tgas"; then
    reason=STOP_OPACITY_DOMAIN
  elif python3 -c 'import math,sys; x=float(sys.argv[1]); sys.exit(0 if not math.isfinite(x) else 1)' "$jeans"; then
    reason=FAIL_HISTORY
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 16 else 1)' "$jeans"; then
    reason=STOP_UNDERRESOLVED
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1e-10 else 1)' "$rho"; then
    reason=FIRST_CORE_DENSITY_CANDIDATE
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 10000.0-1e-8 else 1)' "$simtime"; then
    reason=TARGET_TIME
  fi
  printf '%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%d,%s\n' "$segment" "$current" "$after" "$simtime" "$jeans" "$rho" "$tgas" "$gasmass" "$erad" "$level" "$code" "$reason" >> "$case_dir/progress.csv"
  printf 'SEGMENT=%d END_CYCLE=%d SIM_TIME=%s JEANS_MIN=%s RHO_MAX=%s TGAS_MAX=%s LEVEL=%s EXIT=%d REASON=%s\n' "$segment" "$after" "$simtime" "$jeans" "$rho" "$tgas" "$level" "$code" "$reason" | tee -a "$case_dir/status.txt"
  if [[ "$reason" == FAIL_* ]]; then exit 1; fi
  if [[ "$reason" != CONTINUE ]]; then break; fi
  segment=$((segment + 1))
done
sha256sum "$rst" > "$case_dir/final_restart.sha256"
date '+END=%Y-%m-%d %H:%M:%S %Z' | tee -a "$case_dir/status.txt"
