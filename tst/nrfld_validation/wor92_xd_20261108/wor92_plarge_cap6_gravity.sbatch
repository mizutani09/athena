#!/bin/bash -l
#SBATCH --partition=P-large-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=wor92_cap6
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/long/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/long/job_${SLURM_JOB_ID}"
mkdir -p "$case_dir/output"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
hst="$case_dir/output/cfl030_flux_precision16_local.hst"
reader="$base/scripts/read_athena_restart_metadata_py36.py"
start_epoch=$(date +%s)
stop_epoch=$((start_epoch + 23*3600 + 30*60))
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_BASE_COMMIT=%s\n' 873a1b4c3bda584b52e1cae2b41e333ddb2f793a
  printf 'JOB_ID=%s\n' "$SLURM_JOB_ID"
  printf 'NTASKS=%s\n' "$SLURM_NTASKS"
  printf 'STOP_POLICY=jeans_min_below_16_or_opacity_domain_or_tlim_or_wall_after_each_10_cycle_segment\n'
  module list 2>&1
  sha256sum "$base/source_restart_gravity_trial/bin/athena" "$base/bundle/wor92_core_symmetry_cycle_output_gravity_20261107.in" "$base/bundle/data/eos/protostellar_tomida2015_material.tab" "$base/bundle/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
  sha256sum "$base/source_restart_gravity_trial/src/main.cpp" "$base/source_restart_gravity_trial/src/gravity/mg_gravity.cpp" "$base/source_restart_gravity_trial/src/gravity/mg_gravity.hpp" "$base/source_restart_gravity_trial/src/mesh/mesh.cpp"
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
    current=0
    input_args=(-i "$base/bundle/wor92_core_symmetry_cycle_output_gravity_20261107.in")
  fi
  remaining=$((stop_epoch - $(date +%s)))
  if (( remaining < 900 )); then
    printf 'STOP_WALL_BUDGET cycle=%s\n' "$current" | tee -a "$case_dir/status.txt"
    break
  fi
  runtime=$((remaining - 600))
  if (( runtime > 7200 )); then runtime=7200; fi
  printf -v wallarg '%02d:%02d:%02d' $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
  target=$((current + 10))
  printf 'SEGMENT=%d START_CYCLE=%d TARGET_CYCLE=%d WALL=%s\n' "$segment" "$current" "$target" "$wallarg" | tee -a "$case_dir/status.txt"
  set +e
  mpirun -np 64 "$base/source_restart_gravity_trial/bin/athena" \
    "${input_args[@]}" -d "$case_dir/output" -t "$wallarg" \
    mesh/numlevel=6 time/cfl_number=0.3 \
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
