#!/bin/bash -l
#SBATCH --partition=M-large-a
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=320
#SBATCH --mem=110G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=wor92_mayer5_uniform_firstcore
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_uniform_c0_20261110/slurm-%j.out

set -euo pipefail
: "${PREFLIGHT_JOB_ID:?submit with PREFLIGHT_JOB_ID and afterok dependency}"
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/ideal_mayer5_uniform_c0_20261110/job_${SLURM_JOB_ID}"
preflight_dir="$base/ideal_mayer5_uniform_c0_20261110/job_${PREFLIGHT_JOB_ID}"
seed_job_id=${SEED_JOB_ID:-$PREFLIGHT_JOB_ID}
seed_dir="$base/ideal_mayer5_uniform_c0_20261110/job_${seed_job_id}"
binary="$base/source_wor118_ideal_localgas_20261110/bin/athena"
input="$base/ideal_mayer5_uniform_c0_20261110/wor92_mayer5_uniform_c0_20261110.in"
reader="$base/scripts/read_athena_restart_metadata_py36.py"
expected_binary_sha=56d379d96cea55cb50497ccc4a7ea9a10897784f82f8f48a6b9817671b62993d
expected_input_sha=2d7198a19316a462e3ca4d44b41f0eec79795239f084e5184d50962c8224474f
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
printf '%s  %s\n' "$expected_binary_sha" "$binary" | sha256sum -c -
printf '%s  %s\n' "$expected_input_sha" "$input" | sha256sum -c -
if ! grep -q '^PASS_C0_TO_C10' "$preflight_dir/status.txt"; then
  printf 'FAIL_PREFLIGHT_NOT_PASSED\n' > "$case_dir/status.txt"
  exit 1
fi
if [[ "$seed_job_id" != "$PREFLIGHT_JOB_ID" ]] &&
   ! grep -q '^STOP_WALL_BUDGET\|REASON=FIRST_CORE_DENSITY_CANDIDATE' "$seed_dir/status.txt"; then
  printf 'FAIL_SEED_JOB_NOT_VALIDATED job=%s\n' "$seed_job_id" > "$case_dir/status.txt"
  exit 1
fi
if grep -q 'REASON=FIRST_CORE_DENSITY_CANDIDATE' "$seed_dir/status.txt"; then
  printf 'SKIP_FIRST_CORE_ALREADY_REACHED seed_job=%s\n' "$seed_job_id" > "$case_dir/status.txt"
  exit 0
fi
seed="$case_dir/seed.rst"
cp -- "$seed_dir/output/cfl030_flux_precision16_local.final.rst" "$seed"
cmp -- "$seed_dir/output/cfl030_flux_precision16_local.final.rst" "$seed"
if [[ "$seed_job_id" != "$PREFLIGHT_JOB_ID" ]]; then
  (cd "$seed_dir/output" && sha256sum -c "$seed_dir/final_restart.sha256")
fi
seed_cycle=$(python3 "$reader" "$seed" --field cycle)
seed_blocks=$(python3 "$reader" "$seed" --field meshblocks)
if (( seed_cycle < 10 || seed_blocks < SLURM_NTASKS )); then
  printf 'FAIL_INVALID_SEED_CYCLE_OR_BLOCKS cycle=%s meshblocks=%s\n' "$seed_cycle" "$seed_blocks" > "$case_dir/status.txt"
  exit 1
fi
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_COMMIT=582acd5999c1a5681ab1bc1d1af292676642ed40\n'
  printf 'START=INITIAL_C0_VIA_PREFLIGHT_JOB_%s\n' "$PREFLIGHT_JOB_ID"
  printf 'SEED_JOB_ID=%s\n' "$seed_job_id"
  printf 'SEED_CYCLE=%s\n' "$seed_cycle"
  printf 'SEED_MESHBLOCKS=%s\n' "$seed_blocks"
  printf 'NTASKS=%s\n' "$SLURM_NTASKS"
  printf 'STOP=RHO_MAX_1E-10_OR_ANY_STEP_PRESSURE_FLOOR_OR_JEANS_BELOW_16_OR_WALL_BUDGET\n'
  module list 2>&1
  sha256sum "$seed" "$binary" "$input"
  cat "$base/ideal_mayer5_uniform_c0_20261110/build_provenance.txt"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
printf 'segment,start_cycle,end_cycle,sim_time,jeans_min,pressure_min,rho_max,tgas_max,gas_mass,erad_total,level_max,exit_code,reason\n' > "$case_dir/progress.csv"
start_epoch=$(date +%s)
stop_epoch=$((start_epoch + ${MAX_WALL_SECONDS:-84600}))
stop_margin=${STOP_MARGIN_SECONDS:-900}
athena_margin=${ATHENA_MARGIN_SECONDS:-600}
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
hst="$case_dir/output/cfl030_flux_precision16_local.hst"
cd "$base/bundle"
segment=0
while true; do
  if [[ -s "$rst" ]]; then
    current=$(python3 "$reader" "$rst" --field cycle)
    source_rst="$rst"
  else
    current=$seed_cycle
    source_rst="$seed"
  fi
  remaining=$((stop_epoch - $(date +%s)))
  if (( remaining < stop_margin )); then
    printf 'STOP_WALL_BUDGET cycle=%s\n' "$current" | tee -a "$case_dir/status.txt"
    break
  fi
  runtime=$((remaining - athena_margin))
  if (( runtime > 7200 )); then runtime=7200; fi
  printf -v wallarg '%02d:%02d:%02d' $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
  target=$((current + 10))
  printf 'SEGMENT=%d START_CYCLE=%d TARGET_CYCLE=%d\n' "$segment" "$current" "$target" | tee -a "$case_dir/status.txt"
  set +e
  mpirun -np "$SLURM_NTASKS" "$binary" -r "$source_rst" -d "$case_dir/output" -t "$wallarg" \
    mesh/numlevel=12 time/cfl_number=0.3 time/tlim=10000.0 time/nlim="$target" \
    nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
    output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
    output2/dcycle=10 output4/dcycle=10 > "$case_dir/segment_$(printf '%03d' "$segment").log" 2>&1
  code=$?
  set -e
  if [[ ! -s "$rst" ]]; then
    printf 'FAIL_NO_FINAL_RESTART segment=%d exit=%d\n' "$segment" "$code" | tee -a "$case_dir/status.txt"
    exit 1
  fi
  after=$(python3 "$reader" "$rst" --field cycle)
  read -r hsttime jeans rho level tgas gasmass erad pmin < <(awk 'END {print $1,$35,$16,$36,$20,$23,$34,$30}' "$hst")
  first_bad_step=$(awk 'NF==39 && ($30<=1e-18 || $35<16) {print NR,$1,$30,$35; exit}' "$hst")
  simtime=$(python3 "$reader" "$rst" --field time)
  reason=CONTINUE
  if (( code != 0 )) || grep -q 'FATAL ERROR' "$case_dir/segment_$(printf '%03d' "$segment").log"; then
    reason=FAIL_SOLVER
  elif (( after <= current )); then
    reason=FAIL_NO_PROGRESS
  elif [[ -z "${jeans:-}" || -z "${simtime:-}" || -z "${rho:-}" || -z "${tgas:-}" ]]; then
    reason=FAIL_HISTORY
  elif python3 -c 'import math,sys; v=[float(x) for x in sys.argv[1:]]; sys.exit(0 if any(not math.isfinite(x) or x<=0 for x in v) else 1)' "$jeans" "$rho" "$tgas" "$gasmass"; then
    reason=FAIL_NONFINITE_OR_NONPOSITIVE_HISTORY
  elif [[ -n "$first_bad_step" ]]; then
    reason=STOP_STEP_PRESSURE_FLOOR_OR_JEANS
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1e-10 else 1)' "$rho"; then
    reason=FIRST_CORE_DENSITY_CANDIDATE
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 10000.0-1e-8 else 1)' "$simtime"; then
    reason=TARGET_TIME
  fi
  printf '%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,%d,%s\n' "$segment" "$current" "$after" "$simtime" "$jeans" "$pmin" "$rho" "$tgas" "$gasmass" "$erad" "$level" "$code" "$reason" >> "$case_dir/progress.csv"
  printf 'SEGMENT=%d END_CYCLE=%d SIM_TIME=%s JEANS_MIN=%s PRESSURE_MIN=%s RHO_MAX=%s TGAS_MAX=%s LEVEL=%s EXIT=%d REASON=%s\n' "$segment" "$after" "$simtime" "$jeans" "$pmin" "$rho" "$tgas" "$level" "$code" "$reason" | tee -a "$case_dir/status.txt"
  if [[ -n "$first_bad_step" ]]; then
    printf 'FIRST_BAD_HISTORY_ROW=%s\n' "$first_bad_step" | tee -a "$case_dir/status.txt"
  fi
  if [[ "$reason" == FAIL_* ]]; then exit 1; fi
  if [[ "$reason" != CONTINUE ]]; then break; fi
  segment=$((segment + 1))
done
sha256sum "$rst" > "$case_dir/final_restart.sha256"
date '+END=%Y-%m-%d %H:%M:%S %Z' | tee -a "$case_dir/status.txt"
if [[ "${reason:-}" == STOP_STEP_PRESSURE_FLOOR_OR_JEANS ]]; then exit 2; fi
