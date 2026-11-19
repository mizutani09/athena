#!/bin/bash -l
#SBATCH --partition=M-large-a
#SBATCH --nodes=5
#SBATCH --ntasks=320
#SBATCH --ntasks-per-node=64
#SBATCH --mem=110G
#SBATCH --time=24:00:00
#SBATCH --job-name=wor124_B129_cfl015_long
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_floor_trial"
reader="$base/scripts/read_restart.py"
opacity_audit="$base/scripts/collapse_opacity_domain_audit.py"
gate_job="${WOR124_GATE_JOB:?set WOR124_GATE_JOB to the c880--890 CFL gate}"
gate="$base/strong_field_c880_cfl015_sustain/job_${gate_job}"
gate_status="$gate/status.csv"
initial_rst="${WOR124_INITIAL_RST:-$gate/output/wor124_B129_c880_cfl015_sustain.final.rst}"
initial_cycle="${WOR124_INITIAL_CYCLE:-890}"
label="${WOR124_LABEL:-combined_B129_strict_r2000m2500mg5_cfl015_amr12_5node}"
job="$base/high_amr_long/${label}_job_${SLURM_JOB_ID}"
mkdir -p "$job/output"
rst="$job/output/$label.final.rst"
hst="$job/output/$label.hst"

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
unset ATHENA_WOR92_INEXACT_TRUE_CAP
unset ATHENA_WOR92_INEXACT_STAGNATION_RATIO
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2

if [[ ! -s "$gate_status" ]] ||
   ! grep -q '^cfl015_strict_r2000_max2500_mg5,0,1,0,' "$gate_status"; then
  printf 'FAIL_GATE job=%s status=%s\n' "$gate_job" "$gate_status" \
    > "$job/status.txt"
  exit 1
fi
if [[ ! -s "$initial_rst" ]] ||
   [[ $(python3 "$reader" "$initial_rst" --field cycle) != "$initial_cycle" ]]; then
  printf 'FAIL_INVALID_CHECKPOINT expected_cycle=%s path=%s\n' \
    "$initial_cycle" "$initial_rst" > "$job/status.txt"
  exit 1
fi

cd "$source_dir"
started=$(date +%s)
stop_epoch=$((started + 23*3600 + 30*60))
{
  date -Is
  printf 'source_commit=%s\n' "$(cat "$base/source_commit.txt")"
  printf 'source_dir=%s\n' "$source_dir"
  printf 'gate_job=%s gate_status=%s\n' "$gate_job" "$gate_status"
  cat "$gate_status"
  printf 'label=%s ranks=%s initial_cycle=%s max_amr_level=12 cfl=0.15 fgmres=2000/2500_mg5 threshold=2e-8 inexact_cap=disabled backward_error=false\n' \
    "$label" "$SLURM_NTASKS" "$initial_cycle"
  sha256sum "$initial_rst" "$reader" "$opacity_audit" bin/athena
  sha256sum data/eos/protostellar_tomida2015_material.tab \
            data/opacity/protostellar_semenov2003_nrm_c_a.h5
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"

printf 'segment,start_cycle,end_cycle,meshblocks,time,jeans_min,rho_max,Tgas_max,gas_mass,divB_max,Lz,Eint_gas_cgs,Ekin_gas_cgs,Emag_cgs,Erad_tot_cgs,level_max,elapsed_seconds,exit_code,reason\n' \
  > "$job/progress.csv"

segment=0
while true; do
  input_rst="$initial_rst"
  if [[ -s "$rst" ]]; then input_rst="$rst"; fi
  current=$(python3 "$reader" "$input_rst" --field cycle)
  remaining=$((stop_epoch - $(date +%s)))
  if (( remaining < 900 )); then
    printf 'STOP_WALL_BUDGET cycle=%s\n' "$current" | tee -a "$job/status.txt"
    break
  fi
  runtime=$((remaining - 600))
  if (( runtime > 7200 )); then runtime=7200; fi
  printf -v wallarg '%02d:%02d:%02d' \
    $((runtime/3600)) $(((runtime%3600)/60)) $((runtime%60))
  target=$((current + 10))
  log="$job/segment_$(printf '%03d' "$segment").log"
  printf 'SEGMENT=%d START=%d TARGET=%d WALL=%s\n' \
    "$segment" "$current" "$target" "$wallarg" | tee -a "$job/status.txt"
  segstart=$(date +%s)
  set +e
  mpirun -np "$SLURM_NTASKS" bin/athena \
    -r "$input_rst" -d "$job/output" -t "$wallarg" \
    job/problem_id="$label" mesh/numlevel=12 \
    time/cfl_number=0.15 time/tlim=10000.0 time/nlim="$target" \
    time/ncycle_out=10 output2/dcycle=10 output2/variable=cons,prim,uov \
    output4/dcycle=10 nrfld/diagnostic_verbosity=0 \
    nrfld/jacobian_check=false nrfld/jacobian_check_each_cycle=false \
    nrfld/backward_error_convergence=false \
    nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart=2000 \
    nrfld/fgmres_max_iterations=2500 nrfld/fgmres_mg_vcycles=5 \
    > "$log" 2>&1
  code=$?
  set -e
  elapsed=$(($(date +%s)-segstart))
  if [[ ! -s "$rst" ]]; then
    printf 'FAIL_NO_FINAL_RESTART segment=%d exit=%d\n' "$segment" "$code" \
      | tee -a "$job/status.txt"
    exit 1
  fi
  after=$(python3 "$reader" "$rst" --field cycle)
  blocks=$(python3 "$reader" "$rst" --field meshblocks)
  read -r simtime jeans rho tgas gasmass divb lz eint ekin emag erad level < <(
    awk 'END {print $1,$35,$16,$20,$23,$24,$25,$31,$32,$33,$34,$36}' "$hst")
  reason=CONTINUE
  opacity_line=$(grep '^FLD_OPACITY_DIAGNOSTICS ' "$log" | tail -n 1 || true)
  accepted_count=$(grep -c '^\[WOR92_OPACITY_ACCEPTED\] ' "$log" || true)
  opacity_counts=$(printf '%s\n' "$opacity_line" | sed -n \
    's/.*out_of_domain=\([0-9][0-9]*\) clamped=\([0-9][0-9]*\) negative_results=\([0-9][0-9]*\) zero_results=\([0-9][0-9]*\) nonfinite_results=\([0-9][0-9]*\).*/\1 \2 \3 \4 \5/p')
  opacity_out=0
  opacity_clamped=0
  opacity_neg=0
  opacity_zero=0
  opacity_nonfinite=0
  latest_hdf=$(find "$job/output" -maxdepth 1 -type f -name '*.prim.*.athdf' \
    -printf '%p\n' | sort | tail -n 1)
  opacity_state_ok=0
  if [[ -n "$latest_hdf" ]] &&
     python3 "$opacity_audit" "$latest_hdf" \
       data/opacity/protostellar_semenov2003_nrm_c_a.h5 \
       --backend h5dump \
       >> "$job/opacity_domain_audit.jsonl"; then
    opacity_state_ok=1
  fi
  if (( code != 0 )) || grep -q 'FATAL ERROR' "$log"; then
    reason=FAIL_SOLVER
  elif (( after <= current )); then
    reason=FAIL_NO_PROGRESS
  elif [[ -z "${jeans:-}" || -z "${rho:-}" ]]; then
    reason=FAIL_HISTORY
  elif [[ -z "$opacity_counts" || "$accepted_count" -ne 10 ]]; then
    reason=STOP_OPACITY_DIAGNOSTIC
  elif read -r opacity_out opacity_clamped opacity_neg opacity_zero opacity_nonfinite \
       <<< "$opacity_counts" &&
       (( opacity_out != opacity_clamped || opacity_neg != 0 ||
          opacity_zero != 0 || opacity_nonfinite != 0 )); then
    reason=STOP_OPACITY_RESULT
  elif (( opacity_state_ok != 1 )); then
    reason=STOP_OPACITY_ACCEPTED_DOMAIN
  elif python3 -c 'import math,sys; x=float(sys.argv[1]); sys.exit(0 if not math.isfinite(x) else 1)' "$jeans"; then
    reason=FAIL_HISTORY
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 16 else 1)' "$jeans"; then
    reason=STOP_UNDERRESOLVED
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1e-10 else 1)' "$rho"; then
    reason=FIRST_CORE_DENSITY_CANDIDATE
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 10000 else 1)' "$simtime"; then
    reason=TARGET_TIME
  fi
  printf '%s\n' "$segment,$current,$after,$blocks,$simtime,$jeans,$rho,$tgas,$gasmass,$divb,$lz,$eint,$ekin,$emag,$erad,$level,$elapsed,$code,$reason" \
    >> "$job/progress.csv"
  printf 'SEGMENT=%d CYCLE=%d TIME=%s BLOCKS=%s JEANS=%s RHO=%s REASON=%s\n' \
    "$segment" "$after" "$simtime" "$blocks" "$jeans" "$rho" "$reason" \
    | tee -a "$job/status.txt"
  if (( opacity_out > 0 )); then
    printf 'SEGMENT=%d TRANSIENT_OPACITY_CLAMPS=%d ACCEPTED_STATE_AUDIT=PASS\n' \
      "$segment" "$opacity_out" | tee -a "$job/status.txt"
  fi
  if [[ "$reason" == FAIL_* ]]; then exit 1; fi
  if [[ "$reason" != CONTINUE ]]; then break; fi
  segment=$((segment + 1))
done

sha256sum "$rst" > "$job/final_restart.sha256"
date -Is | tee -a "$job/status.txt"
