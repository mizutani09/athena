#!/bin/bash -l
#SBATCH --partition=M-large-a
#SBATCH --nodes=5
#SBATCH --ntasks=320
#SBATCH --ntasks-per-node=64
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --job-name=paper_B32_amr12_v2
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
: "${WOR124_GATE_JOB:?submit with WOR124_GATE_JOB=<successful gate job id>}"
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_protostellar_firstcore_v2_577f273bd"
gate="$base/paper_amr12_v2/gate_job_${WOR124_GATE_JOB}"
reader="$base/scripts/read_restart.py"
audit="$base/scripts/collapse_opacity_domain_audit.py"
opacity_table="$base/source_full_solve_early_retry_20261119/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
initial_rst="$gate/output/paper_B32_amr12_v2_gate.final.rst"
label=paper_B32_amr12_v2
job="$base/paper_amr12_v2/long_job_${SLURM_JOB_ID}"
mkdir -p "$job/output"

if ! awk -F, 'NR==2 && $2==0 && $3==1 && $4==0 {ok=1} END{exit !ok}' \
    "$gate/status.csv" 2>/dev/null; then
  printf 'FAIL_GATE gate_job=%s\n' "$WOR124_GATE_JOB" > "$job/status.txt"
  exit 1
fi
if [[ ! -s "$initial_rst" ]] || [[ $(python3 "$reader" "$initial_rst" --field cycle) != 1790 ]]; then
  printf 'FAIL_INVALID_CHECKPOINT path=%s\n' "$initial_rst" > "$job/status.txt"
  exit 1
fi

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
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
unset ATHENA_NRFLD_FGMRES_REFINEMENT_STEPS
unset ATHENA_NRFLD_FGMRES_REFINEMENT_ETA_TRIGGER
unset ATHENA_NRFLD_TANGENTIAL_POSITIVITY_FALLBACK

cd "$source_dir"
{
  date -Is
  printf 'source_commit=577f273bddee4128d873cd7da7a4b1495b97ba8c source_tree=8c4a94cb1b53b0ce6a7b574371351007a98472cb tag=protostellar-firstcore-v2\n'
  printf 'ranks=%s gate_job=%s start_cycle=1790 cfl=0.3 numlevel=12 fgmres=40/300_mg2 inexact_cap=1e-7 target_rho=1e-10\n' \
    "$SLURM_NTASKS" "$WOR124_GATE_JOB"
  sha256sum "$initial_rst" bin/athena "$opacity_table"
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"
printf 'segment,start_cycle,end_cycle,meshblocks,time,jeans_min,rho_max,Tgas_max,gas_mass,divB_max,Lz,Eint_gas_cgs,Ekin_gas_cgs,Emag_cgs,Erad_tot_cgs,level_max,elapsed_seconds,exit_code,retry_count,reason\n' \
  > "$job/progress.csv"

started=$(date +%s)
stop_epoch=$((started + 11*3600 + 30*60))
rst="$job/output/$label.final.rst"
hst="$job/output/$label.hst"
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
  if (( runtime > 3600 )); then runtime=3600; fi
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
    time/cfl_number=0.3 time/tlim=10000.0 time/nlim="$target" \
    time/ncycle_out=10 time/xorder=2 \
    output2/dcycle=10 output2/variable=cons,prim,uov output4/dcycle=10 \
    nrfld/diagnostic_verbosity=0 \
    nrfld/jacobian_check=false nrfld/jacobian_check_each_cycle=false \
    nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart=40 \
    nrfld/fgmres_max_iterations=300 nrfld/fgmres_mg_vcycles=2 \
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
  retries=$(grep -c '^\[NR_TANGENTIAL_FULL_RETRY\].*retry_status=converged' "$log" || true)
  reason=CONTINUE
  if (( code != 0 )) || grep -q 'FATAL ERROR' "$log"; then
    reason=FAIL_SOLVER
  elif (( after <= current )); then
    reason=FAIL_NO_PROGRESS
  elif [[ -z "${jeans:-}" || -z "${rho:-}" ]]; then
    reason=FAIL_HISTORY
  elif ! grep -q '^FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0 negative_results=0 zero_results=0 nonfinite_results=0$' "$log"; then
    reason=STOP_OPACITY_DOMAIN
  else
    latest_hdf=$(find "$job/output" -maxdepth 1 -type f -name '*.prim.*.athdf' \
      -printf '%p\n' | sort | tail -n 1)
    if ! python3 "$audit" "$latest_hdf" "$opacity_table" --backend h5dump \
        > "$job/segment_$(printf '%03d' "$segment")_opacity_audit.json"; then
      reason=STOP_ACCEPTED_STATE_AUDIT
    elif python3 -c 'import math,sys; x=float(sys.argv[1]); sys.exit(0 if not math.isfinite(x) else 1)' "$jeans"; then
      reason=FAIL_HISTORY
    elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 16 else 1)' "$jeans"; then
      reason=STOP_UNDERRESOLVED
    elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1e-10 else 1)' "$rho"; then
      reason=FIRST_CORE_DENSITY_CANDIDATE
    elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 10000 else 1)' "$simtime"; then
      reason=TARGET_TIME
    fi
  fi
  printf '%s\n' "$segment,$current,$after,$blocks,$simtime,$jeans,$rho,$tgas,$gasmass,$divb,$lz,$eint,$ekin,$emag,$erad,$level,$elapsed,$code,$retries,$reason" \
    >> "$job/progress.csv"
  printf 'SEGMENT=%d CYCLE=%d TIME=%s BLOCKS=%s JEANS=%s RHO=%s RETRIES=%s REASON=%s\n' \
    "$segment" "$after" "$simtime" "$blocks" "$jeans" "$rho" "$retries" "$reason" \
    | tee -a "$job/status.txt"
  if [[ "$reason" == FAIL_* || "$reason" == STOP_ACCEPTED_STATE_AUDIT ||
        "$reason" == STOP_OPACITY_DOMAIN ]]; then exit 1; fi
  if [[ "$reason" != CONTINUE ]]; then break; fi
  segment=$((segment + 1))
done
sha256sum "$rst" > "$job/final_restart.sha256"
date -Is | tee -a "$job/status.txt"
