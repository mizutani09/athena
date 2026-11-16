#!/bin/bash -l
#SBATCH --partition=M-large-a
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=192
#SBATCH --mem=110G
#SBATCH --time=24:00:00
#SBATCH --job-name=wor124_B32_amr12_rescue
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_floor_trial"
reader="$base/scripts/read_restart.py"
origin="$base/high_amr_long/combined_B32uG_floor_amr11_3node_job_341015"
gate="$base/continuation/B32_c1780to1790_amr12_gate_job_341113/status.csv"
initial_rst="$origin/output/combined_B32uG_floor_amr11_3node.final.rst"
label=combined_B32uG_floor_amr12_rescue
job="$base/high_amr_long/${label}_job_${SLURM_JOB_ID}"
mkdir -p "$job/output"
rst="$job/output/$label.final.rst"
hst="$job/output/$label.hst"
if ! grep -q 'REASON=STOP_UNDERRESOLVED' "$origin/status.txt" 2>/dev/null; then
  printf 'SKIP_NO_UNDERRESOLUTION original_job=341015\n' > "$job/status.txt"
  exit 0
fi
if ! grep -q '^B32_c1780to1790_amr12_gate,0,1,0,' "$gate" 2>/dev/null; then
  printf 'FAIL_AMR12_GATE job=341113\n' > "$job/status.txt"
  exit 1
fi
if [[ ! -s "$initial_rst" ]]; then
  printf 'FAIL_NO_ORIGINAL_RESTART path=%s\n' "$initial_rst" > "$job/status.txt"
  exit 1
fi
initial_cycle=$(python3 "$reader" "$initial_rst" --field cycle)
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
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-6
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
cd "$source_dir"

if [[ ! -s "$initial_rst" ]] ||
   [[ $(python3 "$reader" "$initial_rst" --field cycle) != "$initial_cycle" ]]; then
  printf 'FAIL_INVALID_CHECKPOINT expected_cycle=%s path=%s\n' \
    "$initial_cycle" "$initial_rst" > "$job/status.txt"
  exit 1
fi
started=$(date +%s)
stop_epoch=$((started + 23*3600 + 30*60))
{
  date -Is
  printf 'source_commit=%s\n' "$(cat "$base/source_commit.txt")"
  printf 'source_dir=%s\n' "$source_dir"
  sha256sum "$source_dir/src/eos/general/general_mhd.cpp"
  printf 'label=%s ranks=%s initial_cycle=%s mesh_numlevel=12 cfl=0.3 fgmres=160/600_mg3 cap=1e-6 original_job=341015 gate_job=341113\n' \
    "$label" "$SLURM_NTASKS" "$initial_cycle"
  sha256sum "$initial_rst" "$reader" bin/athena
  sha256sum data/eos/protostellar_tomida2015_material.tab \
            data/opacity/protostellar_semenov2003_nrm_c_a.h5
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"
printf 'segment,start_cycle,end_cycle,meshblocks,time,jeans_min,rho_max,Tgas_max,gas_mass,divB_max,Lz,Eint_gas_cgs,Ekin_gas_cgs,Emag_cgs,Erad_tot_cgs,level_max,elapsed_seconds,exit_code,reason\n' \
  > "$job/progress.csv"

segment=0
max_segments="${WOR124_MAX_SEGMENTS:-0}"
while true; do
  if (( max_segments > 0 && segment >= max_segments )); then
    printf 'STOP_SEGMENT_LIMIT segments=%s\n' "$segment" | tee -a "$job/status.txt"
    break
  fi
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
    time/cfl_number=0.3 time/tlim=10000.0 time/nlim="$target" \
    time/ncycle_out=10 output2/dcycle=10 output2/variable=cons,prim,uov \
    output4/dcycle=10 nrfld/diagnostic_verbosity=0 \
    nrfld/jacobian_check=false nrfld/jacobian_check_each_cycle=false \
    nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart=160 \
    nrfld/fgmres_max_iterations=600 nrfld/fgmres_mg_vcycles=3 \
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
  if (( code != 0 )) || grep -q 'FATAL ERROR' "$log"; then
    reason=FAIL_SOLVER
  elif (( after <= current )); then
    reason=FAIL_NO_PROGRESS
  elif [[ -z "${jeans:-}" || -z "${rho:-}" ]]; then
    reason=FAIL_HISTORY
  elif ! grep -q '^FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0 negative_results=0 zero_results=0 nonfinite_results=0$' "$log"; then
    reason=STOP_OPACITY_DOMAIN
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
  if [[ "$reason" == FAIL_* ]]; then exit 1; fi
  if [[ "$reason" != CONTINUE ]]; then break; fi
  segment=$((segment + 1))
done
sha256sum "$rst" > "$job/final_restart.sha256"
date -Is | tee -a "$job/status.txt"
