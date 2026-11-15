#!/bin/bash -l
#SBATCH --partition=M-large-a
#SBATCH --nodes=1
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=120
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --job-name=wor115_ambient_p01_t2000
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/slurm-%j.out

set -euo pipefail
: "${PREFLIGHT_JOB_ID:?set PREFLIGHT_JOB_ID to the validated c0-c10 run}"
case_root=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115
case_dir="$case_root/job_${SLURM_JOB_ID}"
seed_dir="$case_root/job_${PREFLIGHT_JOB_ID}"
binary=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/source_wor92_mayer5_positive_amr_20261115/bin/athena
input="$case_root/ambient_pressure_1over30.in"
reader=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/scripts/read_athena_restart_metadata_py36.py
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
export ATHENA_WOR92_ZERO_B_POSITIVE_AMR=1
printf '%s  %s\n' b64491aa31da79216c99dbfcc6eb2b9cba805e4c385a4a67d092c303a622b5cb "$binary" | sha256sum -c -
printf '%s  %s\n' 5bd3e640c4a731ac8d87cfd825845db995c460a23be4f1de68ccae78da99d5fd "$input" | sha256sum -c -
grep -q '^PASS_C0_TO_C10 cycle=10' "$seed_dir/status.txt"
(cd "$seed_dir/output" && sha256sum -c "$seed_dir/final_restart.sha256")
cp -- "$seed_dir/output/cfl030_flux_precision16_local.final.rst" "$case_dir/seed.rst"
cmp -- "$seed_dir/output/cfl030_flux_precision16_local.final.rst" "$case_dir/seed.rst"
seed_cycle=$(python3 "$reader" "$case_dir/seed.rst" --field cycle)
seed_blocks=$(python3 "$reader" "$case_dir/seed.rst" --field meshblocks)
if (( seed_cycle != 10 || seed_blocks < SLURM_NTASKS )); then
  printf 'FAIL_SEED cycle=%s blocks=%s\n' "$seed_cycle" "$seed_blocks" > "$case_dir/status.txt"
  exit 1
fi
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_BASE_COMMIT=ae03e85b5a131b6c5cf7ddb3fe7a6bcb1ef9f91a\n'
  printf 'PREFLIGHT_JOB_ID=%s\n' "$PREFLIGHT_JOB_ID"
  printf 'CONTROLLED_CHANGE=background_pressure_ratio_1_to_1over30_ONLY\n'
  printf 'STOP=TIME_2000_OR_RHO_1E-10_OR_NUMERICAL_GATE\n'
  sha256sum "$binary" "$input" "$case_dir/seed.rst"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
printf 'segment,start_cycle,end_cycle,code_time,rho_max,tgas_max,pmin,jeans_min,gas_mass,blocks,exit,reason\n' > "$case_dir/progress.csv"
mass0=$(awk 'NF==39 {print $23; exit}' "$seed_dir/output/cfl030_flux_precision16_local.hst")
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
hst="$case_dir/output/cfl030_flux_precision16_local.hst"
start_epoch=$(date +%s)
segment=0
cd /work/mizutaniks/output/fld/collapse/wor92_xd_20261107/bundle
while true; do
  if [[ -s "$rst" ]]; then
    source_rst="$rst"
  else
    source_rst="$case_dir/seed.rst"
  fi
  current=$(python3 "$reader" "$source_rst" --field cycle)
  if (( $(date +%s) - start_epoch > 13200 )); then
    printf 'STOP_WALL_BUDGET cycle=%s\n' "$current" | tee -a "$case_dir/status.txt"
    break
  fi
  target=$((current + 20))
  printf 'SEGMENT=%d START_CYCLE=%d TARGET_CYCLE=%d\n' "$segment" "$current" "$target" | tee -a "$case_dir/status.txt"
  set +e
  timeout 900 mpirun -np "$SLURM_NTASKS" "$binary" -r "$source_rst" -d "$case_dir/output" -t 00:13:00 \
    mesh/numlevel=12 time/cfl_number=0.3 time/tlim=2000.0 time/nlim="$target" \
    nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
    output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
    output2/dcycle=100 output4/dcycle=20 > "$case_dir/segment_$(printf '%03d' "$segment").log" 2>&1
  code=$?
  set -e
  if [[ ! -s "$rst" ]] || [[ ! -s "$hst" ]]; then
    printf 'FAIL_NO_RESTART_OR_HISTORY segment=%d exit=%d\n' "$segment" "$code" | tee -a "$case_dir/status.txt"
    exit 1
  fi
  after=$(python3 "$reader" "$rst" --field cycle)
  read -r code_time rho tgas pmin jeans mass < <(awk 'NF==39 {t=$1; rho=$16; temp=$20; p=$30; j=$35; m=$23} END {print t,rho,temp,p,j,m}' "$hst")
  bad=$(awk 'NF==39 && ($30<=1e-18 || $35<16) {print NR,$1,$30,$35; exit}' "$hst")
  blocks=$(python3 "$reader" "$rst" --field meshblocks)
  reason=CONTINUE
  if (( code != 0 )) || grep -q 'FATAL ERROR' "$case_dir/segment_$(printf '%03d' "$segment").log"; then
    reason=FAIL_SOLVER_OR_TIMEOUT
  elif (( after <= current )); then
    reason=FAIL_NO_PROGRESS
  elif [[ -n "$bad" ]]; then
    reason=FAIL_PRESSURE_OR_JEANS
  elif ! python3 -c 'import math,sys; m,m0=map(float,sys.argv[1:]); sys.exit(0 if math.isfinite(m) and abs(m/m0-1)<=1e-8 else 1)' "$mass" "$mass0"; then
    reason=FAIL_MASS
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1])>=1e-10 else 1)' "$rho"; then
    reason=STOP_FIRST_CORE_DENSITY
  elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1])>=2000-1e-8 else 1)' "$code_time"; then
    reason=STOP_TIME_2000
  fi
  printf '%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%d,%s\n' "$segment" "$current" "$after" "$code_time" "$rho" "$tgas" "$pmin" "$jeans" "$mass" "$blocks" "$code" "$reason" >> "$case_dir/progress.csv"
  printf 'SEGMENT=%d END_CYCLE=%d CODE_TIME=%s RHO_MAX=%s JEANS_MIN=%s PMIN=%s REASON=%s\n' "$segment" "$after" "$code_time" "$rho" "$jeans" "$pmin" "$reason" | tee -a "$case_dir/status.txt"
  if [[ "$reason" == FAIL_* ]]; then
    printf 'FIRST_BAD_HISTORY_ROW=%s\n' "$bad" | tee -a "$case_dir/status.txt"
    exit 2
  fi
  if [[ "$reason" != CONTINUE ]]; then break; fi
  segment=$((segment + 1))
done
sha256sum "$rst" > "$case_dir/final_restart.sha256"
date '+END=%Y-%m-%d %H:%M:%S %Z' | tee -a "$case_dir/status.txt"
