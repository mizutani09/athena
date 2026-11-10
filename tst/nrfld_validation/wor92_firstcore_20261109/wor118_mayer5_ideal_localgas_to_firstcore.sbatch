#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor118_ideal_localgas_firstcore
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_20261110/continue-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/ideal_mayer5_20261110/continue_job_${SLURM_JOB_ID}"
seed="$base/ideal_mayer5_20261110/replay_job_338740/output/cfl030_flux_precision16_local.final.rst"
binary="$base/source_wor118_ideal_localgas_20261110/bin/athena"
reader="$base/scripts/read_athena_restart_metadata_py36.py"
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
test "$(python3 "$reader" "$seed" --field cycle)" = 1100
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_COMMIT=582acd5999c1a5681ab1bc1d1af292676642ed40\n'
  printf 'SEED_JOB=338740 SEED_CYCLE=1100 NTASKS=%s\n' "$SLURM_NTASKS"
  sha256sum "$seed" "$binary"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
printf 'segment,start_cycle,end_cycle,sim_time,jeans_min,rho_max,tgas_max,gas_mass,exit_code,reason\n' > "$case_dir/progress.csv"
cd "$base/bundle"
start_epoch=$(date +%s)
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
hst="$case_dir/output/cfl030_flux_precision16_local.hst"
for segment in $(seq 0 9); do
  current=$(python3 "$reader" "${source_rst:-$seed}" --field cycle)
  if (( $(date +%s) - start_epoch > 1500 )); then
    printf 'STOP_WALL_BUDGET cycle=%s\n' "$current" | tee -a "$case_dir/status.txt"
    break
  fi
  target=$((current + 10))
  set +e
  mpirun -np "$SLURM_NTASKS" "$binary" -r "${source_rst:-$seed}" \
    -d "$case_dir/output" -t 00:25:00 \
    mesh/numlevel=12 time/cfl_number=0.3 time/tlim=10000.0 time/nlim="$target" \
    nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
    output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
    output2/dcycle=10 output4/dcycle=10 > "$case_dir/segment_$(printf '%03d' "$segment").log" 2>&1
  code=$?
  set -e
  source_rst="$rst"
  after=$(python3 "$reader" "$rst" --field cycle)
  read -r simtime jeans rho tgas gasmass < <(awk 'END {print $1,$35,$16,$20,$23}' "$hst")
  reason=CONTINUE
  if (( code != 0 )) || (( after <= current )); then reason=FAIL_SOLVER_OR_PROGRESS
  elif python3 -c 'import math,sys; x=[float(y) for y in sys.argv[1:]]; sys.exit(0 if all(math.isfinite(y) and y>0 for y in x) else 1)' "$jeans" "$rho" "$tgas" "$gasmass"; then
    if python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 16 else 1)' "$jeans"; then reason=FAIL_UNDERRESOLVED
    elif python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1e-10 else 1)' "$rho"; then reason=FIRST_CORE_DENSITY_CANDIDATE
    fi
  else reason=FAIL_HISTORY; fi
  printf '%d,%d,%d,%s,%s,%s,%s,%s,%d,%s\n' "$segment" "$current" "$after" "$simtime" "$jeans" "$rho" "$tgas" "$gasmass" "$code" "$reason" >> "$case_dir/progress.csv"
  printf 'SEGMENT=%d END_CYCLE=%d RHO_MAX=%s JEANS_MIN=%s EXIT=%d REASON=%s\n' "$segment" "$after" "$rho" "$jeans" "$code" "$reason" | tee -a "$case_dir/status.txt"
  if [[ "$reason" == FAIL_* ]]; then exit 1; fi
  if [[ "$reason" != CONTINUE ]]; then break; fi
done
sha256sum "$rst" > "$case_dir/final_restart.sha256"
date '+END=%Y-%m-%d %H:%M:%S %Z' >> "$case_dir/provenance.txt"
