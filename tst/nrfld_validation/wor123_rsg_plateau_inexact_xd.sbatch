#!/bin/bash
# Bounded inexact-Newton pilot from the immutable 256^3 RSG checkpoint.
# FGMRES still targets 1e-4.  It may return early only after two restart
# intervals when the independently recomputed true residual is <=1e-2 and
# the latest restart reduced that residual by less than ten percent.
# The outer Newton merit, line search, and convergence tests are unchanged.
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --job-name=wor123_plateau_inexact
#SBATCH --output=/work/mizutaniks/wor122/wor123_plateau_inexact_%j.out

set -euo pipefail
. /etc/profile.d/modules.sh
module use /opt/modulefiles /work/opt/intel/modulefiles \
  /work/opt/local_oneapi/modulefiles
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5 >/dev/null 2>&1

export SLURM_MPI_TYPE=pmi2
export LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export ATHENA_NRFLD_INEXACT_TRUE_CAP=1e-2
export ATHENA_NRFLD_INEXACT_STAGNATION_RATIO=0.9
ulimit -c 0

base=/work/mizutaniks/wor122
code="$base/athena_wor123_plateau_20261120"
source_rst="$base/prep256_outer_t20c/wor122_prep256_outer_t20c.00002.rst"
run="$base/wor123_plateau_inexact/job_${SLURM_JOB_ID}"
mkdir -p "$run/output"

test -s "$source_rst"
test -f "$code/Makefile"
cd "$code"
make -j16 > "$run/build.log" 2>&1
test -x bin/athena

{
  date -Is
  printf 'job=%s nodes=%s ranks=%s start_checkpoint=%s target_time=17.25\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$source_rst"
  printf 'fgmres_restart=20 fgmres_max_iterations=200 fgmres_mg_vcycles=1 fgmres_threshold=1e-4\n'
  printf 'inexact_true_cap=1e-2 stagnation_ratio=0.9 outer_newton_unchanged=1\n'
  sha256sum "$source_rst" bin/athena \
    src/linear_solver/linearMG/linearMG.cpp \
    src/linear_solver/linearMG/linearMG.hpp \
    src/newton_raphson/Newton_Raphson_driver.cpp \
    data/eos/rsg_tomida2013_material_eos.tab
  git rev-parse HEAD 2>/dev/null || true
  git status --short 2>/dev/null || true
  module list 2>&1
} > "$run/provenance.txt"

started=$(date +%s)
set +e
/usr/bin/time -v -o "$run/resources.txt" \
  srun -n64 bin/athena -r "$source_rst" -d "$run/output" \
    job/problem_id=wor123_plateau_inexact \
    time/tlim=17.25 time/nlim=1000 problem/short_test_max_dt=0.1 \
    nrfld/fgmres_restart=20 nrfld/fgmres_max_iterations=200 \
    nrfld/fgmres_mg_vcycles=1 nrfld/fgmres_threshold=1e-4 \
    output1/dt=0.1 output2/dt=0.5 output3/dt=0.5 output4/dt=1000 \
    > "$run/stdout.log" 2> "$run/stderr.log"
code_status=$?
set -e
elapsed=$(($(date +%s)-started))

time_limit=0
fatal=0
opacity_ok=0
if grep -q 'Terminating on time limit' "$run/stdout.log"; then time_limit=1; fi
if grep -q 'FATAL ERROR' "$run/stdout.log" "$run/stderr.log"; then fatal=1; fi
if grep -q 'FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0' \
    "$run/stdout.log"; then opacity_ok=1; fi
inexact_accepts=$(grep -c 'inexact_accept=1' "$run/stdout.log" || true)

printf 'case,exit_code,time_limit,has_fatal,opacity_ok,inexact_accepts,elapsed_seconds\n' \
  > "$run/status.csv"
printf 'plateau_inexact,%s,%s,%s,%s,%s,%s\n' \
  "$code_status" "$time_limit" "$fatal" "$opacity_ok" \
  "$inexact_accepts" "$elapsed" >> "$run/status.csv"

grep -E '^\[LINEAR_FGMRES(_RESTART)?\]|^\[NR\] status=' "$run/stdout.log" \
  > "$run/iteration_summary.log" || true
find "$run/output" -maxdepth 1 -type f -name '*.final.rst' -print0 \
  | xargs -0 -r sha256sum > "$run/final_restart.sha256"

if (( code_status != 0 || time_limit != 1 || fatal != 0 || opacity_ok != 1 \
      || inexact_accepts < 1 )); then
  exit 1
fi
date -Is > "$run/completed.txt"
