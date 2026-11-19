#!/bin/bash
# Fixed-inexact pilot from the immutable 256^3 RSG checkpoint.  The outer
# Newton convergence test remains unchanged; only the Schur solve target is
# relaxed from 1e-4 to 1e-2.
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --job-name=wor123_fixed_inexact1e2
#SBATCH --output=/work/mizutaniks/wor122/wor123_fixed_inexact1e2_%j.out

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
ulimit -c 0

base=/work/mizutaniks/wor122
code="$base/athena_prep_thermal"
source_rst="$base/prep256_outer_t20c/wor122_prep256_outer_t20c.00002.rst"
run="$base/wor123_fixed_inexact1e2/job_${SLURM_JOB_ID}"
mkdir -p "$run/output"

test -s "$source_rst"
test -x "$code/bin/athena"
cd "$code"

{
  date -Is
  printf 'job=%s nodes=%s ranks=%s start_checkpoint=%s target_time=17.25\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$source_rst"
  printf 'fgmres_restart=20 fgmres_max_iterations=200 fgmres_mg_vcycles=1 fgmres_threshold=1e-2 outer_newton_unchanged=1\n'
  sha256sum "$source_rst" bin/athena \
    src/linear_solver/linearMG/linearMG.cpp \
    src/newton_raphson/Newton_Raphson_driver.cpp \
    data/eos/rsg_tomida2013_material_eos.tab
  module list 2>&1
} > "$run/provenance.txt"

started=$(date +%s)
set +e
/usr/bin/time -v -o "$run/resources.txt" \
  srun -n64 bin/athena -r "$source_rst" -d "$run/output" \
    job/problem_id=wor123_fixed_inexact1e2 \
    time/tlim=17.25 time/nlim=1000 problem/short_test_max_dt=0.1 \
    nrfld/fgmres_restart=20 nrfld/fgmres_max_iterations=200 \
    nrfld/fgmres_mg_vcycles=1 nrfld/fgmres_threshold=1e-2 \
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

printf 'case,exit_code,time_limit,has_fatal,opacity_ok,elapsed_seconds\n' \
  > "$run/status.csv"
printf 'fixed_inexact_1e2,%s,%s,%s,%s,%s\n' \
  "$code_status" "$time_limit" "$fatal" "$opacity_ok" "$elapsed" \
  >> "$run/status.csv"

grep -E '^\[LINEAR_FGMRES\]|^\[NR\] status=' "$run/stdout.log" \
  > "$run/iteration_summary.log" || true
find "$run/output" -maxdepth 1 -type f -name '*.final.rst' -print0 \
  | xargs -0 -r sha256sum > "$run/final_restart.sha256"

if (( code_status != 0 || time_limit != 1 || fatal != 0 || opacity_ok != 1 )); then
  exit 1
fi
date -Is > "$run/completed.txt"
