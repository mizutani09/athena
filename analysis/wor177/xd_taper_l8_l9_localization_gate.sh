#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --job-name=wor177_l8_l9
#SBATCH --output=/work/mizutaniks/wor177/taper_l8_l9_gate_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
export SLURM_MPI_TYPE=pmi2
export LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
ulimit -c 0

repository=$HOME/simulation/athena-wor177-mesa-hd4wz
input=inputs/rsg/athinput.mesa_scf_rsg_taper_surface_l2_4wz
failed_run=/work/mizutaniks/wor177/taper_surface_l2_4wz_continue_t013_344423/continuation
checkpoint_5000=$failed_run/mesa_scf_rsg_taper_surface_l2_4wz.00002.rst
checkpoint_10000=$failed_run/mesa_scf_rsg_taper_surface_l2_4wz.00003.rst
run_root=/work/mizutaniks/wor177/taper_l8_l9_gate_${SLURM_JOB_ID}
# 0.012 and 0.006 global dynamical times in code units.
target_time=5.640720
slice_dt=2.820360

test -s "$checkpoint_5000"
test -s "$checkpoint_10000"
mkdir -p "$run_root"
cd "$repository"
test -x bin/athena

git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
sha256sum bin/athena "$input" "$checkpoint_5000" "$checkpoint_10000" \
  src/pgen/scf_rsg.cpp analysis/wor162/analyze_relaxation_history.cpp \
  analysis/wor162/analyze_relaxation_slice.cpp \
  > "$run_root/provenance.sha256"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o "$run_root/analyze_relaxation_history"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_slice.cpp \
  -o "$run_root/analyze_relaxation_slice"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/evaluate_relaxation_gates.cpp \
  -o "$run_root/evaluate_relaxation_gates"

localize_checkpoint() {
  local label=$1
  local restart=$2
  local run=$run_root/$label
  mkdir -p "$run"
  /usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
    -o "$run/time.log" \
    srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
    bin/athena -r "$restart" -i "$input" -d "$run" \
    time/nlim=1 time/tlim=10000.0 time/ncycle_out=1 \
    output1/dcycle=-1 output2/dcycle=-1 \
    output3/dt=1.0e-12 output3/next_time=0.0 \
    > "$run/stdout.log" 2> "$run/stderr.log"
  grep -q 'Terminating on cycle limit' "$run/stdout.log"
  if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
      "$run/stdout.log" "$run/stderr.log"; then
    echo "$label localization failed" >&2
    exit 1
  fi
  "$run_root/analyze_relaxation_slice" "$run" 00000 \
    "$run/slice_00000.csv" "$run/slice_00000.json" 0.0 out3
}

run_resolution_case() {
  local label=$1
  shift
  local run=$run_root/$label
  mkdir -p "$run"
  /usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
    -o "$run/time.log" \
    srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
    bin/athena -i "$input" -d "$run" \
    time/nlim=-1 time/tlim="$target_time" time/ncycle_out=100 \
    output1/dcycle=250 output2/dcycle=-1 \
    output3/dt="$slice_dt" output3/next_time=0.0 \
    "$@" > "$run/stdout.log" 2> "$run/stderr.log"
  grep -q 'Terminating on time limit' "$run/stdout.log"
  if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
      "$run/stdout.log" "$run/stderr.log"; then
    echo "$label produced a numeric or runtime failure" >&2
    exit 1
  fi
  if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
    echo "$label unexpectedly entered iterative multigrid polishing" >&2
    exit 1
  fi
  "$run_root/analyze_relaxation_history" \
    "$run/mesa_scf_rsg_taper_surface_l2_4wz.hst" \
    "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"
  for snapshot in 00000 00001 00002; do
    if compgen -G "$run/*.out3.${snapshot}.tab" >/dev/null; then
      "$run_root/analyze_relaxation_slice" "$run" "$snapshot" \
        "$run/slice_${snapshot}.csv" "$run/slice_${snapshot}.json" 0.0 out3
    fi
  done
  set +e
  "$run_root/evaluate_relaxation_gates" "$run/relaxation_diagnostics.csv" \
    "$run/relaxation_gates.json"
  local gate_rc=$?
  set -e
  if [ "$gate_rc" -eq 2 ]; then
    echo "$label gate evaluator could not parse its input" >&2
    exit 1
  fi
  printf 'physical_gate_exit_code=%s\n' "$gate_rc" > "$run/gate_status.txt"
}

# The failed continuation already contains the two most useful pre-catastrophe
# states: cycle 5,000 has K_nonrad/|W|=0.113, while cycle 10,000 is just before
# the interior-Mach catastrophe.  One cycle is enough to emit equatorial tabs.
localize_checkpoint checkpoint_5000 "$checkpoint_5000"
localize_checkpoint checkpoint_10000 "$checkpoint_10000"

# Fresh-map A/B comparison.  L9 replaces the innermost fixed L8 box by a
# half-width L9 box; the mesh generator supplies the required L8 parents.
run_resolution_case fresh_l8
run_resolution_case fresh_l9 \
  mesh/numlevel=10 \
  refinement8/level=9 \
  refinement8/x1min=-2.1484375 refinement8/x1max=2.1484375 \
  refinement8/x2min=-2.1484375 refinement8/x2max=2.1484375 \
  refinement8/x3min=-2.1484375 refinement8/x3max=2.1484375

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run_root/slurm_accounting.txt" || true

for label in checkpoint_5000 checkpoint_10000 fresh_l8 fresh_l9; do
  echo "--- $label ---"
  cat "$run_root/$label/time.log"
  if test -s "$run_root/$label/relaxation_summary.json"; then
    cat "$run_root/$label/relaxation_summary.json"
    cat "$run_root/$label/relaxation_gates.json"
  fi
  for summary in "$run_root/$label"/slice_*.json; do
    test -e "$summary" && cat "$summary"
  done
done
