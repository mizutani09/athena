#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=/work/mizutaniks/wor177/corecut7p00_%j.out

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

case "${CORE_LEVEL:-}" in
  l3)
    input=inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_taper_surface_l2_l3_4wz
    problem_id=mesa_scf_rsg_corecut7p00_l3_4wz
    ;;
  l4)
    input=inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_taper_surface_l2_l4_4wz
    problem_id=mesa_scf_rsg_corecut7p00_l4_4wz
    ;;
  *)
    echo "CORE_LEVEL must be l3 or l4" >&2
    exit 2
    ;;
esac

repository=$HOME/simulation/athena-wor177-aggressive-core
profile=data/ic/rsg21_mesa_scf_corecut7p00_g5p3_nr16384_taper_rho3e-9_gamma5_3.csv
run=/work/mizutaniks/wor177/corecut7p00_${CORE_LEVEL}_t1p5_${SLURM_JOB_ID}
# 1.5 times the 21-Msun, 931.9-Rsun dynamical time (t_dyn=470.06031 code units).
target_time=705.090465
slice_dt=47.006031

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$profile"

git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$input" "$profile" \
  src/pgen/scf_rsg.cpp analysis/wor162/analyze_relaxation_history.cpp \
  analysis/wor162/analyze_relaxation_slice.cpp \
  analysis/wor162/evaluate_relaxation_gates.cpp \
  > "$run/provenance.sha256"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o "$run/analyze_relaxation_history"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_slice.cpp \
  -o "$run/analyze_relaxation_slice"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/evaluate_relaxation_gates.cpp \
  -o "$run/evaluate_relaxation_gates"

/usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
  -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -i "$input" -d "$run" \
  time/nlim=-1 time/tlim="$target_time" time/ncycle_out=500 \
  output1/dcycle=50 output2/dcycle=5000 \
  output3/dt="$slice_dt" output3/next_time=0.0 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo "$CORE_LEVEL produced a numeric or runtime failure" >&2
  exit 1
fi
if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
  echo "$CORE_LEVEL unexpectedly entered iterative multigrid polishing" >&2
  exit 1
fi

"$run/analyze_relaxation_history" \
  "$run/${problem_id}.hst" \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"
for table in "$run"/*.out3.*.tab; do
  test -e "$table" || continue
  snapshot=${table##*.out3.}
  snapshot=${snapshot%.tab}
  "$run/analyze_relaxation_slice" "$run" "$snapshot" \
    "$run/slice_${snapshot}.csv" "$run/slice_${snapshot}.json" 0.0 out3
done
set +e
"$run/evaluate_relaxation_gates" "$run/relaxation_diagnostics.csv" \
  "$run/relaxation_gates.json"
gate_rc=$?
set -e
if [ "$gate_rc" -eq 2 ]; then
  echo "$CORE_LEVEL gate evaluator could not parse its input" >&2
  exit 1
fi
printf 'physical_gate_exit_code=%s\n' "$gate_rc" > "$run/gate_status.txt"
test -s "$run/${problem_id}.final.rst"

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
cat "$run/relaxation_summary.json"
cat "$run/relaxation_gates.json"
cat "$run/gate_status.txt"
