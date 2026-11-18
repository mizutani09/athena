#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=2
#SBATCH --ntasks=112
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --job-name=wor162_l3_t055
#SBATCH --output=/work/mizutaniks/wor162/surface_l3_t055_%j.out

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

repository=$HOME/simulation/athena-wor162
source_run=/work/mizutaniks/wor162/surface_l3_t01_342434
restart=$source_run/scf_rsg_relax.final.rst
run=/work/mizutaniks/wor162/surface_l3_t055_${SLURM_JOB_ID}
test -f "$restart"
mkdir -p "$run"
cp "$source_run/scf_rsg_relax.hst" "$run/scf_rsg_relax.hst"
cd "$repository"

git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$restart" src/pgen/scf_rsg.cpp \
  > "$run/provenance.sha256"

/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n112 --ntasks-per-node=56 bin/athena \
  -r "$restart" -d "$run" \
  time/nlim=-1 time/tlim=258.533 \
  gravity/mgmode=fmg gravity/threshold=-1.0 gravity/niteration=0 \
  output1/dcycle=100 output2/dt=4.7 output3/dt=4.7 \
  output4/dt=47.0 output5/dt=117.5 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
  echo 'unexpected tolerance-driven V-cycle warning in FMG-only continuation' >&2
  exit 1
fi
test -f "$run/scf_rsg_relax.final.rst"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp -o "$run/analyze_history"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_slice.cpp -o "$run/analyze_slice"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/evaluate_relaxation_gates.cpp -o "$run/evaluate_gates"
"$run/analyze_history" "$run/scf_rsg_relax.hst" \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"
"$run/evaluate_gates" "$run/relaxation_diagnostics.csv" \
  "$run/relaxation_gates_after_t025.json" 0.25

latest_snapshot=$(find "$run" -maxdepth 1 -name 'scf_rsg_relax.block*.out2.*.tab' \
  -printf '%f\n' | sed -E 's/.*\.out2\.([0-9]+)\.tab/\1/' | sort -u | tail -n 1)
test -n "$latest_snapshot"
"$run/analyze_slice" "$run" "$latest_snapshot" \
  "$run/slice_${latest_snapshot}.csv" "$run/slice_${latest_snapshot}.json"

cat "$run/time.log"
cat "$run/relaxation_summary.json"
cat "$run/relaxation_gates_after_t025.json"
cat "$run/slice_${latest_snapshot}.json"
