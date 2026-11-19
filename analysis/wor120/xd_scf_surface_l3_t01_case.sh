#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --job-name=wor120_l3_t01
#SBATCH --output=/work/mizutaniks/wor120/hd4wz_scf_l3_t01_%j.out

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

repository=$HOME/simulation/athena-wor120-hd4wz
method=${SCF_METHOD:?set SCF_METHOD to x2 or 4wz}
restart=${SCF_RESTART:?set SCF_RESTART to the accepted L3 seed restart}
source_history=${SCF_HISTORY:-$(dirname "$restart")/scf_rsg_relax.hst}
target_time=${SCF_TARGET_TIME:-47.006}
case "$method" in
  x2)
    integrator=rk2
    xorder=2
    ;;
  4wz)
    integrator=rk4
    xorder=4wz
    ;;
  *)
    echo "unsupported SCF_METHOD: $method" >&2
    exit 2
    ;;
esac

run=/work/mizutaniks/wor120/hd4wz_scf_l3_t01_${method}_${SLURM_JOB_ID}
test -s "$restart"
test -s "$source_history"
mkdir -p "$run"
cp "$source_history" "$run/scf_rsg_relax.hst"
cd "$repository"

git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$restart" src/eos/eos_high_order.cpp \
  src/reconstruct/wenoz_berta.cpp src/pgen/scf_rsg.cpp \
  > "$run/provenance.sha256"

/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" \
  bin/athena -r "$restart" -d "$run" \
  time/integrator="$integrator" time/xorder="$xorder" \
  time/nlim=-1 time/tlim="$target_time" \
  gravity/mgmode=fmg gravity/threshold=-1.0 gravity/niteration=0 \
  output1/dcycle=100 output2/dt=-1 output3/dt=-1 \
  output4/dt="$target_time" output4/next_time="$target_time" output5/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -Eiq 'nan|fatal error|segmentation fault' "$run/stdout.log" "$run/stderr.log"; then
  echo "$method produced a numeric or runtime failure" >&2
  exit 1
fi
if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
  echo "$method unexpectedly entered discarded multigrid polishing iterations" >&2
  exit 1
fi
test -s "$run/scf_rsg_relax.final.rst"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp -o "$run/analyze_history"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/evaluate_relaxation_gates.cpp -o "$run/evaluate_gates"
"$run/analyze_history" "$run/scf_rsg_relax.hst" \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"

# A physical gate failure is a valid scientific result and must not hide the completed
# run from the downstream A/B comparison.  Exit code 2 still means malformed input.
set +e
"$run/evaluate_gates" "$run/relaxation_diagnostics.csv" \
  "$run/relaxation_gates.json"
gate_rc=$?
set -e
if [ "$gate_rc" -eq 2 ]; then
  echo 'relaxation gate evaluation failed to parse its input' >&2
  exit 1
fi
printf 'physical_gate_exit_code=%s\n' "$gate_rc" > "$run/gate_status.txt"

cat "$run/time.log"
grep -E 'Number of MeshBlocks|zone-cycles|cpu time used' "$run/stdout.log" | tail -n 5
cat "$run/relaxation_summary.json"
cat "$run/relaxation_gates.json"
