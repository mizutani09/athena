#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=2
#SBATCH --ntasks=112
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor162_fmg0_gate
#SBATCH --output=/work/mizutaniks/wor162/surface_l3_fmg0_gate_%j.out

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
restart=/work/mizutaniks/wor162/surface_l3_t01_342434/scf_rsg_relax.final.rst
run=/work/mizutaniks/wor162/surface_l3_fmg0_gate_${SLURM_JOB_ID}
test -f "$restart"
mkdir -p "$run"
cd "$repository"

sha256sum bin/athena "$restart" > "$run/provenance.sha256"
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n112 --ntasks-per-node=56 bin/athena \
  -r "$restart" -d "$run" \
  time/nlim=-1 time/tlim=47.106 \
  gravity/mgmode=fmg gravity/threshold=-1.0 gravity/niteration=0 \
  output1/dcycle=5 output2/dt=-1 output3/dt=-1 \
  output4/dt=-1 output5/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
  echo 'unexpected tolerance-driven V-cycle warning in FMG-only gate' >&2
  exit 1
fi
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp -o "$run/analyze_history"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/evaluate_relaxation_gates.cpp -o "$run/evaluate_gates"
"$run/analyze_history" "$run/scf_rsg_relax.hst" \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"
"$run/evaluate_gates" "$run/relaxation_diagnostics.csv" \
  "$run/relaxation_gates.json"

cat "$run/time.log"
tail -n 12 "$run/stdout.log"
cat "$run/relaxation_summary.json"
cat "$run/relaxation_gates.json"
