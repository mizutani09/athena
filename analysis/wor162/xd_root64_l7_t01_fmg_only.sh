#!/bin/bash
#SBATCH --partition=M-large-t
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --job-name=wor162_l7_t01
#SBATCH --output=/work/mizutaniks/wor162/root64_l7_t01_%j.out

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
work=/work/mizutaniks/wor162
run=$work/root64_l7_t01_${SLURM_JOB_ID}
mkdir -p "$run"
cd "$repository"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena src/pgen/scf_rsg.cpp \
  inputs/rsg/athinput.scf_rsg_relaxation_l7 > "$run/provenance.sha256"

/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n56 --ntasks-per-node=56 bin/athena \
  -i inputs/rsg/athinput.scf_rsg_relaxation_l7 -d "$run" \
  time/tlim=47.006 output1/dcycle=100 output2/dt=4.7006 \
  output3/dt=-1 output4/dt=23.503 output5/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o "$run/analyze_relaxation_history"
"$run/analyze_relaxation_history" "$run/scf_rsg_relax.hst" \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_core_mode.cpp -o "$run/analyze_core_mode"
"$run/analyze_core_mode" "$run" "$run/core_mode.csv" \
  00000 00001 00002 00003 00004 00005 00006 00007 00008 00009 00010

cat "$run/time.log"
cat "$run/relaxation_summary.json"
tail -3 "$run/core_mode.csv"
