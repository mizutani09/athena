#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --job-name=wor120_l3_mem
#SBATCH --output=/work/mizutaniks/wor120/hd4wz_scf_l3_memory_%j.out

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

repository=$HOME/simulation/athena-wor120-hd4wz-memory
restart=${SCF_RESTART:?set SCF_RESTART to the accepted L3 seed restart}
source_history=${SCF_HISTORY:-$(dirname "$restart")/scf_rsg_relax.hst}
run=/work/mizutaniks/wor120/hd4wz_scf_l3_memory_${SLURM_JOB_ID}
test -s "$restart"
test -s "$source_history"
mkdir -p "$run"
cp "$source_history" "$run/scf_rsg_relax.hst"
cd "$repository"

git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
test ! -s "$run/git_status.txt"
sha256sum bin/athena "$restart" src/hydro/hydro.cpp \
  src/hydro/calculate_fluxes.cpp src/eos/eos_high_order.cpp \
  > "$run/provenance.sha256"

/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" \
  bin/athena -r "$restart" -d "$run" \
  time/integrator=rk4 time/xorder=4wz time/nlim=100 time/tlim=47.006 \
  gravity/mgmode=fmg gravity/threshold=-1.0 gravity/niteration=0 \
  output1/dcycle=10 output2/dt=-1 output3/dt=-1 \
  output4/dt=-1 output5/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on cycle limit' "$run/stdout.log"
if grep -Eiq 'nan|fatal error|segmentation fault|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo 'memory gate produced a numeric or runtime failure' >&2
  exit 1
fi
if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
  echo 'memory gate unexpectedly entered discarded multigrid polishing iterations' >&2
  exit 1
fi

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp -o "$run/analyze_history"
"$run/analyze_history" "$run/scf_rsg_relax.hst" \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"

cat "$run/time.log"
grep -E 'Number of MeshBlocks|zone-cycles|cpu time used' "$run/stdout.log" | tail -n 5
cat "$run/relaxation_summary.json"
