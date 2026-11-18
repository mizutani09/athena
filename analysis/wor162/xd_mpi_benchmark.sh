#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor162_mpi_bench
#SBATCH --output=/work/mizutaniks/wor162/mpi_benchmark_%j.out

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
benchmark_root=/work/mizutaniks/wor162
mkdir -p "$benchmark_root"
cd "$repository"

for ranks in 8 16 32 56; do
  run=$benchmark_root/bench_r${ranks}_${SLURM_JOB_ID}
  mkdir -p "$run"
  /usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
    srun --exclusive -n "$ranks" --ntasks-per-node="$ranks" \
    bin/athena -i inputs/rsg/athinput.scf_rsg_map -d "$run" \
    time/nlim=50 time/tlim=1000 output1/dcycle=-1 \
    output2/dt=-1 output3/dt=-1 > "$run/stdout.log" 2> "$run/stderr.log"
  grep -q 'Terminating on cycle limit' "$run/stdout.log"
  printf 'ranks=%s ' "$ranks"
  cat "$run/time.log"
  grep -E 'zone-cycles =|cpu time used|zone-cycles/cpu_second' "$run/stdout.log"
done
