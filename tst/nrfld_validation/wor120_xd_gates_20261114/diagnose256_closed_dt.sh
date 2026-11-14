#!/bin/bash
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=150G
#SBATCH --time=00:05:00
#SBATCH --job-name=wor120_closeddt
#SBATCH --output=/work/mizutaniks/wor120/closed_dt_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0
root=/work/mizutaniks/wor120/athena
source_run=/work/mizutaniks/wor120/lte256_closed_probe
run=/work/mizutaniks/wor120/lte256_closed_dt
cd "$root"
test -s "$source_run/wor120_lte256_closed_probe.final.rst"
test ! -e "$run/stdout.log"
mkdir -p "$run"
srun -n64 bin/athena_wor120_e4233143 -r "$source_run/wor120_lte256_closed_probe.final.rst" -d "$run" -t 00:03:00 job/problem_id=wor120_lte256_closed_dt time/nlim=22 time/tlim=2.0 time/ncycle_out=1 time/dt_diagnostics=0 output1/dt=1000.0 output2/dt=1000.0 output3/dt=1000.0 output4/dt=1000.0 > "$run/stdout.log" 2> "$run/stderr.log"
