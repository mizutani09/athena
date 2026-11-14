#!/bin/bash
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=150G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor120_256open
#SBATCH --output=/work/mizutaniks/wor120/run256_open_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0
root=/work/mizutaniks/wor120/athena
run=/work/mizutaniks/wor120/lte256_open1d
cd "$root"
test -s bin/athena_wor120_e4233143
test ! -e "$run/stdout.log"
mkdir -p "$run"
date -Is > "$run/start_time.txt"
sha256sum bin/athena_wor120_e4233143 inputs/rsg/athinput.st35gm03n07_fld_wor55_128_lte_fld_trial data/ic/st35gm03n07_wor55_lte_fld_trial.dat data/opacity/rsg_gray_opacity_wor55.h5 > "$run/inputs.sha256"
srun -n64 bin/athena_wor120_e4233143 -i inputs/rsg/athinput.st35gm03n07_fld_wor55_128_lte_fld_trial -d "$run" -t 00:13:00 job/problem_id=wor120_lte256_open1d mesh/nx1=256 mesh/nx2=256 mesh/nx3=256 meshblock/nx1=32 meshblock/nx2=32 meshblock/nx3=32 time/nlim=20 time/tlim=3.318764 output1/dt=0.8 output2/dt=3.318764 output3/dt=3.318764 output4/dt=1000.0 > "$run/stdout.log" 2> "$run/stderr.log"
date -Is > "$run/end_time.txt"
