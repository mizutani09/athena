#!/bin/bash
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor120_128open
#SBATCH --output=/work/mizutaniks/wor120/run128_open_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0
root=/work/mizutaniks/wor120/athena
run=/work/mizutaniks/wor120/lte128_open5d
cd "$root"
test -s bin/athena_wor120_e4233143
test ! -e "$run/stdout.log"
mkdir -p "$run"
date -Is > "$run/start_time.txt"
sha256sum bin/athena_wor120_e4233143 inputs/rsg/athinput.st35gm03n07_fld_wor55_128_lte_fld_trial data/ic/st35gm03n07_wor55_lte_fld_trial.dat data/opacity/rsg_gray_opacity_wor55.h5 > "$run/inputs.sha256"
srun -n32 bin/athena_wor120_e4233143 -i inputs/rsg/athinput.st35gm03n07_fld_wor55_128_lte_fld_trial -d "$run" -t 00:13:00 job/problem_id=wor120_lte128_open5d time/nlim=500 time/tlim=20.0 output1/dt=1.0 output2/dt=4.0 output3/dt=20.0 output4/dt=20.0 > "$run/stdout.log" 2> "$run/stderr.log"
date -Is > "$run/end_time.txt"
