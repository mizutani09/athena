#!/bin/bash
#SBATCH --partition=M-large-a
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=70G
#SBATCH --time=01:00:00
#SBATCH --job-name=wor122_spinup40
#SBATCH --output=/work/mizutaniks/wor122/nodiff_spinup_40r_%j.out
#SBATCH --error=/work/mizutaniks/wor122/nodiff_spinup_40r_%j.err
set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5 >/dev/null 2>&1
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0
base=/work/mizutaniks/wor122
cd "$base/athena_surface_patch"
run="$base/nodiff_spinup_40r"
mkdir -p "$run"
srun -n32 bin/athena -i inputs/rsg/athinput.wor122_nodiff_spinup_40r -d "$run" > "$run/stdout.log" 2> "$run/stderr.log"
grep -Eq 'Terminating on (time|cycle) limit' "$run/stdout.log"
