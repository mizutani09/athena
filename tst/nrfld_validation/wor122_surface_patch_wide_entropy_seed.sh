#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=70G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor122_entropy_seed
#SBATCH --output=/work/mizutaniks/wor122/surface_patch_entropy_%j.out
#SBATCH --error=/work/mizutaniks/wor122/surface_patch_entropy_%j.err
set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5 >/dev/null 2>&1
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0
base=/work/mizutaniks/wor122
cd "$base/athena_surface_patch"
make CXX=mpiicpx -j8 > "$base/surface_patch_entropy_make.log" 2>&1
run="$base/surface_patch_wide_entropy_seed"
mkdir -p "$run"
srun -n32 bin/athena -i inputs/rsg/athinput.wor122_rsg_patch_wide_entropy_seed -d "$run" > "$run/stdout.log" 2> "$run/stderr.log"
grep -Eq 'Terminating on (time|cycle) limit' "$run/stdout.log"
