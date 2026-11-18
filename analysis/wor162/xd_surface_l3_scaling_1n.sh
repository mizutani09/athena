#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor162_l3_1n
#SBATCH --output=/work/mizutaniks/wor162/surface_l3_scaling_1n_%j.out

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
run=/work/mizutaniks/wor162/surface_l3_scaling_1n_${SLURM_JOB_ID}
mkdir -p "$run"
cd "$repository"
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n56 --ntasks-per-node=56 bin/athena \
  -i inputs/rsg/athinput.scf_rsg_relaxation_surface_l3 -d "$run" \
  time/nlim=5 time/tlim=1.0 output1/dcycle=5 output2/dt=-1 \
  output3/dt=-1 output4/dt=-1 output5/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
grep -q 'Terminating on cycle limit' "$run/stdout.log"
cat "$run/time.log"
tail -n 12 "$run/stdout.log"
tail -n 2 "$run/scf_rsg_relax.hst"
