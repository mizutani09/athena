#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor162_l6_t01a
#SBATCH --output=/work/mizutaniks/wor162/root64_l6_t01a_%j.out

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
run=/work/mizutaniks/wor162/root64_l6_t01_split
mkdir -p "$run"
cd "$repository"
/usr/bin/time -f 'stage1_wall_s=%e maxrss_kb=%M' -o "$run/time_stage1.log" \
  srun -n56 --ntasks-per-node=56 bin/athena \
  -i inputs/rsg/athinput.scf_rsg_relaxation_l6 -d "$run" \
  time/tlim=23.5 output1/dcycle=50 output2/dt=-1 output3/dt=-1 \
  output4/dt=23.5 output5/dt=-1 > "$run/stdout_stage1.log" 2> "$run/stderr_stage1.log"
grep -q 'Terminating on time limit' "$run/stdout_stage1.log"
test -f "$run/scf_rsg_relax.final.rst"
