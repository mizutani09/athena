#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=2
#SBATCH --ntasks=112
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor177_l3_c7
#SBATCH --output=/work/mizutaniks/wor177/surface_l3_central_l7_gate_%j.out

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

repository=$HOME/simulation/athena-wor177-mesa-hd4wz
run=/work/mizutaniks/wor177/surface_l3_central_l7_gate_${SLURM_JOB_ID}
mkdir -p "$run"
cd "$repository"
test -x bin/athena

git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena data/ic/rsg21_mesa_scf_omega0_nr16384.csv \
  inputs/rsg/athinput.mesa_scf_rsg_surface_amr_l7 \
  > "$run/provenance.sha256"

/usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' -o "$run/time.log" \
  srun -n112 --ntasks-per-node=56 bin/athena \
  -i inputs/rsg/athinput.mesa_scf_rsg_surface_amr_l7 -d "$run" \
  time/nlim=0 time/tlim=1.0 output1/dcycle=1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo 'surface-L3/central-L7 gate failed' >&2
  exit 1
fi
grep -Eq 'Terminating on (cycle|time) limit' "$run/stdout.log"
test -s "$run/mesa_scf_rsg_surface_l3_central_l7.hst"
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 16 "$run/stdout.log"
tail -n 2 "$run/mesa_scf_rsg_surface_l3_central_l7.hst"
