#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor177_surf_scale
#SBATCH --output=/work/mizutaniks/wor177/surface_level_scaling_gate_%j.out

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
run_root=/work/mizutaniks/wor177/surface_level_scaling_gate_${SLURM_JOB_ID}
mkdir -p "$run_root"
cd "$repository"
test -x bin/athena

git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
sha256sum bin/athena data/ic/rsg21_mesa_scf_omega0_nr16384.csv \
  inputs/rsg/athinput.mesa_scf_rsg_surface_amr > "$run_root/provenance.sha256"

for level in 0 1 2; do
  run=$run_root/l${level}
  mkdir -p "$run"
  /usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
    -o "$run/time.log" \
    srun -n56 --ntasks-per-node=56 bin/athena \
    -i inputs/rsg/athinput.mesa_scf_rsg_surface_amr -d "$run" \
    problem/surface_refinement_level="$level" \
    problem/cell_quadrature_order=1 \
    job/problem_id="surface_l${level}" time/nlim=0 time/tlim=1.0 \
    output1/dcycle=1 > "$run/stdout.log" 2> "$run/stderr.log"
  grep -Eq 'Terminating on (cycle|time) limit' "$run/stdout.log"
  test -s "$run/surface_l${level}.hst"
  if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
      "$run/stdout.log" "$run/stderr.log"; then
    echo "surface-L${level} scaling gate failed" >&2
    exit 1
  fi
  printf 'surface_L%s ' "$level"
  cat "$run/time.log"
  tail -n 1 "$run/surface_l${level}.hst"
done

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run_root/slurm_accounting.txt" || true
