#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor177_l3_c6
#SBATCH --output=/work/mizutaniks/wor177/surface_l3_central_l6_gate_%j.out

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
run=/work/mizutaniks/wor177/surface_l3_central_l6_gate_${SLURM_JOB_ID}
mkdir -p "$run"
cd "$repository"
test -x bin/athena

git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena data/ic/rsg21_mesa_scf_omega0_nr16384.csv \
  inputs/rsg/athinput.scf_rsg_relaxation_surface_l3 \
  > "$run/provenance.sha256"

# Reuse the verified central-L6/surface-L3 topology while overriding all
# source-model quantities with the WOR-175 MESA-fit nonrotating equilibrium.
/usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' -o "$run/time.log" \
  srun -n56 --ntasks-per-node=56 bin/athena \
  -i inputs/rsg/athinput.scf_rsg_relaxation_surface_l3 -d "$run" \
  problem/scf_profile_file=data/ic/rsg21_mesa_scf_omega0_nr16384.csv \
  problem/angular_velocity_rad_s=0.0 \
  problem/cell_quadrature_order=1 \
  problem/atmosphere_reference_radius_cm=6.4959871737483445e13 \
  problem/surface_polar_radius_cm=6.4959871737483445e13 \
  problem/surface_refinement_use_tracer=false \
  problem/surface_refinement_density_g_cm3=1.0e-9 \
  problem/gas_mass_g=3.2091112192814024e34 \
  problem/core_mass_g=9.665283322407383e33 \
  problem/core_support_radius_cm=5.1263110134271515e10 \
  job/problem_id=mesa_l3_c6 time/nlim=0 time/tlim=1.0 \
  output1/dcycle=1 output2/dt=-1 output3/dt=-1 \
  output4/dt=-1 output5/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo 'surface-L3/central-L6 gate failed' >&2
  exit 1
fi
grep -Eq 'Terminating on (cycle|time) limit' "$run/stdout.log"
test -s "$run/mesa_l3_c6.hst"
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 16 "$run/stdout.log"
tail -n 2 "$run/mesa_l3_c6.hst"
