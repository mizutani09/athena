#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor177_taper1k
#SBATCH --output=/work/mizutaniks/wor177/taper_surface_l2_4wz_1000cycle_%j.out

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
input=inputs/rsg/athinput.mesa_scf_rsg_taper_surface_l2_4wz
profile=data/ic/rsg21_mesa_scf_omega0_nr16384_taper_rho3e-9_gamma5_3.csv
run_root=/work/mizutaniks/wor177/taper_surface_l2_4wz_1000cycle_${SLURM_JOB_ID}
mkdir -p "$run_root"
cd "$repository"
test -x bin/athena

git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
sha256sum bin/athena "$input" "$profile" src/pgen/scf_rsg.cpp \
  analysis/wor162/analyze_relaxation_history.cpp \
  > "$run_root/provenance.sha256"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o "$run_root/analyze_relaxation_history"

run=$run_root/sustained_1000
mkdir -p "$run"
/usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
  -o "$run/time.log" \
  srun -n56 --ntasks-per-node=56 bin/athena -i "$input" -d "$run" \
  time/nlim=1000 time/tlim=10000.0 output1/dcycle=100 \
  > "$run/stdout.log" 2> "$run/stderr.log"
if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo 'sustained_1000 produced a numeric or runtime failure' >&2
  exit 1
fi
if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
  echo 'sustained_1000 emitted a gravity multigrid warning' >&2
  exit 1
fi
grep -q 'Terminating on cycle limit' "$run/stdout.log"
history=$run/mesa_scf_rsg_taper_surface_l2_4wz.hst
test -s "$history"
grep -q 'stellar_radial_mach_max_surface_taper' "$history"
grep -q 'stellar_mass_radial_mach_gt_0p1_g' "$history"
"$run_root/analyze_relaxation_history" "$history" \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"
test -n "$(find "$run" -maxdepth 1 -name '*.rst' -print -quit)"

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run_root/slurm_accounting.txt" || true

cat "$run/time.log"
grep -E 'Number of MeshBlocks|zone-cycles' "$run/stdout.log" | tail -n 8
cat "$run/relaxation_summary.json"
