#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --job-name=wor177_t013
#SBATCH --output=/work/mizutaniks/wor177/taper_surface_l2_4wz_continue_t013_%j.out

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
source_run=/work/mizutaniks/wor177/taper_surface_l2_4wz_1000cycle_344410/sustained_1000
restart=$source_run/mesa_scf_rsg_taper_surface_l2_4wz.final.rst
source_history=$source_run/mesa_scf_rsg_taper_surface_l2_4wz.hst
# 0.13 * (1.03413107e7 s / 2.19999657e4 s) = 61.1078047 code time.
target_time=61.1078047
run_root=/work/mizutaniks/wor177/taper_surface_l2_4wz_continue_t013_${SLURM_JOB_ID}
run=$run_root/continuation
problem=mesa_scf_rsg_taper_surface_l2_4wz

test -s "$restart"
test -s "$source_history"
mkdir -p "$run"
# Athena++ appends to an existing history file.  Seed it with the accepted
# 1,000-cycle history so every diagnostic is continuous from the initial map.
cp "$source_history" "$run/$problem.hst"
cd "$repository"
test -x bin/athena

git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
sha256sum bin/athena "$restart" "$source_history" \
  inputs/rsg/athinput.mesa_scf_rsg_taper_surface_l2_4wz \
  data/ic/rsg21_mesa_scf_omega0_nr16384_taper_rho3e-9_gamma5_3.csv \
  src/pgen/scf_rsg.cpp src/reconstruct/wenoz_berta.cpp \
  src/eos/eos_high_order.cpp analysis/wor162/analyze_relaxation_history.cpp \
  > "$run_root/provenance.sha256"
cat > "$run_root/target.txt" <<EOF
target_time_code=$target_time
target_time_over_tdyn=0.13
source_time_over_tdyn=0.002150176492321
requested_nodes=$SLURM_JOB_NUM_NODES
requested_ranks=$SLURM_NTASKS
EOF

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o "$run_root/analyze_relaxation_history"

# With the measured one-node cost (7.21 wall-days/t_dyn), ideal three-node
# scaling predicts 7.4 h and a conservative 2.2x speedup predicts 9.9 h.
# A restart every 5,000 cycles limits lost work if the timestep contracts.
/usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
  -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -d "$run" \
  time/integrator=rk4 time/xorder=4wz \
  time/nlim=-1 time/tlim="$target_time" \
  gravity/mgmode=fmg gravity/threshold=-1.0 gravity/niteration=0 \
  output1/dcycle=250 output2/dcycle=5000 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo 'continuation produced a numeric or runtime failure' >&2
  exit 1
fi
if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
  echo 'continuation unexpectedly entered discarded multigrid polishing iterations' >&2
  exit 1
fi
test -s "$run/$problem.final.rst"
test -s "$run/$problem.hst"
"$run_root/analyze_relaxation_history" "$run/$problem.hst" \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run_root/slurm_accounting.txt" || true

cat "$run/time.log"
grep -E 'Number of MeshBlocks|zone-cycles|cpu time used' \
  "$run/stdout.log" | tail -n 8
cat "$run/relaxation_summary.json"
