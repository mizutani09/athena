#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --job-name=wor163_4wz_fld
#SBATCH --output=/work/mizutaniks/wor163/fld4wz_cfl03_long_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5
export SLURM_MPI_TYPE=pmi2
export LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
ulimit -c 0

repository=$HOME/simulation/athena-rsg-corecut7p00-fld4wz
restart=${FLD4WZ_RESTART:-/work/mizutaniks/wor163/fld4wz_cfl0p3_t1.0_347652/mesa_scf_rsg_corecut7p00_fld_norot_l3.final.rst}
history_header=/work/mizutaniks/wor163/fld_norot_startup_347544/mesa_scf_rsg_corecut7p00_fld_norot_l3.hst
target_time=${FLD4WZ_TARGET_TIME:-705.090465}
cfl=${FLD4WZ_CFL:-0.3}
slice_dt=23.5030155
run=/work/mizutaniks/wor163/fld4wz_cfl03_long_${SLURM_JOB_ID}
problem_id=mesa_scf_rsg_corecut7p00_fld_norot_l3

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$restart"
test -s "$history_header"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
printf 'restart=%s\ntarget_time=%s\ncfl=%s\n' \
  "$restart" "$target_time" "$cfl" > "$run/run_config.txt"
sha256sum bin/athena "$restart" "$history_header" \
  analysis/wor162/analyze_relaxation_history.cpp > "$run/provenance.sha256"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o "$run/analyze_relaxation_history"

# One and a half dynamical times at the requested hydro CFL.  Implicit FLD has
# no fixed dt cap; nonlinear convergence is checked independently every step.
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -d "$run" \
  time/integrator=rk4 time/xorder=4wz \
  time/cfl_number="$cfl" time/nlim=-1 time/tlim="$target_time" \
  time/ncycle_out=50 \
  fld/mixed_frame_transport=false \
  nrfld/max_update_fraction=1.0 \
  nrfld/nr_backtrack_max=10 nrfld/nr_min_step_scale=1.0e-3 \
  nrfld/diagnostic_verbosity=0 \
  output1/dcycle=10 output2/dcycle=250 \
  output3/dt="$slice_dt" output3/next_time="$slice_dt" \
  output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo "long CFL=0.3 stability run produced a numeric or runtime failure" >&2
  exit 1
fi
test -s "$run/${problem_id}.final.rst"

"$run/analyze_relaxation_history" \
  "$run/${problem_id}.hst" \
  "$run/stability_diagnostics.csv" "$run/stability_summary.json" \
  1.03413107e7 "$history_header"
grep -E '^cycle=|\[NR\] status=' "$run/stdout.log" > "$run/solver_summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
cat "$run/stability_summary.json"
