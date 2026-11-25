#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor163_4wz_cfl
#SBATCH --output=/work/mizutaniks/wor163/fld4wz_cfl_stability_%j.out

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
restart=${FLD4WZ_RESTART:-/work/mizutaniks/wor163/fld4wz_rk4_restart_gate_347626/mesa_scf_rsg_corecut7p00_fld_norot_l3.final.rst}
target_time=${FLD4WZ_TARGET_TIME:-1.0e-1}
cfl=${FLD4WZ_CFL:-0.3}
cfl_tag=${cfl//./p}
run=/work/mizutaniks/wor163/fld4wz_cfl${cfl_tag}_t${target_time}_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$restart"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
printf 'restart=%s\ntarget_time=%s\ncfl=%s\n' \
  "$restart" "$target_time" "$cfl" > "$run/run_config.txt"
sha256sum bin/athena "$restart" > "$run/provenance.sha256"

# Continue with no imposed dt.  The ordinary Athena++ CFL estimate and its
# timestep growth limiter control the approach to the requested CFL number.
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -d "$run" \
  time/integrator=rk4 time/xorder=4wz \
  time/cfl_number="$cfl" time/nlim=-1 time/tlim="$target_time" \
  time/ncycle_out=1 \
  fld/mixed_frame_transport=false \
  nrfld/max_update_fraction=1.0 \
  nrfld/nr_backtrack_max=10 nrfld/nr_min_step_scale=1.0e-3 \
  nrfld/diagnostic_verbosity=0 \
  output1/dcycle=1 output2/dcycle=100 \
  output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo "CFL stability gate produced a numeric or runtime failure" >&2
  exit 1
fi
test -n "$(find "$run" -maxdepth 1 -name '*.final.rst' -print -quit)"
grep -E '^cycle=|\[NR\] status=' "$run/stdout.log" > "$run/summary.log" || true
cat "$run/time.log"
tail -n 40 "$run/summary.log"
