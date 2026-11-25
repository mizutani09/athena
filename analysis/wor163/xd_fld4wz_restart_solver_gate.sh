#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor163_4wz_rst
#SBATCH --output=/work/mizutaniks/wor163/fld4wz_restart_gate_%j.out

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
restart=/work/mizutaniks/wor163/fld_restart_debug_347568/checkpoint/mesa_scf_rsg_corecut7p00_fld_norot_l3.final.rst
integrator=${FLD4WZ_INTEGRATOR:-rk4}
case $integrator in
  rk4|ssprk5_4) ;;
  *) echo "unsupported FLD4WZ_INTEGRATOR=$integrator" >&2; exit 2 ;;
esac
run=/work/mizutaniks/wor163/fld4wz_${integrator}_restart_gate_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$restart"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$restart" > "$run/provenance.sha256"

# Switch the explicit hydro reconstruction and selected fourth-order integrator
# at the clean cycle-100 checkpoint. Continue far enough to cross the original
# dt=1.6e-4 failure and the later dt=2.56e-3 solver threshold without imposing
# a fixed timestep. FLD4WZ_INTEGRATOR permits an RK4/SSPRK(5,4) A/B test.
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -d "$run" \
  time/integrator="$integrator" time/xorder=4wz \
  time/nlim=110 time/tlim=1.0e-2 time/ncycle_out=1 \
  problem/radiation_startup_dt_end=1.0e-3 \
  fld/explicit_positivity_diagnostics=true \
  nrfld/max_update_fraction=1.0 nrfld/diagnostic_verbosity=1 \
  output1/dcycle=1 output2/dcycle=1000 \
  output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
grep -E '^cycle=|NR.*status=|FATAL ERROR' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
test -n "$(find "$run" -maxdepth 1 -name '*.final.rst' -print -quit)"
cat "$run/time.log"
tail -n 50 "$run/summary.log"
