#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor163_fld_dbg
#SBATCH --output=/work/mizutaniks/wor163/fld_restart_debug_%j.out

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

repository=$HOME/simulation/athena-rsg-corecut7p00-fld
restart=/work/mizutaniks/wor163/fld_norot_startup_347554/mesa_scf_rsg_corecut7p00_fld_norot_l3.final.rst
run=/work/mizutaniks/wor163/fld_restart_debug_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$restart"
git rev-parse HEAD > "$run/git_commit.txt"
sha256sum bin/athena "$restart" > "$run/provenance.sha256"

# The restart is at t ~= 4.52e-3 with dt=1e-5. Releasing the user cap below
# the restart time reproduces the native 2x timestep growth in only a few cycles.
set +e
/usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
  -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -d "$run" \
  time/nlim=470 time/tlim=1.0e-2 time/ncycle_out=1 \
  problem/radiation_startup_dt_end=4.5e-3 \
  nrfld/diagnostic_verbosity=2 \
  output1/dcycle=1 output2/dcycle=1000 \
  output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

printf '%s\n' "$status" > "$run/srun_status.txt"
grep -E 'cycle=|NR_INITIAL|NR_TRIAL|max_defect_loc|max_defect_stencil|face_neighbor_levels|diff_contrib|coeff center|src center|cell_physics|NR_GAS_FAILURE|NR.*status=|FATAL ERROR' \
  "$run/stdout.log" > "$run/diagnostic_summary.log" || true
cat "$run/time.log"
tail -n 40 "$run/stdout.log"
tail -n 20 "$run/stderr.log"

# A nonzero Athena status is the expected diagnostic outcome; preserve the
# captured status while allowing Slurm to retain all postprocessed artifacts.
exit 0
