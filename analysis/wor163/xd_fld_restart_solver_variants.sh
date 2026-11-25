#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor163_fld_var
#SBATCH --output=/work/mizutaniks/wor163/fld_solver_variants_%j.out

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
restart=/work/mizutaniks/wor163/fld_restart_debug_347568/checkpoint/mesa_scf_rsg_corecut7p00_fld_norot_l3.final.rst
root=/work/mizutaniks/wor163/fld_solver_variants_${SLURM_JOB_ID}

mkdir -p "$root"
cd "$repository"
test -x bin/athena
test -s "$restart"
git rev-parse HEAD > "$root/git_commit.txt"
sha256sum bin/athena "$restart" > "$root/provenance.sha256"

run_variant() {
  local label=$1
  shift
  local run=$root/$label
  mkdir -p "$run"
  set +e
  /usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
    srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
    bin/athena -r "$restart" -d "$run" \
    time/nlim=110 time/tlim=1.0e-2 time/ncycle_out=1 \
    problem/radiation_startup_dt_end=1.0e-3 \
    nrfld/diagnostic_verbosity=1 \
    output1/dcycle=1 output2/dcycle=1000 \
    output3/dt=-1 output4/dt=-1 \
    "$@" > "$run/stdout.log" 2> "$run/stderr.log"
  local status=$?
  set -e
  printf '%s\n' "$status" > "$run/srun_status.txt"
  grep -E '^cycle=|NR_INITIAL|NR_TRIAL|NR.*status=|FATAL ERROR' \
    "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
}

# Keep every full trial positive without touching the implementation: unlike the
# previous factor of 10, these caps cannot drive a positive Er through zero.
run_variant cap_0p5 \
  nrfld/max_update_fraction=0.5 \
  nrfld/nr_backtrack_max=10 nrfld/nr_min_step_scale=1.0e-3
run_variant cap_1p0 \
  nrfld/max_update_fraction=1.0 \
  nrfld/nr_backtrack_max=10 nrfld/nr_min_step_scale=1.0e-3

for label in cap_0p5 cap_1p0; do
  echo "===== $label ====="
  cat "$root/$label/time.log"
  cat "$root/$label/srun_status.txt"
  grep -E '^cycle=|NR.*status=|FATAL ERROR' "$root/$label/summary.log" | tail -n 30
done
