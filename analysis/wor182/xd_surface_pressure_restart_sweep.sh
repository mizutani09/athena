#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:10:00
#SBATCH --job-name=wor182_p_sweep
#SBATCH --output=/work/mizutaniks/wor182/surface_pressure_sweep_%j.out

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

repository=${WOR182_REPOSITORY:-$HOME/simulation/athena-wor182-radiative-relaxation}
input=inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated
run_root=/work/mizutaniks/wor182/surface_pressure_sweep_${SLURM_JOB_ID}

labels=(time1 time20 time50 time100)
restarts=(
  /work/mizutaniks/wor182/q025_corot_tabulated_eos_348051/mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated.final.rst
  /work/mizutaniks/wor182/q025_corot_tabulated_eos_cfl0p3_t20p0_348055/mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated.final.rst
  /work/mizutaniks/wor182/q025_corot_tabulated_eos_cfl0p3_t50p0_348057/mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated.final.rst
  /work/mizutaniks/wor182/q025_corot_tabulated_eos_cfl0p3_t100p0_348059/mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated.final.rst
)

mkdir -p "$run_root"
cd "$repository"
test -x bin/athena
test -s "$input"
git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
sha256sum bin/athena "$input" > "$run_root/provenance.sha256"

for index in "${!labels[@]}"; do
  label=${labels[$index]}
  restart=${restarts[$index]}
  run=$run_root/$label
  mkdir -p "$run"
  test -s "$restart"
  sha256sum "$restart" > "$run/restart.sha256"
  set +e
  /usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
    srun -n168 --ntasks-per-node=56 \
    bin/athena -r "$restart" -i "$input" -d "$run" \
    time/nlim=1 time/tlim=1.0e9 time/ncycle_out=1 \
    problem/surface_mach_diagnostic_threshold=-1.0 \
    problem/surface_pressure_ratio_diagnostic_threshold=0.95 \
    problem/taper_exchange_diagnostic_time_threshold_s=-1.0 \
    output1/dcycle=1 output2/dt=-1 output3/dt=-1 output4/dt=-1 \
    > "$run/stdout.log" 2> "$run/stderr.log"
  status=$?
  set -e
  if grep -q 'FATAL ERROR' "$run/stdout.log" "$run/stderr.log"; then
    status=1
  fi
  grep 'SCF_SURFACE_PRESSURE_CELL' "$run/stdout.log" \
    > "$run/surface_pressure_cells.log" || true
  printf '%s reports=%s ' "$label" \
    "$(wc -l < "$run/surface_pressure_cells.log")"
  cat "$run/time.log"
  if test "$status" -ne 0; then
    tail -n 80 "$run/stdout.log" "$run/stderr.log"
    exit "$status"
  fi
done
