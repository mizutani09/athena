#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=2
#SBATCH --ntasks=112
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor182_upwind_pulse
#SBATCH --output=/work/mizutaniks/wor182/upwind_pulse_%j.out

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

repository=${WOR182_PULSE_REPOSITORY:-/work/mizutaniks/wor182/upwind_pulse_0a124}
velocity=${WOR182_PULSE_VELOCITY:-1.0e6}
target_time=${WOR182_PULSE_TARGET_TIME:-100.0}
cfl=${WOR182_PULSE_CFL:-0.3}
mode=${WOR182_PULSE_MODE:-moving}
run=/work/mizutaniks/wor182/upwind_pulse_${mode}_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
git rev-parse HEAD > "$run/git_commit.txt"
sha256sum bin/athena inputs/radiation/athinput.nrfld_advect_pulse \
  > "$run/provenance.sha256"
printf 'velocity=%s\ntarget_time=%s\ncfl=%s\nmode=%s\n' \
  "$velocity" "$target_time" "$cfl" "$mode" > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -i inputs/radiation/athinput.nrfld_advect_pulse -d "$run" \
  time/cfl_number="$cfl" time/tlim="$target_time" time/nlim=-1 \
  problem/v0="$velocity" \
  output1/dcycle=1 output2/dt="$target_time" output3/dt="$target_time" \
  nrfld/nr_threshold=1.0e-10 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e
if grep -q 'FATAL ERROR' "$run/stdout.log" "$run/stderr.log"; then
  status=2
fi

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
grep -E '^cycle=|\[NR\] status=|\[LINEAR_FGMRES\]|FATAL ERROR' \
  "$run/stdout.log" "$run/stderr.log" | tail -n 80 || true
exit "$status"
