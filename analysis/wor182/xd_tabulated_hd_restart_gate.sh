#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor182_hd_cont
#SBATCH --output=/work/mizutaniks/wor182/hd_eos_continue_%j.out

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

repository=${WOR182_REPOSITORY:-$HOME/simulation/athena-wor182-lte-radiation-eos}
restart=${WOR182_RESTART:?set WOR182_RESTART to an accepted HD checkpoint}
model_tag=${WOR182_MODEL_TAG:?set WOR182_MODEL_TAG}
target_time=${WOR182_TARGET_TIME:-1.0}
cfl=${WOR182_CFL:-0.2}
nlim=${WOR182_NLIM:--1}
restart_dcycle=${WOR182_RESTART_DCYCLE:-1000}
time_tag=${target_time//./p}
run=/work/mizutaniks/wor182/${model_tag}_hd_t${time_tag}_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$restart"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$restart" > "$run/provenance.sha256"
printf 'restart=%s\nmodel_tag=%s\ntarget_time=%s\ncfl=%s\nnlim=%s\nrestart_dcycle=%s\n' \
  "$restart" "$model_tag" "$target_time" "$cfl" "$nlim" \
  "$restart_dcycle" > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -d "$run" \
  time/tlim="$target_time" time/cfl_number="$cfl" time/nlim="$nlim" \
  time/ncycle_out=1 output1/dcycle=1 output2/dcycle="$restart_dcycle" \
  output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

if grep -Eq 'FATAL ERROR|^scf_rsg:' "$run/stdout.log" "$run/stderr.log"; then
  status=1
fi
grep -E '^cycle=|FATAL ERROR|^scf_rsg:' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
tail -n 3 "$run"/*.hst > "$run/history_tail.log" 2>/dev/null || true
exit "$status"
