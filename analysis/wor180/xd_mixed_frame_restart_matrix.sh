#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor180_mixed
#SBATCH --output=/work/mizutaniks/wor180/mixed_restart_%j.out

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
export ATHENA_FLD_EXPLICIT_POSITIVITY_DIAGNOSTICS=1
ulimit -c 0

repository=${WOR180_REPOSITORY:-$HOME/simulation/athena-wor180-mixed-frame}
restart=${WOR180_RESTART:-/work/mizutaniks/wor163/fld_restart_debug_347568/checkpoint/mesa_scf_rsg_corecut7p00_fld_norot_l3.final.rst}
mixed_source=${WOR180_MIXED_SOURCE:-true}
mixed_transport=${WOR180_MIXED_TRANSPORT:-true}
nlim=${WOR180_NLIM:-110}
tlim=${WOR180_TLIM:-1.0e-2}
case $mixed_source:$mixed_transport in
  true:true|false:true|false:false) ;;
  true:false)
    echo 'source-only is unsupported because MixedFrameCoupled requires transport' >&2
    exit 2
    ;;
  *) echo "invalid mixed-frame switches: $mixed_source $mixed_transport" >&2; exit 2 ;;
esac

run=/work/mizutaniks/wor180/mixed_source-${mixed_source}_transport-${mixed_transport}_${SLURM_JOB_ID}
mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$restart"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$restart" > "$run/provenance.sha256"
{
  echo "mixed_source=$mixed_source"
  echo "mixed_transport=$mixed_transport"
  echo "restart=$restart"
  echo "nlim=$nlim"
  echo "tlim=$tlim"
} > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -d "$run" \
  time/integrator=rk4 time/xorder=4wz \
  time/nlim="$nlim" time/tlim="$tlim" time/ncycle_out=1 \
  problem/radiation_startup_dt_end=1.0e-3 \
  fld/include_mixed_frame_terms="$mixed_source" \
  fld/mixed_frame_transport="$mixed_transport" \
  nrfld/max_update_fraction=1.0 \
  nrfld/nr_backtrack_max=10 nrfld/nr_min_step_scale=1.0e-3 \
  nrfld/diagnostic_verbosity=0 \
  output1/dcycle=1 output2/dcycle=1000 output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

grep -E '^cycle=|NR.*status=|FATAL ERROR|Er_old=|outgoing_delta=' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
exit "$status"
