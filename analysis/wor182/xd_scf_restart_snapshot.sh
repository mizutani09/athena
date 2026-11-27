#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor182_snapshot
#SBATCH --output=/work/mizutaniks/wor182/scf_snapshot_%j.out

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
binary=${WOR182_BINARY:-bin/athena}
restart=${WOR182_RESTART:?set WOR182_RESTART}
model_tag=${WOR182_MODEL_TAG:?set WOR182_MODEL_TAG}
equatorial_tab=${WOR182_EQUATORIAL_TAB:-false}
overlay=inputs/rsg/athinput.scf_rsg_diagnostic_snapshot
run=/work/mizutaniks/wor182/${model_tag}_snapshot_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x "$binary"
test -s "$restart"
test -s "$overlay"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum "$binary" "$restart" "$overlay" > "$run/provenance.sha256"
printf 'restart=%s\nmodel_tag=%s\nequatorial_tab=%s\n' \
  "$restart" "$model_tag" "$equatorial_tab" > "$run/run_config.txt"

output_arguments=(output3/dt=-1 output4/dt=-1)
if [[ "$equatorial_tab" == true ]]; then
  # output3/output4 are the primitive and user-output equatorial slices stored
  # in the restart.  A cycle trigger emits them without advancing the solution.
  output_arguments+=(output3/dcycle=1 output4/dcycle=1)
elif [[ "$equatorial_tab" != false ]]; then
  echo 'WOR182_EQUATORIAL_TAB must be true or false' >&2
  exit 2
fi

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  "$binary" -r "$restart" -i "$overlay" -d "$run" \
  time/tlim=0.0 time/nlim=-1 \
  "${output_arguments[@]}" \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

if grep -Eq 'FATAL ERROR|^scf_rsg:' "$run/stdout.log" "$run/stderr.log"; then
  status=1
fi
for id in diagnostic_prim diagnostic_uov; do
  output=$(find "$run" -maxdepth 1 -name "*${id}*.athdf" -print -quit)
  if [[ -z "$output" || ! -s "$output" ]]; then
    echo "missing ${id} HDF5 snapshot" >&2
    status=1
  else
    sha256sum "$output" >> "$run/snapshots.sha256"
  fi
done
if [[ "$equatorial_tab" == true ]]; then
  for id in out3 out4; do
    output=$(find "$run" -maxdepth 1 -name "*.${id}.*.tab" -print -quit)
    if [[ -z "$output" || ! -s "$output" ]]; then
      echo "missing ${id} equatorial tab snapshot" >&2
      status=1
    fi
  done
fi
cat "$run/time.log" || true
exit "$status"
