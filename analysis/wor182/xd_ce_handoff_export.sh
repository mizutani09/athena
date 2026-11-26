#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor182_ce_export
#SBATCH --output=/work/mizutaniks/wor182/ce_handoff_%j.out

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
binary=${WOR182_BINARY:-bin/athena}
require_float64=${WOR182_REQUIRE_FLOAT64:-true}
restart=${WOR182_RESTART:?set WOR182_RESTART to the accepted relaxed checkpoint}
model_tag=${WOR182_MODEL_TAG:?set WOR182_MODEL_TAG}
overlay=inputs/rsg/athinput.scf_rsg_ce_handoff
run=/work/mizutaniks/wor182/${model_tag}_ce_handoff_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x "$binary"
test -s "$restart"
test -s "$overlay"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum "$binary" "$restart" "$overlay" > "$run/provenance.sha256"
printf '%s\n' \
  'rho_unit_g_cm3=1.0e-8' \
  'energy_density_unit_erg_cm3=1.0e5' \
  'length_unit_cm=6.957e10' \
  'velocity_unit_cm_s=3.162277660168379e6' \
  'time_unit_s=2.199992312698701e4' \
  'momentum_density_unit_g_cm2_s=3.162277660168379e-2' \
  'data=conserved hydro, passive scalars, and self-gravity potential on the native AMR mesh' \
  > "$run/handoff_units.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  "$binary" -r "$restart" -i "$overlay" -d "$run" \
  time/tlim=0.0 time/nlim=-1 output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

if grep -Eq 'FATAL ERROR|^scf_rsg:' "$run/stdout.log" "$run/stderr.log"; then
  status=1
fi
handoff=$(find "$run" -maxdepth 1 -name '*ce_handoff*.athdf' -print -quit)
if [[ -z "$handoff" || ! -s "$handoff" ]]; then
  echo 'full-volume CE handoff HDF5 was not created' >&2
  status=1
else
  sha256sum "$handoff" > "$run/handoff.sha256"
  mpicxx -O2 -std=c++11 analysis/wor182/check_ce_handoff.cpp -lhdf5 \
    -o "$run/check_ce_handoff"
  precision_argument=()
  if [[ "$require_float64" == true ]]; then
    precision_argument+=(--require-float64)
  elif [[ "$require_float64" != false ]]; then
    echo 'WOR182_REQUIRE_FLOAT64 must be true or false' >&2
    status=1
  fi
  if ! "$run/check_ce_handoff" "$handoff" "${precision_argument[@]}" \
      > "$run/handoff_structure.txt"; then
    status=1
  fi
fi
cat "$run/time.log" || true
exit "$status"
