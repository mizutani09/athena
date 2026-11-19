#!/usr/bin/env bash
# Launcher-only smoke for two concurrent one-node Intel-MPI steps on XD.
# Athena initializes MPI before handling -h, so this reproduces the PMI path
# without advancing or writing a physical simulation.
#SBATCH --job-name=wor174_pmi_smoke
#SBATCH --account=a
#SBATCH --partition=M-test-a
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/work/mizutaniks/wor174/xd_ab_20261120/pmi_smoke_%j.out

set -euo pipefail
root=/work/mizutaniks/wor174/xd_ab_20261120
repo=/work/mizutaniks/wor174/athena-solar-topface-clean-20261120
binary=$repo/bin/athena
binary_sha=473b9b6ef5eaf5d30783ca9e3a7ea731a69eb4fb02dcb30f89ee781d27a87c8f
run=$root/pmi_smoke_${SLURM_JOB_ID}

source /etc/profile.d/modules.sh
module use /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load umf/1.0.2 PrgEnv-oneapi/2025.1 hdf5-parallel/1.14.5
export PATH="${I_MPI_ROOT}/bin:${PATH}"
for var in CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
  IFS=: read -r -a entries <<< "${!var:-}"
  filtered=()
  for entry in "${entries[@]}"; do
    case "$entry" in /work/opt/local/miniconda3/*) ;; *) filtered+=("$entry") ;; esac
  done
  export "$var=$(IFS=:; printf '%s' "${filtered[*]}")"
done
export OMP_NUM_THREADS=1 I_MPI_PMI_LIBRARY=/lib64/libpmi2.so

test "$(sha256sum "$binary" | awk '{print $1}')" = "$binary_sha"
mkdir -p "$run"
launch() {
  mode=$1
  srun --mpi=pmi2 --exclusive --nodes=1 --ntasks=32 --ntasks-per-node=32 \
    --cpu-bind=cores --job-name="wor174_pmi_$mode" \
    "$binary" -h > "$run/$mode.log" 2>&1
}

launch false &
pid_false=$!
launch true &
pid_true=$!
set +e
wait "$pid_false"; status_false=$?
wait "$pid_true"; status_true=$?
set -e
printf 'false=%s\ntrue=%s\n' "$status_false" "$status_true" > "$run/step_status.txt"
test "$status_false" -eq 0
test "$status_true" -eq 0
for mode in false true; do
  grep -q '^Athena++' "$run/$mode.log"
  ! grep -q 'PMI2_Job_GetId\|MPI Initialization failed\|Abort(' "$run/$mode.log"
done
printf 'PASS\n' > "$run/result.txt"
