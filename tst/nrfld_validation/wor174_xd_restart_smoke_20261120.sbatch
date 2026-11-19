#!/usr/bin/env bash
# Same-restart true/false smoke for the clean WOR-174 solar implementation.
#SBATCH --job-name=wor174_solar_ab_smoke
#SBATCH --account=a
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/work/mizutaniks/wor174/xd_ab_20261120/smoke_%j.out

set -euo pipefail
root=/work/mizutaniks/wor174/xd_ab_20261120
repo=/work/mizutaniks/wor174/athena-solar-topface-clean-20261120
binary=$repo/bin/athena
binary_sha=473b9b6ef5eaf5d30783ca9e3a7ea731a69eb4fb02dcb30f89ee781d27a87c8f
commit=f3bfffa4f31d9589ebb2660bed30ba2da02304a8
base=/work/mizutaniks/wor91/thermal_relax_20261116/tomida96_20261119
input=$base/athinput.tomida96_20261119
seed=$base/smoke_342931/out/wor91_tomida96_smoke_342931.final.rst
validator=/work/mizutaniks/wor91/scripts/check_96km_restart.py
run=$root/smoke_${SLURM_JOB_ID}

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

test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test "$(sha256sum "$binary" | awk '{print $1}')" = "$binary_sha"
test "$(sha256sum "$seed" | awk '{print $1}')" = 05be5571fb573ee6dc8bd50156a7efa7f434ba9d20a0f6dc71e102cb0d4a331b
mkdir -p "$run"
/usr/bin/python3 "$validator" "$seed" > "$run/seed_metadata.json"
/usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["time"]==5.0 and d["cycle"]==61' "$run/seed_metadata.json"
cp "$input" "$run/input.used"
sha256sum "$binary" "$input" "$seed" \
  "$base/data/tomida_material_eos.tab" "$base/data/solar_initial_profile.dat" \
  /work/mizutaniks/wor91/data/opacity_table_rhoT_3p0_6p3.h5 > "$run/inputs.sha256"

cd /work/mizutaniks/wor91
for mode in false true; do
  out=$run/$mode
  mkdir -p "$out"
  set +e
  srun --mpi=pmi2 --ntasks=32 --cpu-bind=cores \
    "$binary" -r "$seed" -i "$input" -d "$out" \
    "job/problem_id=wor174_${mode}_${SLURM_JOB_ID}" \
    "problem/reconstruct_top_face_opacity=$mode" \
    time/tlim=5.5 time/nlim=10000000 \
    output1/dt=0.05 output1/next_time=5.05 \
    output2/next_time=100000 output3/next_time=100000 \
    output4/next_time=100000 output5/next_time=100000 \
    output6/next_time=100000 output7/next_time=100000 \
    output8/next_time=100000 > "$out/run.log" 2>&1
  raw=$?
  set -e
  printf 'raw_srun_exit=%s\n' "$raw" > "$out/result.txt"
  test "$raw" -eq 0
  grep -q '^Terminating on time limit' "$out/run.log"
  ! grep -q 'FATAL ERROR' "$out/run.log"
  nr_total=$(grep -c '^\[NR\] status=' "$out/run.log")
  nr_good=$(grep -c '^\[NR\] status=converged ' "$out/run.log")
  test "$nr_total" -gt 0
  test "$nr_total" -eq "$nr_good"
  test -s "$out/wor174_${mode}_${SLURM_JOB_ID}.hst"
  printf 'newton_solves=%s\nvalidated=yes\n' "$nr_total" >> "$out/result.txt"
done

printf 'PASS\n' > "$run/result.txt"
