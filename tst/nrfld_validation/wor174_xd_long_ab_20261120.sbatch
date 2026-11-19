#!/usr/bin/env bash
# Mature-state, same-restart, concurrent true/false A/B for WOR-174.
#SBATCH --job-name=wor174_solar_long_ab
#SBATCH --account=a
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=2
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/mizutaniks/wor174/xd_ab_20261120/long_%j.out

set -euo pipefail
root=/work/mizutaniks/wor174/xd_ab_20261120
repo=/work/mizutaniks/wor174/athena-solar-topface-clean-20261120
binary=$repo/bin/athena
binary_sha=473b9b6ef5eaf5d30783ca9e3a7ea731a69eb4fb02dcb30f89ee781d27a87c8f
commit=f3bfffa4f31d9589ebb2660bed30ba2da02304a8
wor91=/work/mizutaniks/wor91
input=$wor91/thermal_relax_20261116/athinput.hd3.topface
source_run=$wor91/thermal_relax_20261116/hd2_warm96_long_342616
seed=$source_run/out/wor91_hd2_warm96_long_342616.final.rst
validator=$wor91/scripts/check_96km_restart.py
run=$root/long_${SLURM_JOB_ID}

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
test "$(sha256sum "$input" | awk '{print $1}')" = cce3b1327fc1b529549209f175e3f4e071d6a4f5233260626bb5b915427f9ae6
test "$(sha256sum "$seed" | awk '{print $1}')" = b47f0a9199be9b457c03ea03e588a75fb2e5942664cc8f8d4f8a62ca44277a4d
grep -qx 'validated=yes' "$source_run/result.txt"
mkdir -p "$run/false" "$run/true"
/usr/bin/python3 "$validator" "$seed" > "$run/seed_metadata.json"
/usr/bin/python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["time"]==65000.0 and d["cycle"]==966669' "$run/seed_metadata.json"
cp "$input" "$run/input.used"
sha256sum "$binary" "$input" "$seed" \
  "$wor91/data/AthenaNataEosTable_solar_deep.tab" \
  "$wor91/data/solar_initial_profile.dat" \
  "$wor91/data/opacity_table_rhoT_3p0_6p3.h5" > "$run/inputs.sha256"
printf 'source_run=%s\nstart_time=65000\nend_time=66000\ns_in=1.6625e9\nPBND=235.97589\n' \
  "$source_run" > "$run/config.txt"

cd "$wor91"
run_mode() {
  mode=$1
  out=$run/$mode
  set +e
  srun --exclusive --nodes=1 --ntasks=32 --ntasks-per-node=32 --cpu-bind=cores \
    --job-name="wor174_$mode" \
    "$binary" -r "$seed" -i "$input" -d "$out" -t 01:55:00 \
    "job/problem_id=wor174_${mode}_${SLURM_JOB_ID}" \
    problem/bottom_bc_mode=rempel_hd2 problem/bottom_heating_cgs=0 \
    problem/PBND=235.97589 \
    problem/s_in_from_profile=false problem/s_in=1.6625e9 \
    "problem/reconstruct_top_face_opacity=$mode" problem/rad_top_alpha=0.56 \
    problem/boundary_diagnostic_verbosity=1 nrfld/nr_threshold=5e-6 \
    output1/dt=0.5 output1/next_time=65000.5 \
    output2/next_time=100000 output3/next_time=100000 \
    output4/next_time=100000 output5/next_time=100000 \
    output6/dt=500 output6/next_time=65500 \
    output7/dt=500 output7/next_time=65500 \
    output8/dt=500 output8/next_time=65500 \
    time/tlim=66000 time/nlim=10000000 > "$out/run.log" 2>&1
  status=$?
  set -e
  printf '%s\n' "$status" > "$out/raw_status.txt"
  return "$status"
}

run_mode false &
pid_false=$!
run_mode true &
pid_true=$!
set +e
wait "$pid_false"; status_false=$?
wait "$pid_true"; status_true=$?
set -e
printf 'false=%s\ntrue=%s\n' "$status_false" "$status_true" > "$run/step_status.txt"
test "$status_false" -eq 0
test "$status_true" -eq 0

for mode in false true; do
  out=$run/$mode
  grep -q '^Terminating on time limit' "$out/run.log"
  ! grep -q 'FATAL ERROR' "$out/run.log"
  nr_total=$(grep -c '^\[NR\] status=' "$out/run.log")
  nr_good=$(grep -c '^\[NR\] status=converged ' "$out/run.log")
  test "$nr_total" -gt 0
  test "$nr_total" -eq "$nr_good"
  final=$(find "$out" -maxdepth 1 -name '*.final.rst' -print -quit)
  test -s "$final"
  /usr/bin/python3 "$validator" "$final" > "$out/restart_metadata.json"
  /usr/bin/python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["time"]==66000.0' "$out/restart_metadata.json"
  printf 'newton_solves=%s\nvalidated=yes\n' "$nr_total" > "$out/result.txt"
done
printf 'PASS\n' > "$run/result.txt"
