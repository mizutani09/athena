#!/usr/bin/env bash
# Continue the validated WOR-91 NATA 24 km state toward t=71000.
# SOURCE_RUN must name the immediately preceding validated run directory.
#SBATCH --job-name=wor91_nata24_chain
#SBATCH --account=a
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=2
#SBATCH --ntasks=112
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --output=/work/mizutaniks/wor91/thermal_relax_20261116/hd2_nata24_20261119/chain_%j.out

set -euo pipefail
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
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

root=/work/mizutaniks/wor91
base="$root/thermal_relax_20261116/hd2_nata24_20261119"
: "${SOURCE_RUN:?SOURCE_RUN is required}"
source_run="$SOURCE_RUN"
seed=$(find "$source_run/out" -maxdepth 1 -name '*.final.rst' -print -quit)
repo=/home/mizutaniks/simulation/athena-wor91-refine24-20261119
binary="$repo/bin/athena"
input="$base/athinput.hd3.refine24_20261119"
validator="$root/scripts/check_amr24_restart.py"
run="$base/chain_${SLURM_JOB_ID}"

grep -qx 'validated=yes' "$source_run/result.txt"
test -s "$seed"
mkdir -p "$run/out"
/usr/bin/python3 "$validator" "$seed" 70000 > "$run/seed_metadata.json"
if /usr/bin/python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["time"]>=71000-1e-6 else 1)' "$run/seed_metadata.json"; then
  ln -s "$seed" "$run/out/wor91_nata24_chain_${SLURM_JOB_ID}.final.rst"
  printf 'job=%s\nsource_run=%s\nalready_complete=yes\nvalidated=yes\ntarget_reached=yes\n' \
    "$SLURM_JOB_ID" "$source_run" > "$run/result.txt"
  exit 0
fi

read -r next20 next100 next200 < <(/usr/bin/python3 -c \
  'import json,math,sys; t=json.load(open(sys.argv[1]))["time"]; print(*(math.floor(t/d+1)*d for d in (20,100,200)))' \
  "$run/seed_metadata.json")
cd "$repo"
sha256sum -c "$base/build.sha256" > "$run/build_check.log"
sha256sum "$binary" "$input" "$seed" \
  "$root/data/AthenaNataEosTable_solar_deep.tab" \
  "$root/data/opacity_table_rhoT_3p0_6p3.h5" > "$run/inputs.sha256"
printf 'source_run=%s\ntarget_time=71000\ns_in=1.6625e9\nPBND=235.97589\n' \
  "$source_run" > "$run/config.txt"

cd "$root"
set +e
srun --mpi=pmi2 --ntasks=112 --cpu-bind=cores \
  "$binary" -r "$seed" -i "$input" -d "$run/out" -t 11:35:00 \
  "job/problem_id=wor91_nata24_chain_${SLURM_JOB_ID}" \
  problem/bottom_bc_mode=rempel_hd2 problem/bottom_heating_cgs=0 \
  problem/PBND=235.97589 problem/s_in_from_profile=false problem/s_in=1.6625e9 \
  problem/reconstruct_top_face_opacity=true problem/rad_top_alpha=0.56 \
  problem/refine_all_one_level=true nrfld/nr_threshold=5e-6 \
  output2/dt=20 "output2/next_time=$next20" \
  output3/dt=20 "output3/next_time=$next20" \
  output4/dt=20 "output4/next_time=$next20" \
  output5/dt=20 "output5/next_time=$next20" \
  output6/dt=100 "output6/next_time=$next100" \
  output7/dt=200 "output7/next_time=$next200" \
  output8/dt=200 "output8/next_time=$next200" \
  time/tlim=71000 time/nlim=10000000 > "$run/run.log" 2>&1
raw=$?
set -e
printf 'job=%s\nraw_srun_exit=%s\n' "$SLURM_JOB_ID" "$raw" > "$run/result.txt"
final=$(find "$run/out" -maxdepth 1 -name '*.final.rst' -print -quit)
if [[ -n "$final" ]] &&
   /usr/bin/python3 "$validator" "$final" 70000 > "$run/restart_metadata.json" &&
   /usr/bin/python3 -c 'import json,sys; a=json.load(open(sys.argv[1]))["time"]; b=json.load(open(sys.argv[2]))["time"]; assert a>b' "$run/restart_metadata.json" "$run/seed_metadata.json" &&
   grep -q '^Terminating on ' "$run/run.log" &&
   ! grep -q 'FATAL ERROR' "$run/run.log"; then
  nr_total=$(grep -c '^\[NR\] status=' "$run/run.log")
  nr_good=$(grep -Ec '^\[NR\] status=(converged|dt_zero_initialization) ' "$run/run.log")
  test "$nr_total" -gt 0
  test "$nr_total" -eq "$nr_good"
  grep -q '^\[NR\] status=converged ' "$run/run.log"
  printf 'validated=yes\n' >> "$run/result.txt"
  /usr/bin/python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["time"]; print("final_time="+str(t)); print("target_reached="+("yes" if t>=71000-1e-6 else "no"))' \
    "$run/restart_metadata.json" >> "$run/result.txt"
else
  printf 'validated=no\n' >> "$run/result.txt"
  exit 1
fi
