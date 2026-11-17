#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor124_optab_gate
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/secondcore_amr_gate_20261117/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
trial=$base/secondcore_amr_gate_20261117
source_dir=$base/source_floor_trial
checkpoint=$base/high_amr_long/combined_B32uG_floor_amr11_3node_job_341015/output/combined_B32uG_floor_amr11_3node.final.rst
reader=$base/scripts/read_restart.py
opacity=${WOR124_GATE_OPACITY:?set WOR124_GATE_OPACITY to the hybrid HDF5 table}
expected_sha=${WOR124_GATE_OPACITY_SHA256:?set WOR124_GATE_OPACITY_SHA256}
case_name=full_n12_hybrid_optab
job=$trial/job_${SLURM_JOB_ID}_${case_name}
mkdir -p "$job/output"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm OMP_NUM_THREADS=1
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-6
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
cd "$source_dir"
if [[ $(python3 "$reader" "$checkpoint" --field cycle) != 2250 ]]; then
  echo 'FAIL_INVALID_CHECKPOINT' > "$job/status.txt"
  exit 1
fi
actual_sha=$(sha256sum "$opacity" | awk '{print $1}')
if [[ "$actual_sha" != "$expected_sha" ]]; then
  printf 'FAIL_OPACITY_SHA expected=%s actual=%s\n' "$expected_sha" "$actual_sha" \
    > "$job/status.txt"
  exit 1
fi
{
  date -Is
  printf 'case=%s numlevel=12 target_cycle=2255 ranks=%s\n' \
    "$case_name" "$SLURM_NTASKS"
  printf 'source_commit='; cat "$base/source_commit.txt"
  sha256sum "$checkpoint" bin/athena \
    data/eos/protostellar_tomida2015_material.tab "$opacity"
  env | grep '^ATHENA_' | sort
  module list 2>&1
} > "$job/provenance.txt"
started=$(date +%s)
set +e
/usr/bin/time -v -o "$job/resources.txt" \
  mpirun -np "$SLURM_NTASKS" bin/athena \
  -r "$checkpoint" -d "$job/output" -t 00:27:00 \
  job/problem_id="$case_name" mesh/numlevel=12 \
  fld/opacity_table_file="$opacity" fld/opacity_table_file_type=hdf5 \
  fld/opacity_table_axis=trho fld/opacity_table_domain_policy=clamp \
  fld/opacity_table_planck_format=log10 \
  fld/opacity_table_rosseland_format=log10 \
  time/cfl_number=0.3 time/nlim=2255 time/tlim=10000 time/ncycle_out=1 \
  output2/dcycle=5 output2/variable=cons,prim,uov \
  output4/dcycle=1 nrfld/diagnostic_verbosity=0 \
  nrfld/jacobian_check=false nrfld/jacobian_check_each_cycle=false \
  nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart=160 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_mg_vcycles=3 \
  > "$job/run.log" 2>&1
code=$?
set -e
elapsed=$(($(date +%s)-started))
printf 'exit=%d elapsed_seconds=%d\n' "$code" "$elapsed" > "$job/status.txt"
if [[ -s "$job/output/$case_name.final.rst" ]]; then
  cycle=$(python3 "$reader" "$job/output/$case_name.final.rst" --field cycle)
  blocks=$(python3 "$reader" "$job/output/$case_name.final.rst" --field meshblocks)
  printf 'final_cycle=%s final_blocks=%s\n' "$cycle" "$blocks" \
    >> "$job/status.txt"
else
  echo 'final_restart_missing=1' >> "$job/status.txt"
  exit 1
fi
if [[ "$code" != 0 || "$cycle" != 2255 ]] \
    || grep -q 'FATAL ERROR' "$job/run.log"; then
  echo 'FAIL_RUN' >> "$job/status.txt"
  exit 1
fi
if ! grep -q '^FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0 negative_results=0 zero_results=0 nonfinite_results=0$' "$job/run.log"; then
  echo 'FAIL_OPACITY' >> "$job/status.txt"
  exit 1
fi
read -r jeans rho tgas divb < <(awk 'END {print $35,$16,$20,$24}' \
  "$job/output/$case_name.hst")
printf 'jeans_min=%s rho_max=%s Tgas_max=%s divB_max=%s\n' \
  "$jeans" "$rho" "$tgas" "$divb" >> "$job/status.txt"
if ! python3 - "$jeans" "$rho" "$tgas" "$divb" <<'PY'
import math, sys
jeans, rho, tgas, divb = map(float, sys.argv[1:])
sys.exit(0 if all(map(math.isfinite, (jeans, rho, tgas, divb)))
         and jeans >= 16 and rho > 0 and tgas > 0 else 1)
PY
then
  echo 'FAIL_PHYSICAL_GATE' >> "$job/status.txt"
  exit 1
fi
echo 'PASS' >> "$job/status.txt"
