#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor124_crop375
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/cutout_trial_20261117/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
trial=$base/cutout_trial_20261117
source_dir=$base/source_floor_trial
case_name=${WOR124_CUTOUT_CASE:-crop375}
if [[ "$case_name" == parent ]]; then
  checkpoint=$base/high_amr_long/combined_B32uG_floor_amr11_3node_job_341015/output/combined_B32uG_floor_amr11_3node.00182.rst
  numlevel=11
elif [[ "$case_name" == crop375 ]]; then
  checkpoint=$trial/c1780_crop375.rst
  numlevel=8
elif [[ "$case_name" == crop750 ]]; then
  checkpoint=$trial/c1780_crop750.rst
  numlevel=9
else
  printf 'Unknown WOR124_CUTOUT_CASE=%s\n' "$case_name" >&2
  exit 2
fi
job=$trial/job_${SLURM_JOB_ID}
mkdir -p "$job/output"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm
export OMP_NUM_THREADS=1
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
{
  date -Is
  printf 'checkpoint='; sha256sum "$checkpoint"
  printf 'binary='; sha256sum bin/athena
  printf 'source_commit='; cat "$base/source_commit.txt"
  printf 'eos='; sha256sum data/eos/protostellar_tomida2015_material.tab
  printf 'opacity='; sha256sum data/opacity/protostellar_semenov2003_nrm_c_a.h5
  env | grep '^ATHENA_' | sort
  module list 2>&1
} > "$job/provenance.txt"
started=$(date +%s)
set +e
/usr/bin/time -v -o "$job/resources.txt" \
  mpirun -np "$SLURM_NTASKS" bin/athena \
  -r "$checkpoint" -d "$job/output" -t 00:27:00 \
  job/problem_id="$case_name" \
  time/nlim="${WOR124_CUTOUT_NLIM:-1785}" time/tlim=10000 time/ncycle_out=1 \
  mesh/numlevel="$numlevel" \
  output2/dcycle="${WOR124_CUTOUT_OUTPUT_CYCLES:-5}" output2/variable=cons,prim,uov \
  output4/dcycle="${WOR124_CUTOUT_OUTPUT_CYCLES:-5}" \
  nrfld/diagnostic_verbosity=0 \
  nrfld/jacobian_check=false nrfld/jacobian_check_each_cycle=false \
  nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart=160 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_mg_vcycles=3 \
  > "$job/run.log" 2>&1
code=$?
set -e
elapsed=$(($(date +%s)-started))
printf 'exit=%d elapsed_seconds=%d\n' "$code" "$elapsed" > "$job/status.txt"
if grep -q 'FATAL ERROR' "$job/run.log"; then
  printf 'fatal_error=1\n' >> "$job/status.txt"
  exit 1
fi
exit "$code"
