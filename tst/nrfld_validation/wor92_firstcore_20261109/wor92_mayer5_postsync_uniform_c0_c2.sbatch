#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor92_mayer5_postsync_uniform_c0_c2
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_postsync_uniform_diag_20261110/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/ideal_mayer5_postsync_uniform_diag_20261110/job_${SLURM_JOB_ID}"
binary="$base/source_wor92_mayer5_postsync_20261110/bin/athena"
input="$base/ideal_mayer5_postsync_uniform_diag_20261110/wor92_mayer5_reference_postreset_c0_20261110.in"
reader="$base/scripts/read_athena_restart_metadata_py36.py"
expected_binary_sha=9e7c2faa80832a7cd5d8aa3fd4689aed1648a7f346d4d492c2c0742217020b7f
expected_input_sha=382ab6df9ca41769a96475c52548fa008f486e3a43b5693118b378d991fa5ebb
mkdir -p "$case_dir/output"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm:ofi
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
printf '%s  %s\n' "$expected_binary_sha" "$binary" | sha256sum -c -
printf '%s  %s\n' "$expected_input_sha" "$input" | sha256sum -c -
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_BASE_COMMIT=582acd5999c1a5681ab1bc1d1af292676642ed40\n'
  printf 'SOURCE_PGEN_SHA256=4787a31fc79d6c570f287d78896aad0ff7b9f5d59db0224e707b56cfcbb609b0\n'
  printf 'START=INITIAL_CONDITION_CYCLE_0\n'
  printf 'NTASKS=%s\n' "$SLURM_NTASKS"
  module list 2>&1
  sha256sum "$binary" "$input"
  cat "$base/ideal_mayer5_postsync_uniform_diag_20261110/build_provenance.txt"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
cd "$base/bundle"
set +e
mpirun -np "$SLURM_NTASKS" "$binary" -i "$input" -d "$case_dir/output" -t 00:26:00 \
  mesh/numlevel=1 time/cfl_number=0.3 time/tlim=10000.0 time/nlim=2 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
  output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
  output2/dcycle=10 output4/dcycle=10 > "$case_dir/launcher.log" 2>&1
code=$?
set -e
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
if (( code != 0 )) || [[ ! -s "$rst" ]]; then
  printf 'FAIL_LAUNCH_OR_RESTART exit=%d\n' "$code" > "$case_dir/status.txt"
  exit 1
fi
cycle=$(python3 "$reader" "$rst" --field cycle)
blocks=$(python3 "$reader" "$rst" --field meshblocks)
if [[ "$cycle" != 2 ]] || grep -q 'FATAL ERROR' "$case_dir/launcher.log"; then
  printf 'FAIL_CYCLE_OR_SOLVER cycle=%s exit=%d\n' "$cycle" "$code" > "$case_dir/status.txt"
  exit 1
fi
hst="$case_dir/output/cfl030_flux_precision16_local.hst"
if ! awk 'NF==39 {n++; if ($30<=1e-18) bad=1; if (n==1) initial=$23; if (initial<=0 || ($23-initial>0? $23-initial : initial-$23)/initial>1e-8) bad=1} END {exit !(n>0 && !bad)}' "$hst"; then
  printf 'FAIL_STEP_PRESSURE_FLOOR_JEANS_OR_MASS cycle=%s\n' "$cycle" > "$case_dir/status.txt"
  exit 1
fi
sha256sum "$rst" > "$case_dir/final_restart.sha256"
printf 'PASS_C0_TO_C2 cycle=%s meshblocks=%s\n' "$cycle" "$blocks" > "$case_dir/status.txt"
date '+END=%Y-%m-%d %H:%M:%S %Z' >> "$case_dir/provenance.txt"
