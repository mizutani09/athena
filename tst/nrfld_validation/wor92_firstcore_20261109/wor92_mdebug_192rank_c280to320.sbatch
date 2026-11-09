#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=192
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor92_mdebug_192rank_c280to320
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/debug/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/debug/job_${SLURM_JOB_ID}"
mkdir -p "$case_dir/output"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_COMMIT=%s\n' f9dc679330bf76dcef0addcbc83b02d5d9121107
  printf 'JOB_ID=%s\n' "$SLURM_JOB_ID"
  printf 'NTASKS=%s\n' "$SLURM_NTASKS"
  module list 2>&1
  sha256sum "$base/source_firstcore_fix_trial/bin/athena" "$base/bundle/wor92_core_symmetry_cycle_output_gravity_20261107.in" "$base/bundle/data/eos/protostellar_tomida2015_material.tab" "$base/bundle/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
  sha256sum "$base/debug/job_338101/output/cfl030_flux_precision16_local.final.rst"
  sha256sum "$base/source_restart_gravity_trial/src/main.cpp"
  sha256sum "$base/source_restart_gravity_trial/src/gravity/mg_gravity.cpp" "$base/source_restart_gravity_trial/src/mesh/mesh.cpp"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
cd "$base/bundle"
set +e
mpirun -np 192 "$base/source_firstcore_fix_trial/bin/athena" \
  -r "$base/debug/job_338101/output/cfl030_flux_precision16_local.final.rst" \
  -d "$case_dir/output" -t 00:13:30 \
  mesh/numlevel=10 time/cfl_number=0.3 time/nlim=320 nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
  output2/dcycle=5 output4/dcycle=5 \
  > "$case_dir/launcher.log" 2>&1
status=$?
set -e
final_rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
if grep -q 'FATAL ERROR' "$case_dir/launcher.log" || [[ ! -s "$final_rst" ]]; then
  status=1
elif [[ $(python3 "$base/scripts/read_athena_restart_metadata_py36.py" "$final_rst" --field cycle) != 320 ]]; then
  status=1
fi
{
  printf 'RAW_EXIT=%s\n' "$status"
  date '+END=%Y-%m-%d %H:%M:%S %Z'
  grep -c '^\[NR\] status=converged' "$case_dir/launcher.log" || true
  grep -c '^\[LINEAR_FGMRES\]' "$case_dir/launcher.log" || true
  ls -lh "$case_dir/output" | tail -12
} > "$case_dir/summary.txt"
exit "$status"
