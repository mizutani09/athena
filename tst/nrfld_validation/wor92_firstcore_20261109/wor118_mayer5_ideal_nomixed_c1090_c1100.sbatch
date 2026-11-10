#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor118_ideal_nomixed_replay
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_20261110/nomixed-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/ideal_mayer5_20261110/nomixed_job_${SLURM_JOB_ID}"
start="$base/ideal_mayer5_20261110/job_338718/output/cfl030_flux_precision16_local.final.rst"
binary="$base/source_wor118_ideal_localgas_20261110/bin/athena"
reader="$base/scripts/read_athena_restart_metadata_py36.py"
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
test "$(python3 "$reader" "$start" --field cycle)" = 1090
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_COMMIT=582acd5999c1a5681ab1bc1d1af292676642ed40\n'
  printf 'START_CYCLE=1090 TARGET_CYCLE=1100 NTASKS=%s\n' "$SLURM_NTASKS"
  sha256sum "$start" "$binary"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
cd "$base/bundle"
set +e
mpirun -np "$SLURM_NTASKS" "$binary" -r "$start" -d "$case_dir/output" -t 00:12:00 \
  fld/mixed_frame_transport=false mesh/numlevel=12 time/cfl_number=0.3 time/tlim=10000.0 time/nlim=1100 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
  output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
  output2/dcycle=5 output4/dcycle=5 > "$case_dir/launcher.log" 2>&1
code=$?
set -e
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
if [[ -s "$rst" ]]; then
  cycle=$(python3 "$reader" "$rst" --field cycle)
else
  cycle=MISSING
fi
printf 'EXIT=%d END_CYCLE=%s\n' "$code" "$cycle" > "$case_dir/status.txt"
if (( code != 0 )) || [[ "$cycle" != 1100 ]]; then exit 1; fi
sha256sum "$rst" > "$case_dir/final_restart.sha256"
printf 'PASS_C1090_TO_C1100_NO_MIXED_TRANSPORT\n' >> "$case_dir/status.txt"
date '+END=%Y-%m-%d %H:%M:%S %Z' >> "$case_dir/provenance.txt"
