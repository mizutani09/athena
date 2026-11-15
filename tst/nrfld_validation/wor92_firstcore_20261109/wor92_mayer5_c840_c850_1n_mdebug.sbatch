#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=120
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor92_mayer5_c840_c850_1n
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_c840_c850_1n_20261115/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/ideal_mayer5_c840_c850_1n_20261115/job_${SLURM_JOB_ID}"
binary="$base/source_wor92_mayer5_postsync_20261110/bin/athena"
seed="$base/ideal_mayer5_c830_c840_1n_20261115/job_340115/output/cfl030_flux_precision16_local.final.rst"
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
printf '%s  %s\n' 9e7c2faa80832a7cd5d8aa3fd4689aed1648a7f346d4d492c2c0742217020b7f "$binary" | sha256sum -c -
printf '%s  %s\n' f6d62a65e9cc7c69d237b4ca027339226590199e2b8eb91664141ce04e6909e8 "$seed" | sha256sum -c -
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_JOB=340115\nSEED_CYCLE=840\nNTASKS=%s\n' "$SLURM_NTASKS"
  sha256sum "$binary" "$seed"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
cd "$base/bundle"
set +e
timeout --signal=TERM --kill-after=20s 600s mpirun -np "$SLURM_NTASKS" \
  "$binary" -r "$seed" -d "$case_dir/output" -t 00:10:00 \
  mesh/numlevel=12 time/cfl_number=0.3 time/tlim=10000.0 time/nlim=850 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
  output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
  output2/dcycle=10 output4/dcycle=1 > "$case_dir/launcher.log" 2>&1
code=$?
set -e
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
cycle=missing
if [[ -s "$rst" ]]; then cycle=$(python3 "$reader" "$rst" --field cycle); fi
printf 'EXIT=%d CYCLE=%s\n' "$code" "$cycle" > "$case_dir/status.txt"
date '+END=%Y-%m-%d %H:%M:%S %Z' >> "$case_dir/provenance.txt"
if (( code != 0 )) || [[ "$cycle" != 850 ]]; then exit 1; fi
