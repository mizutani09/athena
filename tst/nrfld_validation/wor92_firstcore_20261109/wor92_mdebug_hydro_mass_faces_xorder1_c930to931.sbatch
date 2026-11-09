#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --ntasks=64
#SBATCH --mem=110G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor92_hydro_mass_faces_xorder1
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/long/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/debug/job_${SLURM_JOB_ID}"
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
export ATHENA_WOR92_HYDRO_MASS_LEDGER=1
reader="$base/scripts/read_athena_restart_metadata_py36.py"
start="$base/long/job_338156/output/cfl030_flux_precision16_local.final.rst"
binary="$base/source_hydro_mass_faces_v2_trial/bin/athena"
if [[ $(python3 "$reader" "$start" --field cycle) != 930 ]]; then
  printf 'FAIL_INVALID_START_RESTART\n' > "$case_dir/status.txt"
  exit 1
fi
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  sha256sum "$start" "$binary" "$base/bundle/wor92_core_symmetry_cycle_output_gravity_20261107.in" \
    "$base/bundle/data/eos/protostellar_tomida2015_material.tab" \
    "$base/bundle/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
cd "$base/bundle"
mpirun -np "$SLURM_NTASKS" -outfile-pattern "$case_dir/rank-%r.log" \
  "$binary" -r "$start" -d "$case_dir/output" -t 00:12:00 \
  mesh/numlevel=12 time/cfl_number=0.3 time/nlim=931 time/xorder=1 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
  output1/dcycle=1 output1/data_format=%22.15e \
  output2/variable=cons output2/dcycle=1 output4/dcycle=1 > "$case_dir/launcher.log" 2>&1
python3 "$reader" "$case_dir/output/cfl030_flux_precision16_local.final.rst" \
  --field cycle > "$case_dir/final_cycle.txt"
if [[ $(cat "$case_dir/final_cycle.txt") != 931 ]]; then
  printf 'FAIL_WRONG_END_CYCLE\n' > "$case_dir/status.txt"
  exit 1
fi
printf 'PASS_C930_TO_931_HYDRO_MASS_FACE_XORDER1\n' > "$case_dir/status.txt"
date '+END=%Y-%m-%d %H:%M:%S %Z' >> "$case_dir/provenance.txt"
