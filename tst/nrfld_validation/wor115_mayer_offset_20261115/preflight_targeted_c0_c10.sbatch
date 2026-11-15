#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=cna150
#SBATCH --job-name=wor115_targeted_c0_c10
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/slurm-%j.out

set -euo pipefail
case_root=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115
case_dir="$case_root/job_${SLURM_JOB_ID}"
binary="$case_root/source_postamr_sink_targeted/bin/athena"
input="$case_root/../wor92_xd_20261107/ideal_mayer5_postsync_c0_20261110/wor92_mayer5_reference_postreset_c0_20261110.in"
reader=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/scripts/read_athena_restart_metadata_py36.py
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
export ATHENA_WOR92_ZERO_B_POSITIVE_AMR=1
export ATHENA_WOR115_POST_AMR_BACKGROUND_SINK=1
printf '%s  %s\n' 9d786d7c376a574e5b0e3f9ff9cc7379f906bdb2a61124fedbb97bbc04ca7c47 "$binary" | sha256sum -c -
printf '%s  %s\n' 382ab6df9ca41769a96475c52548fa008f486e3a43b5693118b378d991fa5ebb "$input" | sha256sum -c -
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_BASE_COMMIT=ae03e85b5a131b6c5cf7ddb3fe7a6bcb1ef9f91a\n'
  printf 'CHANGE_FROM_VALIDATED=postAMR_targeted_reset_only\n'
  printf 'NTASKS=%s\n' "$SLURM_NTASKS"
  sha256sum "$binary" "$input"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
cd /work/mizutaniks/output/fld/collapse/wor92_xd_20261107/bundle
set +e
timeout 1500 mpirun -np "$SLURM_NTASKS" "$binary" -i "$input" -d "$case_dir/output" -t 00:24:00 \
  mesh/numlevel=12 time/cfl_number=0.3 time/tlim=10000.0 time/nlim=10 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
  output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
  output2/dcycle=10 output4/dcycle=10 > "$case_dir/launcher.log" 2>&1
code=$?
set -e
rst="$case_dir/output/cfl030_flux_precision16_local.final.rst"
hst="$case_dir/output/cfl030_flux_precision16_local.hst"
if (( code != 0 )) || [[ ! -s "$rst" ]] || [[ ! -s "$hst" ]]; then
  printf 'FAIL_EXECUTION exit=%d\n' "$code" > "$case_dir/status.txt"
  exit 1
fi
cycle=$(python3 "$reader" "$rst" --field cycle)
blocks=$(python3 "$reader" "$rst" --field meshblocks)
if [[ "$cycle" != 10 ]] || grep -q 'FATAL ERROR' "$case_dir/launcher.log"; then
  printf 'FAIL_CYCLE_OR_SOLVER cycle=%s exit=%d\n' "$cycle" "$code" > "$case_dir/status.txt"
  exit 1
fi
if ! awk 'NF==39 {n++; if ($30<=1e-18 || $35<16 || $37>1e-8) bad=1; if (n==1) mass0=$23; if (mass0>0 && ($23/mass0<0.99999999 || $23/mass0>1.00000001)) bad=1} END {exit !(n>=10 && !bad)}' "$hst"; then
  printf 'FAIL_MASS_PRESSURE_OR_JEANS cycle=%s\n' "$cycle" > "$case_dir/status.txt"
  exit 1
fi
sha256sum "$rst" > "$case_dir/final_restart.sha256"
printf 'PASS_C0_TO_C10 cycle=%s meshblocks=%s\n' "$cycle" "$blocks" > "$case_dir/status.txt"
date '+END=%Y-%m-%d %H:%M:%S %Z' >> "$case_dir/provenance.txt"
