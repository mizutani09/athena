#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=120
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --job-name=wor115_c1500_old_ab
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/slurm-%j.out

set -euo pipefail
root=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115
case_dir="$root/job_${SLURM_JOB_ID}"
seed="$root/job_340413/output/cfl030_flux_precision16_local.final.rst"
binary=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/source_wor92_mayer5_positive_amr_20261115/bin/athena
binary_sha=b64491aa31da79216c99dbfcc6eb2b9cba805e4c385a4a67d092c303a622b5cb
variant=${TRIAL_VARIANT:-old}
if [[ "$variant" == new_off ]]; then
  binary="$root/source_postamr_sink_trial/bin/athena"
  binary_sha=603345073d820c254d15c7b3c66948137975cda00cdf0ef1815f31eaeec197f2
elif [[ "$variant" == targeted_on ]]; then
  binary="$root/source_postamr_sink_targeted/bin/athena"
  binary_sha=9d786d7c376a574e5b0e3f9ff9cc7379f906bdb2a61124fedbb97bbc04ca7c47
elif [[ "$variant" != old ]]; then
  echo "unsupported TRIAL_VARIANT=$variant" >&2
  exit 2
fi
input="$root/../wor92_xd_20261107/ideal_mayer5_postsync_c0_20261110/wor92_mayer5_reference_postreset_c0_20261110.in"
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
if [[ "$variant" == targeted_on ]]; then
  export ATHENA_WOR115_POST_AMR_BACKGROUND_SINK=1
else
  unset ATHENA_WOR115_POST_AMR_BACKGROUND_SINK
fi
printf '%s  %s\n' "$binary_sha" "$binary" | sha256sum -c -
printf '%s  %s\n' 382ab6df9ca41769a96475c52548fa008f486e3a43b5693118b378d991fa5ebb "$input" | sha256sum -c -
cycle=$(python3 "$reader" "$seed" --field cycle)
[[ "$cycle" == 1500 ]]
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'SOURCE_JOB_ID=340413\nSOURCE_CYCLE=%s\nCONTROLLED_CHANGE=BINARY_VARIANT_%s\n' "$cycle" "$variant"
  sha256sum "$binary" "$input" "$seed"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
cd /work/mizutaniks/output/fld/collapse/wor92_xd_20261107/bundle
set +e
timeout 480 mpirun -np "$SLURM_NTASKS" "$binary" -r "$seed" -d "$case_dir/output" -t 00:07:00 \
  mesh/numlevel=12 problem/N_J=32 time/cfl_number=0.3 time/tlim=10000.0 time/nlim=1506 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
  output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
  output2/dcycle=10 output4/dcycle=10 > "$case_dir/launcher.log" 2>&1
code=$?
set -e
after=unknown
if [[ -s "$case_dir/output/cfl030_flux_precision16_local.final.rst" ]]; then
  after=$(python3 "$reader" "$case_dir/output/cfl030_flux_precision16_local.final.rst" --field cycle)
fi
printf 'EXIT=%s SOURCE_CYCLE=%s FINAL_CYCLE=%s\n' "$code" "$cycle" "$after" > "$case_dir/status.txt"
date '+END=%Y-%m-%d %H:%M:%S %Z' >> "$case_dir/provenance.txt"
(( code == 0 )) && [[ "$after" == 1506 ]]
