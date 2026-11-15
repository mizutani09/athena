#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=120
#SBATCH --ntasks-per-node=120
#SBATCH --mem=110G
#SBATCH --time=00:25:00
#SBATCH --job-name=wor115_postamr_c1421
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/slurm-%j.out

set -euo pipefail
case_root=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115
case_dir="$case_root/job_${SLURM_JOB_ID}"
binary=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/source_postamr_sink_trial/bin/athena
binary_sha=603345073d820c254d15c7b3c66948137975cda00cdf0ef1815f31eaeec197f2
variant=${TRIAL_VARIANT:-whole_mesh}
if [[ "$variant" == targeted ]]; then
  binary="$case_root/source_postamr_sink_targeted/bin/athena"
  binary_sha=9d786d7c376a574e5b0e3f9ff9cc7379f906bdb2a61124fedbb97bbc04ca7c47
elif [[ "$variant" != whole_mesh ]]; then
  echo "unsupported TRIAL_VARIANT=$variant" >&2
  exit 2
fi
seed=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_c1415_positive_amr_20261115/job_340331/output/cfl030_flux_precision16_local.final.rst
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
printf '%s  %s\n' "$binary_sha" "$binary" | sha256sum -c -
printf '%s  %s\n' d02ae7ad40028c2f1a359a93ab37168a040f78ecb5ac43c1f831346c8eebc64d "$seed" | sha256sum -c -
if [[ $(python3 "$reader" "$seed" --field cycle) != 1420 ]]; then
  printf 'FAIL_SEED_CYCLE\n' > "$case_dir/status.txt"
  exit 1
fi
{
  date '+START=%Y-%m-%d %H:%M:%S %Z'
  printf 'CONTROL=VALIDATED_C1420_RESTART\nCHANGE=mesh/numlevel_12_to_13_AND_problem/N_J_32_to_48_AND_postAMR_sink_%s\n' "$variant"
  sha256sum "$binary" "$seed"
  env | grep '^ATHENA_' | sort
} > "$case_dir/provenance.txt"
cd /work/mizutaniks/output/fld/collapse/wor92_xd_20261107/bundle
set +e
timeout 900 mpirun -np "$SLURM_NTASKS" "$binary" -r "$seed" -d "$case_dir/output" -t 00:14:00 \
  mesh/numlevel=13 problem/N_J=48 time/cfl_number=0.3 time/tlim=10000.0 time/nlim=1421 \
  nrfld/fgmres_max_iterations=600 nrfld/fgmres_restart=80 nrfld/fgmres_mg_vcycles=3 \
  output1/dcycle=1 output1/data_format=%22.15e output2/variable=prim \
  output2/dcycle=1 output4/dcycle=1 > "$case_dir/launcher.log" 2>&1
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
if [[ "$cycle" != 1421 ]] || grep -q 'FATAL ERROR' "$case_dir/launcher.log"; then
  printf 'FAIL_CYCLE_OR_SOLVER cycle=%s exit=%d\n' "$cycle" "$code" > "$case_dir/status.txt"
  exit 1
fi
if ! awk 'NF==39 {n++; if ($30<=1e-18 || $35<16 || $23/2.936555728050246e33<0.99999999 || $23/2.936555728050246e33>1.00000001) bad=1} END {exit !(n>=1 && !bad)}' "$hst"; then
  printf 'FAIL_MASS_PRESSURE_OR_JEANS cycle=%s\n' "$cycle" > "$case_dir/status.txt"
  exit 1
fi
sha256sum "$rst" > "$case_dir/final_restart.sha256"
printf 'PASS_POSTAMR_HIGHRES_C1421 cycle=%s meshblocks=%s\n' "$cycle" "$blocks" > "$case_dir/status.txt"
date '+END=%Y-%m-%d %H:%M:%S %Z' >> "$case_dir/provenance.txt"
