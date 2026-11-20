#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=paper_B32_v2_gate
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
source_dir="$base/source_protostellar_firstcore_v2_577f273bd"
origin="$base/high_amr_long/combined_B32uG_floor_amr11_3node_job_341015/output/combined_B32uG_floor_amr11_3node.00182.rst"
reader="$base/scripts/read_restart.py"
audit="$base/scripts/collapse_opacity_domain_audit.py"
table_dir="$base/source_full_solve_early_retry_20261119"
eos_table="$table_dir/data/eos/protostellar_tomida2015_material.tab"
opacity_table="$table_dir/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
label=paper_B32_amr12_v2_gate
job="$base/paper_amr12_v2/gate_job_${SLURM_JOB_ID}"
mkdir -p "$job/output"

if [[ ! -s "$source_dir/FROZEN_SOURCE_COMMIT" ]] ||
   [[ $(<"$source_dir/FROZEN_SOURCE_COMMIT") != 577f273bddee4128d873cd7da7a4b1495b97ba8c ]]; then
  printf 'FAIL_SOURCE_TAG path=%s\n' "$source_dir" > "$job/status.txt"
  exit 1
fi
if [[ ! -s "$origin" ]] || [[ $(python3 "$reader" "$origin" --field cycle) != 1780 ]]; then
  printf 'FAIL_INVALID_CHECKPOINT path=%s\n' "$origin" > "$job/status.txt"
  exit 1
fi
if [[ ! -s "$eos_table" ]] || [[ ! -s "$opacity_table" ]]; then
  printf 'FAIL_MISSING_TABLE eos=%s opacity=%s\n' "$eos_table" "$opacity_table" \
    > "$job/status.txt"
  exit 1
fi

module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm:ofi OMP_NUM_THREADS=1
export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
export ATHENA_NRFLD_TANGENTIAL_FULL_SOLVE_RETRY=1
export ATHENA_NRFLD_EXPERIMENTAL_VARIABLE_LIMITER=1
export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7
export ATHENA_WOR92_INEXACT_STAGNATION_RATIO=0.9
export ATHENA_WOR92_CENTERED_FACE_TRIAL=1
export ATHENA_WOR92_POSITIVE_NR_GHOST=2
unset ATHENA_NRFLD_FGMRES_REFINEMENT_STEPS
unset ATHENA_NRFLD_FGMRES_REFINEMENT_ETA_TRIGGER
unset ATHENA_NRFLD_TANGENTIAL_POSITIVITY_FALLBACK

mkdir -p "$source_dir/data/eos" "$source_dir/data/opacity"
ln -sfn "$eos_table" "$source_dir/data/eos/protostellar_tomida2015_material.tab"
ln -sfn "$opacity_table" "$source_dir/data/opacity/protostellar_semenov2003_nrm_c_a.h5"
cd "$source_dir"
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table -b --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -hdf5 -mpi --ccmd=icpx --mpiccmd=mpiicpx --cflag=-O3 \
  > "$job/configure.log" 2>&1
make -j8 > "$job/build.log" 2>&1

{
  date -Is
  printf 'source_commit=577f273bddee4128d873cd7da7a4b1495b97ba8c source_tree=8c4a94cb1b53b0ce6a7b574371351007a98472cb tag=protostellar-firstcore-v2\n'
  printf 'ranks=%s start_cycle=1780 target=1790 cfl=0.3 numlevel=12 fgmres=40/300_mg2 inexact_cap=1e-7\n' "$SLURM_NTASKS"
  sha256sum "$origin" "$reader" "$eos_table" "$opacity_table" bin/athena \
    src/newton_raphson/Newton_Raphson_driver.cpp
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"

started=$(date +%s)
set +e
/usr/bin/time -v -o "$job/resources.txt" \
  mpirun -np "$SLURM_NTASKS" bin/athena \
    -r "$origin" -d "$job/output" \
    job/problem_id="$label" mesh/numlevel=12 \
    time/cfl_number=0.3 time/nlim=1790 time/ncycle_out=1 time/xorder=2 \
    output2/dcycle=10 output2/variable=cons,prim,uov output4/dcycle=10 \
    nrfld/diagnostic_verbosity=1 \
    nrfld/jacobian_check=false nrfld/jacobian_check_each_cycle=false \
    nrfld/fgmres_threshold=2e-8 nrfld/fgmres_restart=40 \
    nrfld/fgmres_max_iterations=300 nrfld/fgmres_mg_vcycles=2 \
    > "$job/run.log" 2>&1
code=$?
set -e
elapsed=$(($(date +%s)-started))
limit=0 fatal=0 retry=0
if grep -q 'Terminating on cycle limit' "$job/run.log"; then limit=1; fi
if grep -q 'FATAL ERROR' "$job/run.log"; then fatal=1; fi
if grep -q '^\[NR_TANGENTIAL_FULL_RETRY\].*retry_status=converged' "$job/run.log"; then retry=1; fi
printf 'case,exit_code,cycle_limit,has_fatal,retry_converged,elapsed_seconds\n' > "$job/status.csv"
printf '%s,%s,%s,%s,%s,%s\n' "$label" "$code" "$limit" "$fatal" "$retry" \
  "$elapsed" >> "$job/status.csv"
if (( code != 0 || limit != 1 || fatal != 0 )); then exit 1; fi

rst="$job/output/$label.final.rst"
if [[ ! -s "$rst" ]] || [[ $(python3 "$reader" "$rst" --field cycle) != 1790 ]]; then
  printf 'FAIL_FINAL_RESTART path=%s\n' "$rst" > "$job/postcheck.txt"
  exit 1
fi
latest_hdf=$(find "$job/output" -maxdepth 1 -type f -name '*.prim.*.athdf' \
  -printf '%p\n' | sort | tail -n 1)
python3 "$audit" "$latest_hdf" "$opacity_table" --backend h5dump \
  > "$job/opacity_domain_audit.json"
sha256sum "$rst" > "$job/final_restart.sha256"
date -Is > "$job/completed.txt"
