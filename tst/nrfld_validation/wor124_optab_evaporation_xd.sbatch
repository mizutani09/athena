#!/bin/bash -l
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor124_evap
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/optab_evaporation_gate_20261117/slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116/optab_evaporation_gate_20261117
source_dir=$base/source
bundle=$base/bundle
job=$base/job_${SLURM_JOB_ID}
mkdir -p "$job"

module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
export I_MPI_FABRICS=shm
export OMP_NUM_THREADS=1
export ATHENA_WOR92_INEXACT_TRUE_CAP=1e-6

cd "$source_dir"
python3 ./configure.py --prob=test_optab_eos --eos=general/eos_table \
  -nrmgfld -hdf5 -mpi > "$job/configure.log" 2>&1
make clean >> "$job/configure.log" 2>&1
make -j16 > "$job/build.log" 2>&1

{
  date -Is
  git rev-parse HEAD
  git status --short
  sha256sum bin/athena "$bundle/wor124_optab_evaporation_gate.in" \
    "$bundle/protostellar_tomida2015_material.tab" \
    "$bundle"/protostellar_hybrid_end{200,300,400}.h5
  module list 2>&1
  env | grep '^ATHENA_' | sort
} > "$job/provenance.txt"

printf 'width,ranks,exit,cycles,Tgas,Trad,allE,above_strict,fgmres_max,opacity_ok,elapsed_seconds\n' \
  > "$job/status.csv"
for width in 200 300 400; do
  out=$job/end${width}_2rank
  mkdir -p "$out"
  started=$(date +%s)
  set +e
  /usr/bin/time -v -o "$out/resources.txt" \
    mpirun -np 2 bin/athena -i "$bundle/wor124_optab_evaporation_gate.in" \
      -d "$out" \
      hydro/eos_file_name="$bundle/protostellar_tomida2015_material.tab" \
      fld/opacity_table_file="$bundle/protostellar_hybrid_end${width}.h5" \
      mesh/nx1=8 mesh/x1max=2.0 \
      > "$out/run.log" 2>&1
  code=$?
  set -e
  read -r final_time _ _ _ _ _ _ _ _ _ Tgas Trad _ _ _ _ allE \
    < <(tail -1 "$out/wor124_evap.hst")
  cycles=$(awk '/time=.*cycle=/{for(i=1;i<=NF;i++)if($i~/^cycle=/){split($i,a,"=");v=a[2]}}END{print v+0}' \
    "$out/run.log")
  above=$(awk '/^\[LINEAR_FGMRES\]/{for(i=1;i<=NF;i++)if($i~/^relative_residual=/){split($i,a,"=");if(a[2]+0>2e-8)n++}}END{print n+0}' \
    "$out/run.log")
  fgmax=$(awk '/^\[LINEAR_FGMRES\]/{for(i=1;i<=NF;i++)if($i~/^relative_residual=/){split($i,a,"=");if(a[2]+0>m)m=a[2]}}END{printf "%.17g",m}' \
    "$out/run.log")
  opacity_ok=0
  if grep -q '^FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0 negative_results=0 zero_results=0 nonfinite_results=0$' \
      "$out/run.log"; then opacity_ok=1; fi
  printf '%s,2,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$width" "$code" \
    "$cycles" "$Tgas" "$Trad" "$allE" "$above" "$fgmax" "$opacity_ok" \
    "$(($(date +%s)-started))" >> "$job/status.csv"
  if [[ $code != 0 || $cycles != 71 || $opacity_ok != 1 ]] \
      || grep -q 'FATAL ERROR' "$out/run.log"; then
    echo "FAIL end${width}" > "$job/status.txt"
    exit 1
  fi
done
echo PASS > "$job/status.txt"
date -Is > "$job/completed.txt"
