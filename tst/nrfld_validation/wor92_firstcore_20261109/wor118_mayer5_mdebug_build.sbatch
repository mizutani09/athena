#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor118_mayer5_build
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_20261110/build-%j.log

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/ideal_mayer5_20261110"
source_dir="$base/source_wor118_production_trial"
target_dir="$base/source_wor118_mayer5_match_20261110"
mkdir -p "$case_dir"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
if [[ ! -e "$target_dir" ]]; then
  rsync -a --exclude=/obj --exclude=/bin --exclude=/build_adaptive.log \
    --exclude=/configure.log "$source_dir/" "$target_dir/"
else
  # A failed compiler-wrapper preflight left only generated objects here.
  test ! -s "$target_dir/bin/athena"
  diff -qr --exclude=defs.hpp "$source_dir/src" "$target_dir/src" > "$case_dir/source_diff.txt"
  test ! -s "$case_dir/source_diff.txt"
  rm -rf -- "$target_dir/obj" "$target_dir/bin"
fi
cd "$target_dir"
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=adiabatic -b --grav=mg -nrmgfld --flux=hlld_fld --nscalars=1 \
  -mpi -hdf5 --nghost=2 --cxx=icpx --mpiccmd=mpiicpx \
  --hdf5_path=/work/opt/local_oneapi/hdf5-parallel/1.14.5/INTEL/2024.0 \
  > "$case_dir/configure.log"
make -j8 > "$case_dir/build.log" 2>&1
test -s bin/athena
grep -q 'Equation of state:.*adiabatic' "$case_dir/configure.log"
{
  printf 'SOURCE_COMMIT=ff60f8f07842acb00e7753932bcfe30867e3a66e\n'
  printf 'SOURCE_PARENT=%s\n' "$source_dir"
  printf 'SOURCE_TARGET=%s\n' "$target_dir"
  printf 'BINARY_SHA256=%s\n' "$(sha256sum bin/athena | cut -d' ' -f1)"
  printf 'SOURCE_SRC_SHA256=%s\n' "$(find src -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"
  date '+END=%Y-%m-%d %H:%M:%S %Z'
} > "$case_dir/build_provenance.txt"
cat "$case_dir/build_provenance.txt"
