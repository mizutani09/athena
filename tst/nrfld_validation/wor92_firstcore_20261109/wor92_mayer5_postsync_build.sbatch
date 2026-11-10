#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor92_mayer5_postsync_build
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/ideal_mayer5_postsync_c0_20261110/build-%j.log

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
case_dir="$base/ideal_mayer5_postsync_c0_20261110"
source_dir="$base/source_wor92_mayer5_halo_20261110"
target_dir="$base/source_wor92_mayer5_postsync_20261110"
test ! -e "$target_dir"
mkdir -p "$target_dir"
rsync -a --exclude=/obj --exclude=/bin --exclude=/build.log \
  "$source_dir/" "$target_dir/"
cp "$case_dir/main_postsync.cpp" \
  "$target_dir/src/main.cpp"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
cd "$target_dir"
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=adiabatic -b --grav=mg -nrmgfld --flux=hlld_fld --nscalars=1 \
  -mpi -hdf5 --nghost=2 --cxx=icpx --mpiccmd=mpiicpx \
  --hdf5_path=/work/opt/local_oneapi/hdf5-parallel/1.14.5/INTEL/2024.0 \
  > "$case_dir/configure.log"
make -j8 > "$case_dir/build.log" 2>&1
test -s bin/athena
{
  printf 'SOURCE_BASE_COMMIT=582acd5999c1a5681ab1bc1d1af292676642ed40\n'
  printf 'SOURCE_CHANGE=background_radiation_temperature_halo_roundoff_and_postreset_boundary_sync\n'
  sha256sum src/nr_multigrid/NRFLD.cpp src/pgen/protostellar_core_fld.cpp \
    src/newton_raphson/Newton_Raphson.cpp src/main.cpp bin/athena
  date '+END=%Y-%m-%d %H:%M:%S %Z'
} > "$case_dir/build_provenance.txt"
cat "$case_dir/build_provenance.txt"
