#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor92_build_hydro_mass_ledger
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/debug/build_hydro_mass_ledger-%j.out
set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
source_dir="$base/source_newton_converged_trial"
target_dir="$base/source_hydro_mass_ledger_trial"
patch_file="$base/scripts/hydro_mass_flux_ledger.patch"
if [[ $(sha256sum "$source_dir/src/hydro/add_flux_divergence.cpp" | cut -d' ' -f1) != 2f480c182db1d3ba5c7c7a54e6067d3254bbd3d4268db9432bac7f8ad3700465 ]]; then
  echo FAIL_SOURCE_SHA
  exit 1
fi
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
if [[ ! -d "$target_dir" ]]; then
  cp -a "$source_dir" "$target_dir"
  cd "$target_dir"
  patch -p1 < "$patch_file"
fi
cd "$target_dir"
make -j8
sha256sum "$patch_file" src/hydro/add_flux_divergence.cpp bin/athena
echo PASS_HYDRO_MASS_LEDGER_BUILD
