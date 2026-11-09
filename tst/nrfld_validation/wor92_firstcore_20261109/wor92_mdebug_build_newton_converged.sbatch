#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor92_build_newton_converged
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/debug/build_newton_converged-%j.out
set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
cd "$base/source_newton_converged_trial"
make -j8
sha256sum bin/athena src/newton_raphson/Newton_Raphson_driver.cpp
