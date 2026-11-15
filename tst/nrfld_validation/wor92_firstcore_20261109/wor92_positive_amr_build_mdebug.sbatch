#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor92_positive_amr_build
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107/source_wor92_mayer5_positive_amr_20261115/build-slurm-%j.out

set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor92_xd_20261107
cd "$base/source_wor92_mayer5_positive_amr_20261115"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
make -j 8 > positive_amr_build.log 2>&1
sha256sum bin/athena src/mesh/mesh_refinement.cpp > positive_amr_hashes.txt
