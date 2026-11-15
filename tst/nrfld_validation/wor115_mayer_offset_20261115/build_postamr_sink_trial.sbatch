#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor115_postamr_sink_build
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/build-%j.out

set -euo pipefail
root=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/source_postamr_sink_trial
cd "$root"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
printf '%s  %s\n' e02092440b53f6d2ed7b77d5994b7f15c04d4f1da3b0b66741ebeb2dbb71f3a0 src/main.cpp | sha256sum -c -
make -j 8 > postamr_sink_build.log 2>&1
sha256sum bin/athena src/main.cpp > postamr_sink_build_hashes.txt
