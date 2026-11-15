#!/bin/bash -l
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor115_postamr_targeted_build
#SBATCH --output=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/build-%j.out

set -euo pipefail
root=/work/mizutaniks/output/fld/collapse/wor115_mayer_offset_20261115/source_postamr_sink_targeted
cd "$root"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
printf '%s  %s\n' 41350fcbcd4deb3cf7c61e7820156d45fee3becbc28f44c2ce7eec07461f98fa src/main.cpp | sha256sum -c -
printf '%s  %s\n' f9cd54f987bfb6f7cf0b297be492b74e687015f1755a6a75eb3f82abb2ba15a3 src/mesh/mesh.hpp | sha256sum -c -
printf '%s  %s\n' acefda23719633f84c2cd22ff33688c4e4cc18005dacff190da5486c7506490c src/mesh/meshblock.cpp | sha256sum -c -
printf '%s  %s\n' 3812ee522a63251ce4ed498f51241dbb4dc3f00a0fbb58c711b4578f8ce6fc15 src/mesh/amr_loadbalance.cpp | sha256sum -c -
printf '%s  %s\n' 754ea5cdd1051f64a53b4294e12597bc9067ee1b4bec5c863e5b98fd4d8db93c src/pgen/default_pgen.cpp | sha256sum -c -
printf '%s  %s\n' 7a0cbc8a2c5b74a7def7830788cf2e0db6e3cd4944b6a5bb6875b9f8f9b50272 src/pgen/protostellar_core_fld.cpp | sha256sum -c -
make -j 8 > postamr_sink_build.log 2>&1
sha256sum bin/athena src/main.cpp > postamr_sink_build_hashes.txt
