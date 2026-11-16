#!/bin/bash -l
set -euo pipefail
base=/work/mizutaniks/output/fld/collapse/wor124_rotmhd_20261116
cd "$base/source"
module use /opt/modulefiles /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load intel/2024.0 mpi/2021.13 hdf5-parallel/1.14.5
python3 configure.py --prob=protostellar_core_fld --coord=cartesian \
  --eos=general/eos_table -b --grav=mg -nrmgfld --flux=hlld_fld \
  --nscalars=1 -hdf5 -mpi --ccmd=icpx --mpiccmd=mpiicpx --cflag=-O3 \
  > "$base/configure.log" 2>&1
make -j8 > "$base/build.log" 2>&1
sha256sum bin/athena > "$base/binary.sha256"
date -Is > "$base/build_completed.txt"
