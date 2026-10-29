#!/usr/bin/env bash
set -euo pipefail

python3 configure.py \
  --prob=rsg_star_in_box \
  --coord=cartesian \
  --eos=general/eos_table \
  --flux=hllc \
  -nrmgfld -mpi -omp -hdf5 \
  ${FLD_CONFIG_FLAGS:-}

make clean
make -j"${BUILD_JOBS:-16}"
