#!/usr/bin/env bash
# WOR-67 full Mach-5 run after a same-configuration 100-cycle sizing run.
#SBATCH --partition=M-large-a
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/work/mizutaniks/wor67/slurm/%x_%j.log

set -euo pipefail
module load hdf5-parallel/1.14.5
export PATH="${I_MPI_ROOT}/bin:${PATH}"
for var in CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
  IFS=: read -r -a entries <<< "${!var:-}"
  filtered=()
  for entry in "${entries[@]}"; do
    case "$entry" in
      /work/opt/local/miniconda3/py312_24.9.2/*) ;;
      *) filtered+=("$entry") ;;
    esac
  done
  export "$var=$(IFS=:; printf '%s' "${filtered[*]}")"
done
export I_MPI_HYDRA_BOOTSTRAP=slurm
command -v mpicxx
mpicxx -show | head -1
command -v mpirun

: "${WOR67_LAYOUT:?}"
: "${WOR67_NX1:?}"
: "${WOR67_RUN_NAME:?}"
: "${WOR67_REUSE_BUILD:?Pilot build_source directory required}"
WOR67_RANKS=${WOR67_RANKS:-${SLURM_NTASKS}}
WOR67_OUTPUT_ROOT=${WOR67_OUTPUT_ROOT:-/work/mizutaniks/wor67/output}
cd /work/mizutaniks/wor67/athena
test "$(git rev-parse --short HEAD)" = 9419d224

python3 tst/nrfld_validation/radiative_shock_validation.py \
  --output-root "$WOR67_OUTPUT_ROOT" --run-name "$WOR67_RUN_NAME" \
  --case mach5 --preset hpc --layout "$WOR67_LAYOUT" \
  --nx1-mach5 "$WOR67_NX1" --slab 4 --mpi-ranks "$WOR67_RANKS" \
  --reuse-build "$WOR67_REUSE_BUILD" --run-timeout 82800
