#!/usr/bin/env bash
#SBATCH --job-name=wor105_default_build
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mizutaniks/wor105/slurm/%x_%j.log

set -euo pipefail
source /etc/profile.d/modules.sh
source /work/opt/local/bin/enable-oneapi.sh
module swap PrgEnv-oneapi/2024.2 PrgEnv-oneapi/2025.1
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
repo=/work/mizutaniks/wor105/athena-mach5
cd "$repo"
test "$(git rev-parse --short HEAD)" = f4f6ec41
python3 configure.py --prob=nrfld_radiative_shock --eos=adiabatic \
  -nrmgfld -hdf5 -mpi --nghost=2 --cxx=g++
make -j8
{
  printf 'source_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'binary_sha256=%s\n' "$(sha256sum bin/athena | cut -d' ' -f1)"
} > /work/mizutaniks/wor105/build/provenance.txt
cat /work/mizutaniks/wor105/build/provenance.txt
