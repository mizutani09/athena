#!/usr/bin/env bash
# Rebuild only the instrumented NRFLD source against the pinned WOR-67 binary.
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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

WOR67_REPO_DIR=${WOR67_REPO_DIR:-/work/mizutaniks/wor67/athena-analysis}
WOR67_PINNED_BUILD=${WOR67_PINNED_BUILD:-/work/mizutaniks/wor67/output/wor67_hpc_uniform_n2048/build/build_source}
WOR67_LEDGER_BUILD=${WOR67_LEDGER_BUILD:-/work/mizutaniks/wor67/build/wor67_interface_ledger}
cd "$WOR67_REPO_DIR"
git merge-base --is-ancestor c6cc5214 HEAD
git diff --quiet c6cc5214 HEAD -- src/nr_multigrid/NRFLD.cpp
test "$(sha256sum "$WOR67_PINNED_BUILD/bin/athena" | cut -d' ' -f1)" = \
  8dcf371fa64a99dd56c655e7934efc53b549ac58bc4924718b59cdbc3b45728f
test ! -e "$WOR67_LEDGER_BUILD"
mkdir -p "$(dirname "$WOR67_LEDGER_BUILD")"
cp -a "$WOR67_PINNED_BUILD" "$WOR67_LEDGER_BUILD"
cp src/nr_multigrid/NRFLD.cpp "$WOR67_LEDGER_BUILD/src/nr_multigrid/NRFLD.cpp"
cd "$WOR67_LEDGER_BUILD"
make -j8
{
  printf 'source_commit=%s\n' "$(git -C "$WOR67_REPO_DIR" rev-parse HEAD)"
  printf 'source_file_sha256=%s\n' "$(sha256sum src/nr_multigrid/NRFLD.cpp | cut -d' ' -f1)"
  printf 'binary_sha256=%s\n' "$(sha256sum bin/athena | cut -d' ' -f1)"
} > "$WOR67_LEDGER_BUILD/provenance.txt"
cat "$WOR67_LEDGER_BUILD/provenance.txt"
