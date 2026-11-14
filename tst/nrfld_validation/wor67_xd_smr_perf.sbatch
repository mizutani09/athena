#!/usr/bin/env bash
# SMR 100-cycle timing with the full run's output cadence and identical binary.
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=96G
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
export I_MPI_HYDRA_BOOTSTRAP=slurm

root=/work/mizutaniks/wor67
case_dir=$root/output/wor67_hpc_smr_base512_pilot100_outputcadence
source_input=$root/output/wor67_hpc_smr_base512_pilot100_r64/cases/mach5/input.used
exe=$root/output/wor67_hpc_uniform_n2048_pilot100b/build/build_source/bin/athena
test "$(sha256sum "$exe" | cut -d' ' -f1)" = 8dcf371fa64a99dd56c655e7934efc53b549ac58bc4924718b59cdbc3b45728f
mkdir -p "$case_dir/output"

python3 - "$source_input" "$case_dir/input.used" <<'PY'
from pathlib import Path
import sys
section = None
out = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.startswith("<") and line.endswith(">"):
        section = line[1:-1]
    if section in ("output2", "output3") and line.lstrip().startswith("dt ="):
        line = "dt = 0.002"
    out.append(line)
Path(sys.argv[2]).write_text("\n".join(out) + "\n")
PY

cd "$case_dir"
/usr/bin/time -v -o resources.txt \
  mpirun -np 64 "$exe" -i input.used -d output >run.log 2>&1
