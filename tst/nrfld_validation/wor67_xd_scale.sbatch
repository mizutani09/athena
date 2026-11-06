#!/usr/bin/env bash
# Matched-output-cadence 100-cycle strong-scaling measurement for WOR-67.
# Submit with WOR67_SOURCE_PILOT, WOR67_RUN_NAME, WOR67_REUSE_BUILD, WOR67_PPN.
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
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

: "${WOR67_SOURCE_PILOT:?Completed N4096 pilot directory required}"
: "${WOR67_RUN_NAME:?Unique scale run name required}"
: "${WOR67_REUSE_BUILD:?Pinned build_source required}"
: "${WOR67_PPN:?MPI ranks per node required}"
WOR67_REPO_DIR=${WOR67_REPO_DIR:-/work/mizutaniks/wor67/athena-analysis}
WOR67_OUTPUT_ROOT=${WOR67_OUTPUT_ROOT:-/work/mizutaniks/wor67/output}
cd "$WOR67_REPO_DIR"
test "$(git rev-parse HEAD:src)" = "$(git rev-parse 9419d224:src)"
exe=$WOR67_REUSE_BUILD/bin/athena
test -x "$exe"
test "$(sha256sum "$exe" | cut -d' ' -f1)" = 8dcf371fa64a99dd56c655e7934efc53b549ac58bc4924718b59cdbc3b45728f
test -f "$WOR67_SOURCE_PILOT/cases/mach5/input.used"
test "$SLURM_NTASKS" -eq "$((SLURM_JOB_NUM_NODES * WOR67_PPN))"

case_dir=$WOR67_OUTPUT_ROOT/$WOR67_RUN_NAME/cases/mach5
mkdir -p "$case_dir/output"
python3 - "$WOR67_SOURCE_PILOT/cases/mach5/input.used" "$case_dir/input.used" <<'PY'
from pathlib import Path
import sys
section = None
lines = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if line.startswith("<") and line.endswith(">"):
        section = line[1:-1]
    if section in ("output2", "output3") and line.lstrip().startswith("dt ="):
        line = "dt = 0.002"
    lines.append(line)
Path(sys.argv[2]).write_text("\n".join(lines) + "\n")
PY
{
  printf 'job_id=%s\nnodes=%s\nranks=%s\nppn=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$WOR67_PPN"
  printf 'source_pilot=%s\nexe_sha256=%s\ninput_sha256=%s\n' \
    "$WOR67_SOURCE_PILOT" "$(sha256sum "$exe" | cut -d' ' -f1)" \
    "$(sha256sum "$case_dir/input.used" | cut -d' ' -f1)"
  printf 'source_tree=%s\n' "$(git rev-parse HEAD:src)"
} > "$case_dir/provenance.txt"

cd "$case_dir"
/usr/bin/time -v -o resources.txt \
  mpirun -np "$SLURM_NTASKS" -ppn "$WOR67_PPN" "$exe" \
  -i input.used -d output >run.log 2>&1

python3 - "$WOR67_REPO_DIR" "$case_dir" <<'PY'
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(sys.argv[1])/"tst/nrfld_validation"))
from radiative_shock_validation import verify_case
result = verify_case(Path(sys.argv[2]), "mach5", 100)
if result["solver"]["relaxed_accepted"]:
    raise ValueError("relaxed Newton acceptance in scale run")
Path("summary.json").write_text(json.dumps(result, indent=2)+"\n")
print("cycles", result["cycles"], "wall_seconds", result["wall_seconds"],
      "meshblocks", result["meshblocks_final"], flush=True)
PY
