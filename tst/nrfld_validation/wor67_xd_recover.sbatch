#!/usr/bin/env bash
# Recover a completed WOR-67 solve whose optional movie generation failed.
# Submit with --export=ALL,WOR67_RUN_NAME=<existing full-run directory>.
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/work/mizutaniks/wor67/slurm/%x_%j.log

set -euo pipefail
: "${WOR67_RUN_NAME:?Existing full-run directory required}"
case "$WOR67_RUN_NAME" in
  wor67_hpc_uniform_n2048|wor67_hpc_amr_base1024|wor67_hpc_smr_base512) ;;
  *) echo "Unexpected WOR67_RUN_NAME: $WOR67_RUN_NAME" >&2; exit 2 ;;
esac

cd /work/mizutaniks/wor67/athena-analysis
test "$(git rev-parse HEAD:src)" = "$(git rev-parse 9419d224:src)"
root=/work/mizutaniks/wor67/output/$WOR67_RUN_NAME
test -f "$root/cases/mach5/input.used"
test -f "$root/cases/mach5/run.log"
python3 tst/nrfld_validation/radiative_shock_validation.py --postprocess "$root"
