#!/usr/bin/env bash
# Run after the three N2048-equivalent full jobs have completed.
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mizutaniks/wor67/slurm/%x_%j.log

set -euo pipefail
cd /work/mizutaniks/wor67/athena
test "$(git rev-parse HEAD:src)" = "$(git rev-parse 9419d224:src)"
root=/work/mizutaniks/wor67/output

python3 tst/nrfld_validation/wor67_spike_audit.py \
  --case "N2048=$root/wor67_hpc_uniform_n2048/cases/mach5" \
  --case "AMR1_N2048=$root/wor67_hpc_amr_base1024/cases/mach5" \
  --case "SMR3_N2048=$root/wor67_hpc_smr_base512/cases/mach5" \
  --output "$root/wor67_hpc_comparison_n2048"
