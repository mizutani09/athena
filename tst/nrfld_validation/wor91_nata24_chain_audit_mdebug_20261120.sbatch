#!/usr/bin/env bash
# Audit the terminal NATA 24 km continuation after the chain finishes.
#SBATCH --job-name=wor91_nata24_final_audit
#SBATCH --account=a
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/work/mizutaniks/wor91/thermal_relax_20261116/hd2_nata24_20261119/final_audit_%j.out

set -euo pipefail
: "${SOURCE_RUN:?SOURCE_RUN is required}"
root=/work/mizutaniks/wor91
grep -qx 'validated=yes' "$SOURCE_RUN/result.txt"
/usr/bin/python3 "$root/scripts/wor91_hd2_warm96_audit_20261119.py" "$SOURCE_RUN"
