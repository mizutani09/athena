#!/bin/bash
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor120_analyze
#SBATCH --output=/work/mizutaniks/wor120/analyze_%j.out
set -euo pipefail
root=/work/mizutaniks/wor120
mkdir -p "$root/results"
cd "$root/athena"
/home/mizutaniks/.conda/envs/py311env/bin/python3 tools/rsg/wor120_xd_short_gate_summary.py \
  --case lte64_0p84d:$root/smoke64 \
  --case lte128_5d:$root/lte128_open5d \
  --case lte256_0p84d:$root/lte256_open1d \
  --case lte256_closed_stall:$root/closed_analysis \
  --json "$root/results/short_gates.json" \
  --figure "$root/results/short_gates.png"
