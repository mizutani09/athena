#!/bin/bash
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor120_cellaudit
#SBATCH --output=/work/mizutaniks/wor120/closed_cell_audit_%j.out
set -euo pipefail
/home/mizutaniks/.conda/envs/py311env/bin/python3 /work/mizutaniks/wor120/closed_cell_audit.py
