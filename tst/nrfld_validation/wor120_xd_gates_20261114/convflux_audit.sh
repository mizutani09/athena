#!/bin/bash
#SBATCH -J wor120_convflux
#SBATCH -p P-debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /work/mizutaniks/wor120/results/convflux-%j.out
#SBATCH -e /work/mizutaniks/wor120/results/convflux-%j.err
set -euo pipefail
cd /work/mizutaniks/wor120/athena
/home/mizutaniks/.conda/envs/py311env/bin/python3 tools/rsg/wor120_convective_flux_audit.py \
  --eos data/eos/rsg_tomida2013_material_eos.tab \
  --case lte128:/work/mizutaniks/wor120/lte128_open5d \
  --case lte256:/work/mizutaniks/wor120/lte256_open1d \
  --json /work/mizutaniks/wor120/results/convective_flux.json \
  --figure /work/mizutaniks/wor120/results/convective_flux.png
