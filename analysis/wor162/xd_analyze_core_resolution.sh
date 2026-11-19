#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor162_core_res
#SBATCH --output=/work/mizutaniks/wor162/core_resolution_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load umf/0.10.0
module load PrgEnv-oneapi/2025.1

repository=$HOME/simulation/athena-wor162
work=/work/mizutaniks/wor162
run=$work/core_resolution_${SLURM_JOB_ID}
mkdir -p "$run"
cd "$repository"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_core_mode.cpp -o "$run/analyze_core_mode"

"$run/analyze_core_mode" "$work/root64_flux_map_342090" \
  "$run/core_l5.csv" 00000 00001
"$run/analyze_core_mode" "$work/root64_l6_342048" \
  "$run/core_l6.csv" 00000 00001 00002 00003 00004 00005
"$run/analyze_core_mode" "$work/root64_l7_342039" \
  "$run/core_l7.csv" 00000 00001 00002 00003 00004 00005
"$run/analyze_core_mode" "$work/surface_l3_t01_342434" \
  "$run/core_surface_l6.csv" 00000 00002 00004 00006 00008 00010

for result in "$run"/core_*.csv; do
  echo "== $result"
  cat "$result"
done
