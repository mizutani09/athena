#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor162_core_mode
#SBATCH --output=/work/mizutaniks/wor162/core_mode_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load umf/0.10.0
module load PrgEnv-oneapi/2025.1

repository=$HOME/simulation/athena-wor162
early=/work/mizutaniks/wor162/surface_l3_t01_342434
late=/work/mizutaniks/wor162/surface_l3_t055_342602
run=/work/mizutaniks/wor162/core_mode_${SLURM_JOB_ID}
mkdir -p "$run"
cd "$repository"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_core_mode.cpp -o "$run/analyze_core_mode"
/usr/bin/time -f 'early_wall_s=%e maxrss_kb=%M' -o "$run/time_early.log" \
  "$run/analyze_core_mode" "$early" "$run/core_mode_early.csv" \
  00000 00005 00010
/usr/bin/time -f 'late_wall_s=%e maxrss_kb=%M' -o "$run/time_late.log" \
  "$run/analyze_core_mode" "$late" "$run/core_mode_late.csv" \
  00015 00020 00025 00030 00035 00040 00045 00050 00055

cat "$run/time_early.log"
cat "$run/time_late.log"
cat "$run/core_mode_early.csv"
cat "$run/core_mode_late.csv"
