#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor162_l7_slice
#SBATCH --output=/work/mizutaniks/wor162/root64_l7_t01_slice_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load umf/0.10.0
module load PrgEnv-oneapi/2025.1

repository=$HOME/simulation/athena-wor162
input=/work/mizutaniks/wor162/root64_l7_t01_342926
run=/work/mizutaniks/wor162/root64_l7_t01_slice_${SLURM_JOB_ID}
mkdir -p "$run"
cd "$repository"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_slice.cpp -o "$run/analyze_relaxation_slice"
"$run/analyze_relaxation_slice" "$input" 00010 \
  "$run/slice_00010.csv" "$run/slice_00010.json"
cat "$run/slice_00010.json"
