#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=wor120_l3_t01cmp
#SBATCH --output=/work/mizutaniks/wor120/hd4wz_scf_l3_t01_compare_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1

repository=$HOME/simulation/athena-wor120-hd4wz
x2_history=${SCF_X2_HISTORY:?set SCF_X2_HISTORY}
wz_history=${SCF_4WZ_HISTORY:?set SCF_4WZ_HISTORY}
run=/work/mizutaniks/wor120/hd4wz_scf_l3_t01_compare_${SLURM_JOB_ID}
test -s "$x2_history"
test -s "$wz_history"
mkdir -p "$run"
cd "$repository"

git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum "$x2_history" "$wz_history" > "$run/provenance.sha256"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor120/compare_scf_ab.cpp -o "$run/compare_scf_ab"
"$run/compare_scf_ab" "$x2_history" "$wz_history" \
  "$run/ab_comparison.json"
cat "$run/ab_comparison.json"
