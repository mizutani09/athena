#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --job-name=wor120_l3_m2cmp
#SBATCH --output=/work/mizutaniks/wor120/hd4wz_scf_l3_memory2_t01_compare_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1

repository=$HOME/simulation/athena-wor120-hd4wz-memory2
x2_history=${SCF_X2_HISTORY:?set SCF_X2_HISTORY}
wz_history=${SCF_4WZ_HISTORY:?set SCF_4WZ_HISTORY}
x2_run=$(dirname "$x2_history")
wz_run=$(dirname "$wz_history")
x2_job_id=${SCF_X2_JOB_ID:-${x2_run##*_}}
wz_job_id=${SCF_4WZ_JOB_ID:-${wz_run##*_}}
run=/work/mizutaniks/wor120/hd4wz_scf_l3_memory2_t01_compare_${SLURM_JOB_ID}
test -s "$x2_history"
test -s "$wz_history"
test -s "$x2_run/stdout.log"
test -s "$x2_run/time.log"
test -s "$wz_run/stdout.log"
test -s "$wz_run/time.log"
mkdir -p "$run"
cd "$repository"

git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum "$x2_history" "$wz_history" > "$run/provenance.sha256"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor120/compare_scf_ab.cpp -o "$run/compare_scf_ab"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor120/compare_scf_cost.cpp -o "$run/compare_scf_cost"
"$run/compare_scf_ab" "$x2_history" "$wz_history" \
  "$run/ab_comparison.json"
sacct -j "${x2_job_id}.0" --units=K -n -P \
  -o JobID,State,ElapsedRaw,AveRSS,MaxRSS,MaxRSSNode,MaxRSSTask \
  > "$run/x2_sacct.txt"
sacct -j "${wz_job_id}.0" --units=K -n -P \
  -o JobID,State,ElapsedRaw,AveRSS,MaxRSS,MaxRSSNode,MaxRSSTask \
  > "$run/4wz_sacct.txt"
"$run/compare_scf_cost" \
  "$x2_run/stdout.log" "$x2_run/time.log" "$run/x2_sacct.txt" \
  "$wz_run/stdout.log" "$wz_run/time.log" "$run/4wz_sacct.txt" \
  "$run/cost_comparison.json"
cat "$run/ab_comparison.json"
cat "$run/cost_comparison.json"
