#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor120_hd4wz_ab
#SBATCH --output=/work/mizutaniks/wor120/hd4wz_scf_ab_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
export SLURM_MPI_TYPE=pmi2
export LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
ulimit -c 0

repository=$HOME/simulation/athena-wor120-hd4wz
run_root=/work/mizutaniks/wor120/hd4wz_scf_ab_${SLURM_JOB_ID}
input=inputs/rsg/athinput.scf_rsg_relaxation_surface_l3
mkdir -p "$run_root"
cd "$repository"

git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
sha256sum bin/athena "$input" src/eos/eos_high_order.cpp \
  src/reconstruct/wenoz_berta.cpp > "$run_root/provenance.sha256"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o "$run_root/analyze_relaxation_history"

run_case() {
  local label=$1
  local integrator=$2
  local xorder=$3
  local run=$run_root/$label
  mkdir -p "$run"
  /usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
    srun -n56 --ntasks-per-node=56 bin/athena -i "$input" -d "$run" \
    time/integrator="$integrator" time/xorder="$xorder" \
    time/nlim=1 time/tlim=1.0 \
    output1/dcycle=1 output2/dt=-1 output3/dt=-1 \
    output4/dt=-1 output5/dt=-1 \
    > "$run/stdout.log" 2> "$run/stderr.log"
  if grep -Eiq 'nan|fatal error|segmentation fault' "$run/stdout.log" "$run/stderr.log"; then
    echo "$label produced a numeric or runtime failure" >&2
    exit 1
  fi
  grep -q 'Terminating on cycle limit' "$run/stdout.log"
  "$run_root/analyze_relaxation_history" "$run/scf_rsg_relax.hst" \
    "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"
  printf '%s ' "$label"
  cat "$run/time.log"
  tail -n 2 "$run/scf_rsg_relax.hst"
  cat "$run/relaxation_summary.json"
}

run_case rk2_x2 rk2 2
run_case rk4_4wz rk4 4wz
