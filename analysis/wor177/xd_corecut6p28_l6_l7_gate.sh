#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --job-name=wor177_corecut_l67
#SBATCH --output=/work/mizutaniks/wor177/corecut6p28_l6_l7_%j.out

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

repository=$HOME/simulation/athena-wor177-core-excision-hd4wz
input_l6=inputs/rsg/athinput.mesa_scf_rsg_corecut6p28_taper_surface_l2_l6_4wz
input_l7=inputs/rsg/athinput.mesa_scf_rsg_corecut6p28_taper_surface_l2_l7_4wz
profile=data/ic/rsg21_mesa_scf_corecut6p28_g5p3_nr16384_taper_rho3e-9_gamma5_3.csv
run_root=/work/mizutaniks/wor177/corecut6p28_l6_l7_${SLURM_JOB_ID}
# t_dyn = 470.06 code units for the same 21-Msun, 931.9-Rsun scaling.
target_time=11.7515
slice_dt=2.3503

mkdir -p "$run_root"
cd "$repository"
test -x bin/athena
test -s "$profile"

git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
sha256sum bin/athena "$input_l6" "$input_l7" "$profile" \
  src/pgen/scf_rsg.cpp analysis/wor162/analyze_relaxation_history.cpp \
  analysis/wor162/analyze_relaxation_slice.cpp \
  analysis/wor162/evaluate_relaxation_gates.cpp \
  > "$run_root/provenance.sha256"

icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_history.cpp \
  -o "$run_root/analyze_relaxation_history"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/analyze_relaxation_slice.cpp \
  -o "$run_root/analyze_relaxation_slice"
icpx -O2 -std=c++11 -Wall -Wextra -Werror -pedantic \
  analysis/wor162/evaluate_relaxation_gates.cpp \
  -o "$run_root/evaluate_relaxation_gates"

run_case() {
  local label=$1
  local input=$2
  local problem_id=$3
  local run=$run_root/$label
  mkdir -p "$run"
  /usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
    -o "$run/time.log" \
    srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
    bin/athena -i "$input" -d "$run" \
    time/nlim=-1 time/tlim="$target_time" time/ncycle_out=100 \
    output1/dcycle=250 \
    output2/dcycle=-1 output2/dt="$target_time" \
    output2/next_time="$target_time" \
    output3/dt="$slice_dt" output3/next_time=0.0 \
    > "$run/stdout.log" 2> "$run/stderr.log"
  grep -q 'Terminating on time limit' "$run/stdout.log"
  if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
      "$run/stdout.log" "$run/stderr.log"; then
    echo "$label produced a numeric or runtime failure" >&2
    exit 1
  fi
  if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
    echo "$label unexpectedly entered iterative multigrid polishing" >&2
    exit 1
  fi
  "$run_root/analyze_relaxation_history" \
    "$run/${problem_id}.hst" \
    "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json"
  for table in "$run"/*.out3.*.tab; do
    test -e "$table" || continue
    snapshot=${table##*.out3.}
    snapshot=${snapshot%.tab}
    "$run_root/analyze_relaxation_slice" "$run" "$snapshot" \
      "$run/slice_${snapshot}.csv" "$run/slice_${snapshot}.json" 0.0 out3
  done
  set +e
  "$run_root/evaluate_relaxation_gates" "$run/relaxation_diagnostics.csv" \
    "$run/relaxation_gates.json"
  local gate_rc=$?
  set -e
  if [ "$gate_rc" -eq 2 ]; then
    echo "$label gate evaluator could not parse its input" >&2
    exit 1
  fi
  printf 'physical_gate_exit_code=%s\n' "$gate_rc" > "$run/gate_status.txt"
  test -s "$run/${problem_id}.final.rst"
}

run_case l6 "$input_l6" mesa_scf_rsg_corecut6p28_l6_4wz
run_case l7 "$input_l7" mesa_scf_rsg_corecut6p28_l7_4wz

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run_root/slurm_accounting.txt" || true

for label in l6 l7; do
  echo "--- $label ---"
  cat "$run_root/$label/time.log"
  cat "$run_root/$label/relaxation_summary.json"
  cat "$run_root/$label/relaxation_gates.json"
  cat "$run_root/$label/gate_status.txt"
done
