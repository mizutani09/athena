#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=12:00:00
#SBATCH --job-name=wor177_corecut_l7_t1s2
#SBATCH --output=/work/mizutaniks/wor177/corecut6p28_l7_t1_stage2_%j.out

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
input=inputs/rsg/athinput.mesa_scf_rsg_corecut6p28_taper_surface_l2_l7_4wz
restart=/work/mizutaniks/wor177/corecut6p28_l7_t065_344983/continuation/mesa_scf_rsg_corecut6p28_l7_4wz.final.rst
history_prefix=/work/mizutaniks/wor177/corecut6p28_l7_t065_344983/continuation/mesa_scf_rsg_corecut6p28_l7_4wz.hst
history_header=/work/mizutaniks/wor177/corecut6p28_l6_l7_344840/l7/mesa_scf_rsg_corecut6p28_l7_4wz.hst
run_root=/work/mizutaniks/wor177/corecut6p28_l7_t1_stage2_${SLURM_JOB_ID}
run=$run_root/continuation
target_time=470.06
slice_dt=47.006

test -s "$restart"
test -s "$history_prefix"
test -s "$history_header"
mkdir -p "$run"
cd "$repository"
test -x bin/athena

git rev-parse HEAD > "$run_root/git_commit.txt"
git status --short > "$run_root/git_status.txt"
sha256sum bin/athena "$input" "$restart" "$history_prefix" "$history_header" \
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

/usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
  -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -i "$input" -d "$run" \
  time/nlim=-1 time/tlim="$target_time" time/ncycle_out=1000 \
  output1/dcycle=1000 output2/dcycle=1000000 output3/dt="$slice_dt" \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo "L7 stage-2 continuation produced a numeric or runtime failure" >&2
  exit 1
fi
if grep -q 'Warning in MultigridDriver::SolveIterative' "$run/stdout.log"; then
  echo "L7 stage-2 continuation unexpectedly entered iterative MG polishing" >&2
  exit 1
fi

"$run_root/analyze_relaxation_history" \
  <(sed -n '/^[[:space:]]*[0-9+.-]/p' "$history_prefix"; \
    sed -n '/^[[:space:]]*[0-9+.-]/p' \
      "$run/mesa_scf_rsg_corecut6p28_l7_4wz.hst") \
  "$run/relaxation_diagnostics.csv" "$run/relaxation_summary.json" \
  1.03413107e7 "$history_header"
while IFS= read -r snapshot; do
  "$run_root/analyze_relaxation_slice" "$run" "$snapshot" \
    "$run/slice_${snapshot}.csv" "$run/slice_${snapshot}.json" 0.0 out3
done < <(find "$run" -maxdepth 1 -type f -name '*.out3.*.tab' -printf '%f\n' \
  | sed -E 's/.*\.out3\.([0-9]+)\.tab/\1/' | sort -u)
set +e
"$run_root/evaluate_relaxation_gates" "$run/relaxation_diagnostics.csv" \
  "$run/relaxation_gates.json"
gate_rc=$?
set -e
if [ "$gate_rc" -eq 2 ]; then
  echo "gate evaluator could not parse combined L7 history" >&2
  exit 1
fi
printf 'physical_gate_exit_code=%s\n' "$gate_rc" > "$run/gate_status.txt"
test -s "$run/mesa_scf_rsg_corecut6p28_l7_4wz.final.rst"

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run_root/slurm_accounting.txt" || true
cat "$run/time.log"
cat "$run/relaxation_summary.json"
cat "$run/relaxation_gates.json"
cat "$run/gate_status.txt"
