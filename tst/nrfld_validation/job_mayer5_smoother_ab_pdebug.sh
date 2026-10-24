#!/bin/bash
#SBATCH --partition=P-debug
#SBATCH --nodes=1
#SBATCH --ntasks=112
#SBATCH --mem=430G
#SBATCH --time=00:10:00
#SBATCH --array=0-2
#SBATCH --job-name=m5_smab
#SBATCH --output=smoother_ab_slurm_%A_%a.txt

# One-cycle, same-restart comparison of two bounded NR-FLD solver candidates.
# Edit only the three paths in the site configuration block when relocating
# the run. No environment variables are required.

set -u

# -------------------------- site configuration --------------------------
WORK_ROOT="/work/mizutaniks/output/fld/collapse/test11"
ATHENA="${WORK_ROOT}/athena_wor25"
SOURCE_RST="${WORK_ROOT}/run12h_325898/protostellar_core_fld_mayer5.00004.rst"
RESTART_READER="${WORK_ROOT}/read_athena_restart_metadata.py"
# -----------------------------------------------------------------------

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="rb"
    SMOOTHER="jacobi-rb"
    DIVERGENCE_PATIENCE=1
    ;;
  1)
    TAG="patience2"
    SMOOTHER="jacobi-rb"
    DIVERGENCE_PATIENCE=2
    ;;
  2)
    TAG="double"
    SMOOTHER="jacobi-double"
    DIVERGENCE_PATIENCE=1
    ;;
  *)
    echo "FATAL: SLURM_ARRAY_TASK_ID must be 0, 1, or 2" >&2
    exit 2
    ;;
esac

for REQUIRED in "${ATHENA}" "${SOURCE_RST}" "${RESTART_READER}"; do
  if [[ ! -f "${REQUIRED}" ]]; then
    echo "FATAL: required file not found: ${REQUIRED}" >&2
    exit 2
  fi
done

START_CYCLE=$(python3 "${RESTART_READER}" "${SOURCE_RST}" --field cycle) || exit 2
TARGET_CYCLE=$((START_CYCLE + 1))
RUN_DIR="${WORK_ROOT}/smoother_ab_${TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
LOG="${WORK_ROOT}/smoother_ab_${TAG}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"
SUMMARY="${RUN_DIR}/numerical_summary.txt"

if [[ -e "${RUN_DIR}" ]]; then
  echo "FATAL: output directory already exists: ${RUN_DIR}" >&2
  exit 2
fi
mkdir -p "${RUN_DIR}"

# Loading a minimal input together with -r is the supported Athena++ way to
# add a parameter which is absent from an older restart header. It leaves all
# other restart parameters untouched. A common one-cycle xline output permits
# a field comparison without writing a full 3-D dump.
OVERLAY="${RUN_DIR}/ab_overlay.in"
printf '<output6>\ndcycle = 1\n' >"${OVERLAY}"
if [[ "${DIVERGENCE_PATIENCE}" -eq 2 ]]; then
  printf '<nrfld>\nlinear_divergence_patience = 2\nlinear_max_transient_growth = 2.0\n' \
    >>"${OVERLAY}"
fi
INPUT_ARGS=(-i "${OVERLAY}")

{
  echo "Mayer Section 5 pre-failure smoother A/B"
  echo "tag=${TAG}"
  echo "smoother=${SMOOTHER}"
  echo "divergence_patience=${DIVERGENCE_PATIENCE}"
  echo "source_restart=${SOURCE_RST}"
  echo "start_cycle=${START_CYCLE}"
  echo "target_cycle=${TARGET_CYCLE}"
  echo "run_dir=${RUN_DIR}"
  echo "athena=${ATHENA}"
  sha256sum "${ATHENA}"
  python3 "${RESTART_READER}" "${SOURCE_RST}"
  date
} | tee "${RUN_DIR}/provenance.txt"

# Preserve every parameter stored in the restart except the selected candidate
# and the absolute cycle limit. In particular, do not change the linear cap,
# Newton tolerances, post smoother, implicit update count, gravity, or timestep.
set +e
srun --ntasks=112 "${ATHENA}" \
  -r "${SOURCE_RST}" \
  "${INPUT_ARGS[@]}" \
  -d "${RUN_DIR}" \
  -t 00:08:30 \
  "nrfld/smoother=${SMOOTHER}" \
  "time/nlim=${TARGET_CYCLE}" \
  >"${LOG}" 2>&1
RAW_STATUS=$?
set -e

FINAL_RST="${RUN_DIR}/protostellar_core_fld_mayer5.final.rst"
FINAL_CYCLE=-1
if [[ -f "${FINAL_RST}" ]]; then
  FINAL_CYCLE=$(python3 "${RESTART_READER}" "${FINAL_RST}" --field cycle 2>/dev/null || echo -1)
fi

STRICT=$(grep -c '^\[NR\] status=converged ' "${LOG}" || true)
RELAXED=$(grep -c '^\[NR\] status=accepted_relaxed ' "${LOG}" || true)
FAILURES=$(grep -Ec '^\[NR\] status=(backtracking_exhausted|max_iterations|stagnation|initial_nonfinite|final_nonfinite|unknown)' \
  "${LOG}" || true)
REJECTED=$(grep -c 'Rejecting divergent V-cycle' "${LOG}" || true)
BACKTRACKS=$(grep -Ec '^\[NR_TRIAL\].*accepted=0' "${LOG}" || true)
LINEAR_SLOW=$(grep -c 'Slow multigrid convergence occurred' "${LOG}" || true)

RESULT="FAIL"
if [[ "${FINAL_CYCLE}" -eq "${TARGET_CYCLE}" && "${FAILURES}" -eq 0 \
      && $((STRICT + RELAXED)) -gt 0 ]]; then
  RESULT="PASS"
fi

{
  echo "tag=${TAG}"
  echo "smoother=${SMOOTHER}"
  echo "divergence_patience=${DIVERGENCE_PATIENCE}"
  echo "raw_status=${RAW_STATUS}"
  echo "start_cycle=${START_CYCLE}"
  echo "target_cycle=${TARGET_CYCLE}"
  echo "final_cycle=${FINAL_CYCLE}"
  echo "strict_solves=${STRICT}"
  echo "relaxed_solves=${RELAXED}"
  echo "failed_solves=${FAILURES}"
  echo "rejected_vcycles=${REJECTED}"
  echo "backtracks=${BACKTRACKS}"
  echo "slow_linear_solves=${LINEAR_SLOW}"
  echo "numerical_result=${RESULT}"
  echo "final_restart=${FINAL_RST}"
  echo "log=${LOG}"
} | tee "${SUMMARY}"

echo "---------------- NR RESULTS ----------------"
grep '^\[NR\] status=' "${LOG}" || true
echo "---------------- TERMINATION ----------------"
grep -E 'Terminating on|FATAL ERROR|Out Of Memory|srun: error' "${LOG}" | tail -n 20 || true
echo "NUMERICAL_RESULT ${TAG}=${RESULT} raw_status=${RAW_STATUS} final_cycle=${FINAL_CYCLE}"
date

# Do not propagate the known site-specific MPI shutdown status. Numerical
# success is encoded above from the restart cycle and Newton reasons.
exit 0
