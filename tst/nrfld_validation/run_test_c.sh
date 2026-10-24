#!/usr/bin/env bash
set -u

# Run from the repository root after configuring nrfld_diff_amr_interface.
ATHENA_BIN=${ATHENA_BIN:-./bin/athena}
INPUT=${INPUT:-inputs/radiation/athinput.nrfld_diff_amr_interface}
OUTROOT=${OUTROOT:-tst/outputs/nrfld_test_c_$(date +%Y%m%d_%H%M%S)}
DIAGNOSTIC_VERBOSITY=${DIAGNOSTIC_VERBOSITY:-1}

if [[ ! -x "${ATHENA_BIN}" ]]; then
  echo "FATAL: executable not found: ${ATHENA_BIN}" >&2
  exit 2
fi

mkdir -p "${OUTROOT}"

run_case() {
  local tag=$1
  shift
  local out="${OUTROOT}/${tag}"
  mkdir -p "${out}"
  echo "===== ${tag} ====="
  "${ATHENA_BIN}" -i "${INPUT}" -d "${out}" \
    nrfld/diagnostic_verbosity="${DIAGNOSTIC_VERBOSITY}" "$@" \
    >"${OUTROOT}/${tag}.log" 2>&1
  local status=$?
  if grep -q 'FATAL ERROR' "${OUTROOT}/${tag}.log"; then
    status=1
  fi
  local rejected
  rejected=$(grep -c 'Rejecting divergent V-cycle' "${OUTROOT}/${tag}.log" || true)
  local nr
  nr=$(grep '^\[NR\] status=' "${OUTROOT}/${tag}.log" | tail -n 1 || true)
  echo "TEST_C_RESULT case=${tag} status=${status} rejected_vcycles=${rejected} ${nr}"
  return "${status}"
}

overall=0

# C0: neither material interface nor AMR.
run_case c0_uniform_coeff_uniform_grid \
  mesh/refinement=none problem/density_high=1.0 || overall=1

# C1: AMR alone.
run_case c1_uniform_coeff_amr \
  mesh/refinement=static problem/density_high=1.0 || overall=1

# C2: material interface alone.
run_case c2_variable_coeff_uniform_grid \
  mesh/refinement=none problem/density_high=1.15 || overall=1

# C3a: material interfaces coincide with the two AMR boundaries.
run_case c3a_variable_coeff_amr_aligned \
  mesh/refinement=static problem/density_high=1.15 \
  problem/slab_left=0.25 problem/slab_right=0.75 \
  problem/pulse_center=0.25 || overall=1

# C3b: both material interfaces lie inside the uniformly fine region.
run_case c3b_variable_coeff_amr_shifted \
  mesh/refinement=static problem/density_high=1.15 \
  problem/slab_left=0.375 problem/slab_right=0.625 \
  problem/pulse_center=0.375 || overall=1

exit "${overall}"
