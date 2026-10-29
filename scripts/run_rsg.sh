#!/usr/bin/env bash
set -euo pipefail

input=${1:?usage: run_rsg.sh INPUT RUN_DIR}
run_dir=${2:?usage: run_rsg.sh INPUT RUN_DIR}
nproc=${NPROC:-128}
actual_input=$input

if grep -q '^base[[:space:]]*=' "$input"; then
  mkdir -p "$run_dir"
  actual_input="$run_dir/athinput.materialized"
  python3 scripts/materialize_rsg_input.py "$input" "$actual_input"
fi

scripts/capture_rsg_manifest.sh "$actual_input" "$run_dir"
mpirun -n "$nproc" ./bin/athena -i "$run_dir/athinput" -d "$run_dir" \
  > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
