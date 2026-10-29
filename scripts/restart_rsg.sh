#!/usr/bin/env bash
set -euo pipefail

restart=${1:?usage: restart_rsg.sh RESTART RUN_DIR}
run_dir=${2:?usage: restart_rsg.sh RESTART RUN_DIR}
nproc=${NPROC:-128}
mkdir -p "$run_dir"
cp "$restart" "$run_dir/restart.input.rst"
sha256sum "$restart" > "$run_dir/restart.sha256"
mpirun -n "$nproc" ./bin/athena -r "$restart" -d "$run_dir" \
  > "$run_dir/stdout.log" 2> "$run_dir/stderr.log"
