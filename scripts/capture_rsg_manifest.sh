#!/usr/bin/env bash
set -euo pipefail

input=${1:?usage: capture_rsg_manifest.sh INPUT RUN_DIR}
run_dir=${2:?usage: capture_rsg_manifest.sh INPUT RUN_DIR}
mkdir -p "$run_dir"

git rev-parse HEAD > "$run_dir/git_commit.txt"
git status --short > "$run_dir/git_status.txt"
git diff --binary > "$run_dir/git_diff.patch"
cp "$input" "$run_dir/athinput"
sha256sum "$input" > "$run_dir/input.sha256"
sha256sum data/eos/* data/opacity/* data/ic/* > "$run_dir/physics_tables.sha256"
mpicxx --version > "$run_dir/compiler.txt"
mpiexec --version > "$run_dir/mpi.txt" 2>&1
./bin/athena -c > "$run_dir/athena_config.txt"
