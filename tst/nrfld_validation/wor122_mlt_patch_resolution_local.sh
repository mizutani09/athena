#!/usr/bin/env bash
# Reproduce the short Henyey-MLT local-patch resolution screen.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
binary=${1:?usage: $0 /absolute/path/to/athena [output-root]}
out_root=${2:-/tmp/wor122_mlt_patch_resolution}
mkdir -p "$out_root"
cd "$repo"

for spec in '0p625 0.625 96 5' '0p3125 0.3125 192 2.5' '0p15625 0.15625 384 1.25'; do
  read -r tag dx nz half <<< "$spec"
  run="$out_root/dx${tag}"
  mkdir -p "$run"
  mpirun --oversubscribe -np 4 "$binary" \
    -i inputs/rsg/athinput.wor122_henyey_mlt_patch_discrete -d "$run" \
    "job/problem_id=wor122_henyey_dx${tag}" \
    "mesh/nx3=$nz" \
    "mesh/x1min=-$half" "mesh/x1max=$half" \
    "mesh/x2min=-$half" "mesh/x2max=$half" \
    > "$run/run.log" 2> "$run/stderr.log"
  python3 tst/nrfld_validation/wor122_initial_diffusion_audit.py \
    --log "$run/run.log" --dx "$dx" --lheat 229655.17870623546 \
    --output "$run/audit.json"
done
