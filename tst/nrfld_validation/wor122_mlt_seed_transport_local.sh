#!/usr/bin/env bash
# Short matched seed tests for the Henyey-MLT surface patch.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
binary=${1:?usage: $0 /absolute/path/to/athena [output-root]}
out_root=${2:-/tmp/wor122_mlt_seed_transport}
mkdir -p "$out_root"
cd "$repo"

for case in full nodiff closedtop fixedtop; do
  run="$out_root/$case"
  mkdir -p "$run"
  cut_diff=false
  marshak_alpha=0.5
  top_boundary=marshak_vacuum
  if [[ "$case" == nodiff ]]; then cut_diff=true; fi
  if [[ "$case" == closedtop ]]; then marshak_alpha=1e-8; fi
  if [[ "$case" == fixedtop ]]; then top_boundary=fixed_temperature; fi
  mpirun --oversubscribe -np 4 "$binary" \
    -i inputs/rsg/athinput.wor122_henyey_mlt_patch_discrete -d "$run" \
    "job/problem_id=wor122_mlt_seed_$case" \
    "fld/cut_diff=$cut_diff" "problem/marshak_alpha=$marshak_alpha" \
    "problem/top_radiation_boundary=$top_boundary" \
    problem/density_perturb_frac=0.02 \
    problem/buoyancy_velocity_frac=0.20 \
    time/tlim=8 time/cfl_number=0.03 \
    output8/dt=4 output9/dt=4 output10/dt=8 \
    > "$run/run.log" 2> "$run/stderr.log"
  python3 tools/rsg/analyze_rsg_surface_transfer.py "$run" \
    --output "$run/transport.json" --snapshots 0,1,2
  python3 tools/rsg/analyze_rsg_entropy_mode.py "$run" \
    --output "$run/mode.json"
done
python3 tools/rsg/plot_wor122_mlt_seed_transport.py \
  "$out_root/full" "$out_root/nodiff" "$out_root/closedtop" \
  "$out_root/fixedtop" \
  --output-json "$out_root/summary.json" \
  --output-figure "$out_root/summary.png"
