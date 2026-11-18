#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor122_prep256_release_analysis
#SBATCH --output=/work/mizutaniks/wor122/prep256_release_analysis_%j.out
set -euo pipefail

base=/work/mizutaniks/wor122
cd "$base/athena_prep_thermal"
export PYTHONPATH="$PWD/tst/nrfld_validation${PYTHONPATH:+:$PYTHONPATH}"
python=/home/mizutaniks/.conda/envs/py311env/bin/python3
input=inputs/rsg/athinput.wor122_rsg_smr_onepatch
case_args=()
axis_args=()

for mode in hold off; do
  if [[ "$mode" == hold ]]; then
    initial_mode=hold
  else
    initial_mode=taper
  fi
  old="$base/prep256_${initial_mode}_t120/wor122_prep256_${initial_mode}_t120"
  run="$base/prep256_${mode}_t130"
  new="$run/wor122_prep256_${mode}_t130"
  pold="${old}.prim.00028.athdf"
  uold="${old}.uov.00028.athdf"
  pnew="${new}.prim.00030.athdf"
  unew="${new}.uov.00030.athdf"
  for path in "$pold" "$uold" "$pnew" "$unew"; do test -s "$path"; done
  case_args+=(--case "$mode" "$pold" "$uold" "$pnew" "$unew")
  axis_args+=(--snapshot "${mode}_t120" "$pold"
              --snapshot "${mode}_t130" "$pnew")
  ln -sfn "$pold" "$run/a_initial.prim.00000.athdf"
  ln -sfn "$uold" "$run/a_initial.uov.00000.athdf"
  "$python" tools/rsg/wor120_convective_flux_audit.py \
    --eos data/eos/rsg_tomida2013_material_eos.tab \
    --case "prep256_${mode}_t130:$run" \
    --json "$run/convective_flux.json" \
    --figure "$run/convective_flux.png"
  "$python" "$base/wor122_analyze_snapshot.py" "$pnew" "$unew" \
    > "$run/snapshot_summary.json"
done
"$python" tst/nrfld_validation/wor122_rsg_shell_evolution.py \
  --input "$input" "${case_args[@]}" \
  --output "$base/prep256_release_shell_evolution.json"
"$python" "$base/wor122_axis_flow_monitor.py" \
  --input "$input" "${axis_args[@]}" \
  --output "$base/prep256_release_axis_flow.json"
