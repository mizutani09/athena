#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor122_prep256_taper_analysis
#SBATCH --output=/work/mizutaniks/wor122/prep256_taper_analysis_%j.out
set -euo pipefail

base=/work/mizutaniks/wor122
cd "$base/athena_prep_thermal"
export PYTHONPATH="$PWD/tst/nrfld_validation${PYTHONPATH:+:$PYTHONPATH}"
python=/home/mizutaniks/.conda/envs/py311env/bin/python3
input=inputs/rsg/athinput.wor122_rsg_smr_onepatch
old="$base/prep256_fgmres_r40_t110/wor122_prep256_fgmres_r40_t110"
pold="${old}.prim.00026.athdf"
uold="${old}.uov.00026.athdf"
test -s "$pold"
test -s "$uold"

args=()
axis_args=(--snapshot t110 "$pold")
for mode in hold taper; do
  run="$base/prep256_${mode}_t120"
  prefix="$run/wor122_prep256_${mode}_t120"
  pnew="${prefix}.prim.00028.athdf"
  unew="${prefix}.uov.00028.athdf"
  test -s "$pnew"
  test -s "$unew"
  args+=(--case "$mode" "$pold" "$uold" "$pnew" "$unew")
  axis_args+=(--snapshot "$mode" "$pnew")
  ln -sfn "$pold" "$run/a_initial.prim.00000.athdf"
  ln -sfn "$uold" "$run/a_initial.uov.00000.athdf"
  "$python" tools/rsg/wor120_convective_flux_audit.py \
    --eos data/eos/rsg_tomida2013_material_eos.tab \
    --case "prep256_${mode}_t120:$run" \
    --json "$run/convective_flux.json" \
    --figure "$run/convective_flux.png"
  "$python" "$base/wor122_analyze_snapshot.py" "$pnew" "$unew" \
    > "$run/snapshot_summary.json"
done
"$python" tst/nrfld_validation/wor122_rsg_shell_evolution.py \
  --input "$input" "${args[@]}" \
  --output "$base/prep256_taper_shell_evolution.json"
"$python" "$base/wor122_axis_flow_monitor.py" \
  --input "$input" "${axis_args[@]}" \
  --output "$base/prep256_taper_axis_flow.json"
