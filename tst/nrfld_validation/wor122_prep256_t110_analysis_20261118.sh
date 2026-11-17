#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor122_prep256_t110_analysis
#SBATCH --output=/work/mizutaniks/wor122/prep256_t110_analysis_%j.out
set -euo pipefail

base=/work/mizutaniks/wor122
cd "$base/athena_prep_thermal"
export PYTHONPATH="$PWD/tst/nrfld_validation${PYTHONPATH:+:$PYTHONPATH}"
python=/home/mizutaniks/.conda/envs/py311env/bin/python3
input=inputs/rsg/athinput.wor122_rsg_smr_onepatch

prefix52="$base/prep256_fgmres_r40_t52/wor122_prep256_fgmres_r40_t52"
prefix80="$base/prep256_fgmres_r40_t80/wor122_prep256_fgmres_r40_t80"
prefix110="$base/prep256_fgmres_r40_t110/wor122_prep256_fgmres_r40_t110"
p52="${prefix52}.prim.00020.athdf"
u52="${prefix52}.uov.00020.athdf"
p80="${prefix80}.prim.00023.athdf"
u80="${prefix80}.uov.00023.athdf"
p110="${prefix110}.prim.00026.athdf"
u110="${prefix110}.uov.00026.athdf"

for path in "$p52" "$u52" "$p80" "$u80" "$p110" "$u110"; do
  test -s "$path"
done

"$python" tst/nrfld_validation/wor122_rsg_shell_evolution.py \
  --input "$input" \
  --case t52_to_t80 "$p52" "$u52" "$p80" "$u80" \
  --case t80_to_t110 "$p80" "$u80" "$p110" "$u110" \
  --case t52_to_t110 "$p52" "$u52" "$p110" "$u110" \
  --output "$base/prep256_t110_shell_evolution.json"

for stage in 80 110; do
  prefix_var="prefix${stage}"
  run="$base/prep256_fgmres_r40_t${stage}"
  case "$stage" in
    80) prior_p="$p52"; prior_u="$u52" ;;
    110) prior_p="$p80"; prior_u="$u80" ;;
  esac
  ln -sfn "$prior_p" "$run/a_initial.prim.00000.athdf"
  ln -sfn "$prior_u" "$run/a_initial.uov.00000.athdf"
  "$python" tools/rsg/wor120_convective_flux_audit.py \
    --eos data/eos/rsg_tomida2013_material_eos.tab \
    --case "prep256_fgmres_r40_t${stage}:$run" \
    --json "$run/convective_flux.json" \
    --figure "$run/convective_flux.png"
  "$python" "$base/wor122_analyze_snapshot.py" \
    "${!prefix_var}.prim.$(if test "$stage" = 80; then echo 00023; else echo 00026; fi).athdf" \
    "${!prefix_var}.uov.$(if test "$stage" = 80; then echo 00023; else echo 00026; fi).athdf" \
    > "$run/snapshot_summary.json"
done

"$python" "$base/wor122_axis_flow_monitor.py" \
  --input "$input" \
  --snapshot t52 "$p52" \
  --snapshot t80 "$p80" \
  --snapshot t110 "$p110" \
  --output "$base/prep256_t110_axis_flow.json"
