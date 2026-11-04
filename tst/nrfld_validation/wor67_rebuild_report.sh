#!/usr/bin/env bash
# Rebuild the completed WOR-67 local comparison from saved simulation output.
set -euo pipefail

output_root=${1:-/mnt/d/output/rad}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python3 "$script_dir/wor67_spike_audit.py" \
  --case "N64=$output_root/wor10_mach5_n64_20261104/cases/mach5" \
  --case "N128=$output_root/wor10_mach5_n128_20261104/cases/mach5" \
  --case "N256=$output_root/wor10_mach5_n256_20261104/cases/mach5" \
  --case "AMR1_N256=$output_root/validation_radiative_shock_amr1_20261102/cases/mach5" \
  --case "N512=$output_root/wor10_n512_20261104" \
  --case "N512_dt_half=$output_root/wor67_uniform_n512_cfl015_20261104" \
  --case "N1024=$output_root/wor67_uniform_n1024_cfl030_20261104" \
  --case "N1024_dt_half=$output_root/wor67_uniform_n1024_cfl015_20261104" \
  --case "AMR1_N1024=$output_root/wor67_amr_base512_20261104/cases/mach5" \
  --case "N2048=$output_root/wor67_uniform_n2048_cfl030_20261104" \
  --output "$output_root/wor67_spike_audit_20261104"
