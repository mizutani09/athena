#!/usr/bin/env bash
#SBATCH --job-name=wor105_default
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mizutaniks/wor105/slurm/%x_%j.log

set -euo pipefail
source /etc/profile.d/modules.sh
source /work/opt/local/bin/enable-oneapi.sh
module swap PrgEnv-oneapi/2024.2 PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5
export PATH="${I_MPI_ROOT}/bin:${PATH}"
for var in CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
  IFS=: read -r -a entries <<< "${!var:-}"
  filtered=()
  for entry in "${entries[@]}"; do
    case "$entry" in
      /work/opt/local/miniconda3/py312_24.9.2/*) ;;
      *) filtered+=("$entry") ;;
    esac
  done
  export "$var=$(IFS=:; printf '%s' "${filtered[*]}")"
done
export I_MPI_HYDRA_BOOTSTRAP=slurm
root=/work/mizutaniks/wor105
binary=$root/athena-mach5/bin/athena
restart=/work/mizutaniks/wor67/output/wor67_hpc_amr_base1024/cases/mach5/output/radshock_mach5_amr.final.rst
source_input=/work/mizutaniks/wor67/output/wor67_hpc_amr_base1024_late_interface1/cases/mach5/input.used
test "$(sha256sum "$restart" | cut -d' ' -f1)" = c7f1848f6613c76f3d6a053fa45244fdd937b9610021fb066c265f7ee9753816
test "$(sha256sum "$source_input" | cut -d' ' -f1)" = ec6ba18778419808172a2b0ec3581caf3117190623cfe31c50eb99f92b33c16d
test "$(sha256sum "$binary" | cut -d' ' -f1)" = "$(sed -n 's/^binary_sha256=//p' "$root/build/provenance.txt")"
case_dir=$root/output/default
test ! -e "$case_dir"
mkdir -p "$case_dir/output"
cp "$source_input" "$case_dir/input.used"
python3 - "$case_dir/input.used" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()
s=s.replace('dt = 0.002', 'dt = -1')
s=s.replace('<output4>', 'dcycle = 36617\n\n<output4>')
p.write_text(s)
PY
{
  printf 'job_id=%s\nmode=default\nranks=%s\n' "$SLURM_JOB_ID" "$SLURM_NTASKS"
  printf 'source_commit=%s\n' "$(git -C "$root/athena-mach5" rev-parse HEAD)"
  printf 'binary_sha256=%s\nrestart_sha256=%s\ninput_sha256=%s\n' \
    "$(sha256sum "$binary" | cut -d' ' -f1)" \
    "$(sha256sum "$restart" | cut -d' ' -f1)" \
    "$(sha256sum "$case_dir/input.used" | cut -d' ' -f1)"
} > "$case_dir/provenance.txt"
export ATHENA_NRFLD_ENERGY_LEDGER=1 ATHENA_NRFLD_INTERFACE_LEDGER=1
unset ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF
unset ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV
unset ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION
unset ATHENA_NRFLD_EXPERIMENTAL_COUPLED ATHENA_NRFLD_EXPERIMENTAL_REGRID
unset ATHENA_NRFLD_EXPERIMENTAL_MACH5 ATHENA_NRFLD_LEGACY_INTERFACE
cd "$case_dir"
/usr/bin/time -v mpirun -np "$SLURM_NTASKS" -ppn 32 \
  -outfile-pattern "$case_dir/rank.%r.log" \
  "$binary" -r "$restart" -i input.used -d output \
  'time/tlim=0.041' 'time/nlim=36617' >launcher.log 2>&1
test "$(find "$case_dir" -maxdepth 1 -name 'rank.*.log' | wc -l)" -eq 32
