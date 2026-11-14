#!/usr/bin/env bash
# One-cycle WOR-67 late-state energy/interface audit with separate rank logs.
# Export WOR67_BINARY, WOR67_RESTART, WOR67_INPUT, WOR67_CYCLE,
# WOR67_RUN_NAME, WOR67_PPN and ATHENA_NRFLD_{ENERGY,INTERFACE}_LEDGER=1.
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mizutaniks/wor67/slurm/%x_%j.log

set -euo pipefail
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

: "${WOR67_BINARY:?}"
: "${WOR67_RESTART:?}"
: "${WOR67_INPUT:?}"
: "${WOR67_CYCLE:?}"
: "${WOR67_RUN_NAME:?}"
: "${WOR67_PPN:?}"
: "${ATHENA_NRFLD_ENERGY_LEDGER:?}"
: "${ATHENA_NRFLD_INTERFACE_LEDGER:?}"
WOR67_STEPS=${WOR67_STEPS:-1}
test "$WOR67_STEPS" -ge 1
test -x "$WOR67_BINARY"
test -f "$WOR67_RESTART"
test -f "$WOR67_INPUT"
test "$SLURM_NTASKS" -eq "$((SLURM_JOB_NUM_NODES * WOR67_PPN))"
WOR67_OUTPUT_ROOT=${WOR67_OUTPUT_ROOT:-/work/mizutaniks/wor67/output}
case_dir=$WOR67_OUTPUT_ROOT/$WOR67_RUN_NAME/cases/mach5
mkdir -p "$case_dir/output"
cp "$WOR67_INPUT" "$case_dir/input.used"
{
  printf 'job_id=%s\nnodes=%s\nranks=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS"
  printf 'start_cycle=%s\nsteps=%s\n' "$WOR67_CYCLE" "$WOR67_STEPS"
  printf 'exe_sha256=%s\nrestart_sha256=%s\ninput_sha256=%s\n' \
    "$(sha256sum "$WOR67_BINARY" | cut -d' ' -f1)" \
    "$(sha256sum "$WOR67_RESTART" | cut -d' ' -f1)" \
    "$(sha256sum "$case_dir/input.used" | cut -d' ' -f1)"
} > "$case_dir/provenance.txt"
cd "$case_dir"
mpirun -np "$SLURM_NTASKS" -ppn "$WOR67_PPN" \
  -outfile-pattern "$case_dir/rank.%r.log" \
  "$WOR67_BINARY" -r "$WOR67_RESTART" -i input.used -d output \
  'time/tlim=0.041' "time/nlim=$((WOR67_CYCLE+WOR67_STEPS))" \
  'output1/dt=-1' 'output2/dt=-1' 'output3/dt=1e-12' \
  >launcher.log 2>&1
test "$(find "$case_dir" -maxdepth 1 -name 'rank.*.log' | wc -l)" -eq "$SLURM_NTASKS"
