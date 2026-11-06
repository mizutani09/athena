#!/usr/bin/env bash
# Same-checkpoint A/B; export WOR102_MODE=baseline|candidate and WOR102_STEPS=1|10.
# Set WOR102_RANK_RSS=1 and WOR102_RUN_SUFFIX=_rss for per-rank peak RSS.
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mizutaniks/wor102/slurm/%x_%j.log

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

: "${WOR102_MODE:?baseline or candidate}"
: "${WOR102_STEPS:?1 or 10}"
case "$WOR102_MODE" in baseline|candidate) ;; *) exit 2 ;; esac
case "$WOR102_STEPS" in 1|10) ;; *) exit 2 ;; esac
test "$SLURM_NTASKS" -eq 32
root=/work/mizutaniks/wor102
repo=$root/athena-mach5
binary=$repo/bin/athena
restart=/work/mizutaniks/wor67/output/wor67_hpc_amr_base1024/cases/mach5/output/radshock_mach5_amr.final.rst
source_input=/work/mizutaniks/wor67/output/wor67_hpc_amr_base1024_late_interface1/cases/mach5/input.used
test -x "$binary"
test "$(sha256sum "$restart" | cut -d' ' -f1)" = c7f1848f6613c76f3d6a053fa45244fdd937b9610021fb066c265f7ee9753816
test "$(sha256sum "$source_input" | cut -d' ' -f1)" = ec6ba18778419808172a2b0ec3581caf3117190623cfe31c50eb99f92b33c16d
case_dir=$root/output/${WOR102_MODE}_${WOR102_STEPS}${WOR102_RUN_SUFFIX:-}/cases/mach5
test ! -e "$case_dir"
mkdir -p "$case_dir/output"
cp "$source_input" "$case_dir/input.used"
grep -Eq '^linear_solver = fgmres$' "$case_dir/input.used"
{
  printf 'job_id=%s\nmode=%s\nsteps=%s\nranks=%s\n' \
    "$SLURM_JOB_ID" "$WOR102_MODE" "$WOR102_STEPS" "$SLURM_NTASKS"
  printf 'binary_source_commit=%s\n' "$(sed -n 's/^source_commit=//p' "$root/build/provenance.txt")"
  printf 'run_script_commit=%s\n' "$(git -C "$repo" rev-parse HEAD)"
  printf 'exe_sha256=%s\nrestart_sha256=%s\ninput_sha256=%s\n' \
    "$(sha256sum "$binary" | cut -d' ' -f1)" \
    "$(sha256sum "$restart" | cut -d' ' -f1)" \
    "$(sha256sum "$case_dir/input.used" | cut -d' ' -f1)"
} > "$case_dir/provenance.txt"
export ATHENA_NRFLD_ENERGY_LEDGER=1 ATHENA_NRFLD_INTERFACE_LEDGER=1
if [[ "$WOR102_MODE" = candidate ]]; then
  export ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1
  export ATHENA_NRFLD_CONSERVATIVE_SHARED_COEFF=1
  export ATHENA_NRFLD_CONSERVATIVE_DIFFERENCE_JV=0
  export ATHENA_NRFLD_EXPERIMENTAL_TANGENTIAL_RECONSTRUCTION=1
  export ATHENA_NRFLD_EXPERIMENTAL_COUPLED=1
  export ATHENA_NRFLD_EXPERIMENTAL_REGRID=1
  export ATHENA_NRFLD_EXPERIMENTAL_MACH5=1
fi
cd "$case_dir"
arguments=("$binary" -r "$restart" -i input.used -d output
  'time/tlim=0.041' "time/nlim=$((36517+WOR102_STEPS))"
  'output1/dt=-1' 'output2/dt=-1' 'output3/dt=1e-12')
if [[ "${WOR102_RANK_RSS:-0}" = 1 ]]; then
  export WOR102_CASE_DIR="$case_dir"
  /usr/bin/time -v mpirun -np "$SLURM_NTASKS" -ppn 32 \
    -outfile-pattern "$case_dir/rank.%r.log" \
    bash -c 'rank=${PMI_RANK:-${PMIX_RANK:-unknown}}; exec /usr/bin/time -o "$WOR102_CASE_DIR/rss.$rank.$$.txt" -f "max_rss_kb=%M" "$@"' \
    bash "${arguments[@]}" >launcher.log 2>&1
else
  /usr/bin/time -v mpirun -np "$SLURM_NTASKS" -ppn 32 \
    -outfile-pattern "$case_dir/rank.%r.log" \
    "${arguments[@]}" >launcher.log 2>&1
fi
test "$(find "$case_dir" -maxdepth 1 -name 'rank.*.log' | wc -l)" -eq 32
if [[ "${WOR102_RANK_RSS:-0}" = 1 ]]; then
  test "$(find "$case_dir" -maxdepth 1 -name 'rss.*.txt' | wc -l)" -eq 32
fi
