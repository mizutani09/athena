#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor182_q025_cont
#SBATCH --output=/work/mizutaniks/wor182/q025_corot_fld_continue_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5
export SLURM_MPI_TYPE=pmi2
export LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
export ATHENA_FLD_EXPLICIT_POSITIVITY_DIAGNOSTICS=1
ulimit -c 0

repository=${WOR182_REPOSITORY:-$HOME/simulation/athena-wor182-rotating-fld}
restart=${WOR182_RESTART:-/work/mizutaniks/wor182/q025_corot_fld_l4_347695/mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4.final.rst}
input=${WOR182_INPUT:-}
surface_mach_threshold=${WOR182_SURFACE_MACH_THRESHOLD:-}
target_time=${WOR182_TARGET_TIME:-1.0}
cfl=${WOR182_CFL:-0.1}
nlim=${WOR182_NLIM:--1}
integrator=${WOR182_INTEGRATOR:-ssprk5_4}
restart_dcycle=${WOR182_RESTART_DCYCLE:-100}
diagnostic_verbosity=${WOR182_DIAGNOSTIC_VERBOSITY:-0}
radiation_force_floor_factor=${WOR182_RADIATION_FORCE_DENSITY_FLOOR_FACTOR:-}
slice_output_dt=${WOR182_SLICE_OUTPUT_DT:--1}
cfl_tag=${cfl//./p}
run=/work/mizutaniks/wor182/q025_corot_fld_l4_cfl${cfl_tag}_t${target_time}_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$restart"
input_arguments=()
mach_arguments=()
force_mask_arguments=()
if [[ -n "$input" ]]; then
  test -s "$input"
  input_arguments=(-i "$input")
fi
if [[ -n "$surface_mach_threshold" ]]; then
  if [[ -z "$input" ]]; then
    echo "WOR182_INPUT is required with WOR182_SURFACE_MACH_THRESHOLD" >&2
    exit 2
  fi
  mach_arguments=(problem/surface_mach_diagnostic_threshold="$surface_mach_threshold")
fi
if [[ -n "$radiation_force_floor_factor" ]]; then
  force_mask_arguments=(
    fld/radiation_force_density_floor_factor="$radiation_force_floor_factor")
fi
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
printf 'restart=%s\ninput=%s\nsurface_mach_threshold=%s\nradiation_force_density_floor_factor=%s\nslice_output_dt=%s\ntarget_time=%s\ncfl=%s\nnlim=%s\nintegrator=%s\nrestart_dcycle=%s\ndiagnostic_verbosity=%s\n' \
  "$restart" "$input" "$surface_mach_threshold" \
  "$radiation_force_floor_factor" "$slice_output_dt" \
  "$target_time" "$cfl" "$nlim" "$integrator" \
  "$restart_dcycle" "$diagnostic_verbosity" > "$run/run_config.txt"
sha256sum bin/athena "$restart" > "$run/provenance.sha256"

# Continue from the accepted L4 startup state. No explicit dt cap is imposed;
# Athena's ordinary growth limiter approaches the requested hydro CFL.
set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" "${input_arguments[@]}" -d "$run" \
  time/integrator="$integrator" time/xorder=4wz \
  time/cfl_number="$cfl" time/nlim="$nlim" time/tlim="$target_time" \
  time/ncycle_out=1 \
  fld/include_mixed_frame_terms=true fld/mixed_frame_transport=true \
  nrfld/linear_solver=fgmres nrfld/diagnostic_verbosity="$diagnostic_verbosity" \
  output1/dcycle=1 output2/dcycle="$restart_dcycle" \
  output3/dt="$slice_output_dt" output4/dt="$slice_output_dt" \
  "${mach_arguments[@]}" \
  "${force_mask_arguments[@]}" \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

grep -E '^cycle=|\[NR\] status=|FATAL ERROR|Er_old=|source_delta=|outgoing_delta=' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
exit "$status"
