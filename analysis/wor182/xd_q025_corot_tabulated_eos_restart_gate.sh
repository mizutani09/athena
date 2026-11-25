#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor182_tab_cont
#SBATCH --output=/work/mizutaniks/wor182/tabulated_eos_continue_%j.out

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
ulimit -c 0

repository=${WOR182_REPOSITORY:-$HOME/simulation/athena-wor182-tabulated-eos-simd}
restart=${WOR182_RESTART:-/work/mizutaniks/wor182/q025_corot_tabulated_eos_347871/mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated.final.rst}
input=${WOR182_INPUT:-inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated}
target_time=${WOR182_TARGET_TIME:-0.1}
cfl=${WOR182_CFL:-0.2}
nlim=${WOR182_NLIM:--1}
integrator=${WOR182_INTEGRATOR:-rk4}
restart_dcycle=${WOR182_RESTART_DCYCLE:-100}
slice_output_dt=${WOR182_SLICE_OUTPUT_DT:--1}
fgmres_threshold=${WOR182_FGMRES_THRESHOLD:-1.0e-4}
diagnostic_verbosity=${WOR182_DIAGNOSTIC_VERBOSITY:-0}
surface_mach_diagnostic_threshold=${WOR182_SURFACE_MACH_DIAGNOSTIC_THRESHOLD:--1.0}
nr_backtrack_max=${WOR182_NR_BACKTRACK_MAX:-4}
nr_min_step_scale=${WOR182_NR_MIN_STEP_SCALE:-0.05}
mixed_frame_terms=${WOR182_MIXED_FRAME_TERMS:-true}
mixed_frame_transport=${WOR182_MIXED_FRAME_TRANSPORT:-true}
jacobian_check=${WOR182_JACOBIAN_CHECK:-false}
jacobian_check_epsilon=${WOR182_JACOBIAN_CHECK_EPSILON:-1.0e-4}
cfl_tag=${cfl//./p}
time_tag=${target_time//./p}
run=/work/mizutaniks/wor182/q025_corot_tabulated_eos_cfl${cfl_tag}_t${time_tag}_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$restart"
test -s "$input"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$restart" "$input" \
  data/eos/rsg_tomida2013_material_eos.tab \
  data/opacity/rsg_gray_opacity_t7.h5 > "$run/provenance.sha256"
printf 'restart=%s\ninput=%s\ntarget_time=%s\ncfl=%s\nnlim=%s\nintegrator=%s\nrestart_dcycle=%s\nslice_output_dt=%s\nfgmres_threshold=%s\ndiagnostic_verbosity=%s\nsurface_mach_diagnostic_threshold=%s\nnr_backtrack_max=%s\nnr_min_step_scale=%s\nmixed_frame_terms=%s\nmixed_frame_transport=%s\njacobian_check=%s\njacobian_check_epsilon=%s\n' \
  "$restart" "$input" "$target_time" "$cfl" "$nlim" "$integrator" \
  "$restart_dcycle" "$slice_output_dt" "$fgmres_threshold" \
  "$diagnostic_verbosity" "$surface_mach_diagnostic_threshold" \
  "$nr_backtrack_max" "$nr_min_step_scale" \
  "$mixed_frame_terms" "$mixed_frame_transport" \
  "$jacobian_check" "$jacobian_check_epsilon" \
  > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -i "$input" -d "$run" \
  time/integrator="$integrator" time/xorder=4wz \
  time/cfl_number="$cfl" time/nlim="$nlim" time/tlim="$target_time" \
  time/ncycle_out=1 \
  fld/include_mixed_frame_terms="$mixed_frame_terms" \
  fld/mixed_frame_transport="$mixed_frame_transport" \
  nrfld/linear_solver=fgmres nrfld/fgmres_threshold="$fgmres_threshold" \
  nrfld/diagnostic_verbosity="$diagnostic_verbosity" \
  nrfld/nr_backtrack_max="$nr_backtrack_max" \
  nrfld/nr_min_step_scale="$nr_min_step_scale" \
  nrfld/jacobian_check="$jacobian_check" \
  nrfld/jacobian_check_epsilon="$jacobian_check_epsilon" \
  problem/surface_mach_diagnostic_threshold="$surface_mach_diagnostic_threshold" \
  output1/dcycle=1 output2/dcycle="$restart_dcycle" \
  output3/dt="$slice_output_dt" output4/dt="$slice_output_dt" \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

grep -E '^cycle=|\[NR\] status=|\[LINEAR_FGMRES\]|FATAL ERROR' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
exit "$status"
