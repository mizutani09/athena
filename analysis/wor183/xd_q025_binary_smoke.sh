#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor183_q025_binary
#SBATCH --output=/work/mizutaniks/wor183/q025_binary_%j.out

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

repository=${WOR183_REPOSITORY:-$HOME/simulation/athena-wor182-binary-smoke}
restart=${WOR183_RESTART:-/work/mizutaniks/wor182/q025_corot_tabulated_eos_cfl0p3_t10p0_347916/mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated.final.rst}
input=${WOR183_INPUT:-inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated}
target_time=${WOR183_TARGET_TIME:-10.25}
companion_start=${WOR183_COMPANION_START:-10.0}
companion_ramp_time=${WOR183_COMPANION_RAMP_TIME:-1.0}
companion_mass_ratio=${WOR183_COMPANION_MASS_RATIO:-0.25}
companion_separation_cm=${WOR183_COMPANION_SEPARATION_CM:-1.2939385270211592e14}
companion_softening_cm=${WOR183_COMPANION_SOFTENING_CM:-3.0e12}
companion_omega=${WOR183_COMPANION_OMEGA:-4.0100850320255668e-8}
run=/work/mizutaniks/wor183/q025_binary_smoke_${SLURM_JOB_ID}

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
printf 'restart=%s\ntarget_time=%s\ncompanion_start=%s\ncompanion_ramp_time=%s\ncompanion_mass_ratio=%s\ncompanion_separation_cm=%s\ncompanion_softening_cm=%s\ncompanion_omega_rad_s=%s\n' \
  "$restart" "$target_time" "$companion_start" "$companion_ramp_time" \
  "$companion_mass_ratio" "$companion_separation_cm" \
  "$companion_softening_cm" "$companion_omega" > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -r "$restart" -i "$input" -d "$run" \
  time/integrator=rk4 time/xorder=4wz time/cfl_number=0.3 \
  time/nlim=-1 time/tlim="$target_time" time/ncycle_out=1 \
  fld/include_mixed_frame_terms=true fld/mixed_frame_transport=true \
  nrfld/linear_solver=fgmres nrfld/fgmres_threshold=1.0e-4 \
  nrfld/diagnostic_verbosity=0 \
  problem/surface_refinement_level=2 \
  problem/surface_refinement_density_g_cm3=3.0e-9 \
  problem/surface_mach_diagnostic_threshold=-1 \
  problem/companion_enabled=true \
  problem/companion_mass_ratio="$companion_mass_ratio" \
  problem/companion_separation_cm="$companion_separation_cm" \
  problem/companion_softening_cm="$companion_softening_cm" \
  problem/companion_start_time="$companion_start" \
  problem/companion_ramp_time="$companion_ramp_time" \
  problem/companion_angular_velocity_rad_s="$companion_omega" \
  problem/companion_phase=0.0 \
  problem/companion_indirect_acceleration=true \
  problem/companion_gravity_cfl=0.3 \
  output1/dcycle=1 output2/dcycle=1000 output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
grep -E '^cycle=|\[NR\] status=|\[LINEAR_FGMRES\]|FATAL ERROR' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
exit "$status"
