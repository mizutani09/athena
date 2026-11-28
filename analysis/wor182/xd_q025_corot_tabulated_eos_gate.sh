#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor182_tab_eos
#SBATCH --output=/work/mizutaniks/wor182/tabulated_eos_%j.out

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

repository=${WOR182_REPOSITORY:-$HOME/simulation/athena-wor182-tabulated-eos}
input=${WOR182_INPUT:-inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated}
model_tag=${WOR182_MODEL_TAG:-q025_corot}
profile_file=${WOR182_PROFILE_FILE:-data/ic/rsg21_mesa_scf_corecut7p00_q025_corot_nr16384_taper_rho3e-9_gamma5_3_atm1p7e-11_t1e4.csv}
angular_velocity=${WOR182_ANGULAR_VELOCITY:-4.0100850320255668e-8}
atmosphere_density=${WOR182_ATMOSPHERE_DENSITY:-1.7e-11}
atmosphere_pressure=${WOR182_ATMOSPHERE_PRESSURE:-2.224714707e1}
atmosphere_reference_radius=${WOR182_ATMOSPHERE_REFERENCE_RADIUS:-7.0479108661544016e13}
surface_polar_radius=${WOR182_SURFACE_POLAR_RADIUS:-6.3573075498874664e13}
surface_match_radius=${WOR182_SURFACE_MATCH_RADIUS:-6.4517279030778633e13}
surface_match_polar_radius=${WOR182_SURFACE_MATCH_POLAR_RADIUS:-5.9356292186235469e13}
rotate_atmosphere=${WOR182_ROTATE_ATMOSPHERE:-true}
gas_mass=${WOR182_GAS_MASS:-2.8801452052462170e34}
planck_taper_radius=${WOR182_PLANCK_TAPER_RADIUS:-6.4517279030778633e13}
planck_taper_polar_radius=${WOR182_PLANCK_TAPER_POLAR_RADIUS:-5.9356292186235469e13}
nlim=${WOR182_NLIM:-0}
tlim=${WOR182_TLIM:-1.0e-3}
integrator=${WOR182_INTEGRATOR:-rk4}
cfl=${WOR182_CFL:-0.1}
core_luminosity_lsun=${WOR182_CORE_LUMINOSITY_LSUN:-1.1631598e5}
luminosity_heating_radius_cm=${WOR182_LUMINOSITY_HEATING_RADIUS_CM:-4.5207406562668779e12}
rebuild_radiative_surface=${WOR182_REBUILD_RADIATIVE_SURFACE:-false}
radiative_surface_match_density=${WOR182_RADIATIVE_SURFACE_MATCH_DENSITY:-1.0e-9}
radiative_surface_superadiabatic_peak=${WOR182_RADIATIVE_SURFACE_SUPERADIABATIC_PEAK:-0.0}
radiative_surface_superadiabatic_width=${WOR182_RADIATIVE_SURFACE_SUPERADIABATIC_WIDTH:-0.1}
pointwise_fld_surface_cells=${WOR182_POINTWISE_FLD_SURFACE_CELLS:-false}
surface_mach_diagnostic_threshold=${WOR182_SURFACE_MACH_DIAGNOSTIC_THRESHOLD:--1.0}
relaxation_velocity_tau=${WOR182_RELAXATION_VELOCITY_TAU:--1.0}
relaxation_taper_start=${WOR182_RELAXATION_TAPER_START:--1.0}
relaxation_taper_end=${WOR182_RELAXATION_TAPER_END:--1.0}
tasks_per_node=${WOR182_TASKS_PER_NODE:-56}
run=/work/mizutaniks/wor182/${model_tag}_tabulated_eos_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$input"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$input" data/eos/rsg_tomida2013_material_eos.tab \
  data/opacity/rsg_gray_opacity_t7.h5 > "$run/provenance.sha256"
printf 'input=%s\nmodel_tag=%s\nprofile_file=%s\nangular_velocity_rad_s=%s\natmosphere_density_g_cm3=%s\natmosphere_pressure_dyn_cm2=%s\natmosphere_reference_radius_cm=%s\nsurface_polar_radius_cm=%s\nsurface_match_radius_cm=%s\nsurface_match_polar_radius_cm=%s\nrotate_atmosphere=%s\ngas_mass_g=%s\nplanck_taper_radius_cm=%s\nplanck_taper_polar_radius_cm=%s\nintegrator=%s\nnlim=%s\ntlim=%s\ncfl=%s\ncore_luminosity_lsun=%s\nluminosity_heating_radius_cm=%s\nrebuild_radiative_surface=%s\nradiative_surface_match_density_g_cm3=%s\nradiative_surface_superadiabatic_peak=%s\nradiative_surface_superadiabatic_width_fraction=%s\npointwise_fld_surface_cells=%s\nsurface_mach_diagnostic_threshold=%s\nrelaxation_velocity_tau=%s\nrelaxation_taper_start=%s\nrelaxation_taper_end=%s\ntasks_per_node=%s\n' \
  "$input" "$model_tag" "$profile_file" "$angular_velocity" \
  "$atmosphere_density" "$atmosphere_pressure" \
  "$atmosphere_reference_radius" "$surface_polar_radius" \
  "$surface_match_radius" "$surface_match_polar_radius" \
  "$rotate_atmosphere" "$gas_mass" "$planck_taper_radius" \
  "$planck_taper_polar_radius" \
  "$integrator" "$nlim" "$tlim" "$cfl" \
  "$core_luminosity_lsun" "$luminosity_heating_radius_cm" \
  "$rebuild_radiative_surface" \
  "$radiative_surface_match_density" \
  "$radiative_surface_superadiabatic_peak" \
  "$radiative_surface_superadiabatic_width" \
  "$pointwise_fld_surface_cells" \
  "$surface_mach_diagnostic_threshold" "$relaxation_velocity_tau" \
  "$relaxation_taper_start" "$relaxation_taper_end" "$tasks_per_node" \
  > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node="$tasks_per_node" \
  bin/athena -i "$input" -d "$run" \
  time/integrator="$integrator" time/nlim="$nlim" time/tlim="$tlim" \
  time/cfl_number="$cfl" time/ncycle_out=1 \
  problem/scf_profile_file="$profile_file" \
  problem/angular_velocity_rad_s="$angular_velocity" \
  problem/atmosphere_density_g_cm3="$atmosphere_density" \
  problem/atmosphere_pressure_dyn_cm2="$atmosphere_pressure" \
  problem/atmosphere_reference_radius_cm="$atmosphere_reference_radius" \
  problem/surface_polar_radius_cm="$surface_polar_radius" \
  problem/surface_match_radius_cm="$surface_match_radius" \
  problem/surface_match_polar_radius_cm="$surface_match_polar_radius" \
  problem/rotate_atmosphere="$rotate_atmosphere" \
  problem/gas_mass_g="$gas_mass" \
  problem/atmosphere_planck_taper_radius_cm="$planck_taper_radius" \
  problem/atmosphere_planck_taper_polar_radius_cm="$planck_taper_polar_radius" \
  problem/core_luminosity_lsun="$core_luminosity_lsun" \
  problem/luminosity_heating_radius_cm="$luminosity_heating_radius_cm" \
  problem/rebuild_radiative_surface="$rebuild_radiative_surface" \
  problem/radiative_surface_match_density_g_cm3="$radiative_surface_match_density" \
  problem/radiative_surface_superadiabatic_peak="$radiative_surface_superadiabatic_peak" \
  problem/radiative_surface_superadiabatic_width_fraction="$radiative_surface_superadiabatic_width" \
  problem/pointwise_fld_surface_cells="$pointwise_fld_surface_cells" \
  problem/surface_mach_diagnostic_threshold="$surface_mach_diagnostic_threshold" \
  problem/relaxation_velocity_tau="$relaxation_velocity_tau" \
  problem/relaxation_taper_start="$relaxation_taper_start" \
  problem/relaxation_taper_end="$relaxation_taper_end" \
  output1/dcycle=1 output2/dcycle=1000 \
  output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

if grep -Eq 'FATAL ERROR|^scf_rsg:' "$run/stdout.log" "$run/stderr.log"; then
  status=1
fi

grep -E '^cycle=|NR.*status=|FATAL ERROR|^scf_rsg:|EOS|scf_rsg profile=' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
tail -n 3 "$run"/*.hst > "$run/luminosity_history_tail.log" 2>/dev/null || true
exit "$status"
