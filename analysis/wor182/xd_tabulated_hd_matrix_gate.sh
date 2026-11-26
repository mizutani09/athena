#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor182_hd_eos
#SBATCH --output=/work/mizutaniks/wor182/hd_eos_%j.out

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

repository=${WOR182_REPOSITORY:-$HOME/simulation/athena-wor182-lte-radiation-eos}
input=${WOR182_INPUT:-inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated}
rotation=${WOR182_ROTATION:-rotating}
eos_mode=${WOR182_EOS_MODE:-material}
tlim=${WOR182_TLIM:-0.1}
cfl=${WOR182_CFL:-0.2}
nlim=${WOR182_NLIM:--1}
restart_dcycle=${WOR182_RESTART_DCYCLE:-1000}

case "$rotation" in
  rotating)
    profile=data/ic/rsg21_mesa_scf_corecut7p00_q025_corot_nr16384_taper_rho3e-9_gamma5_3_atm1p7e-11_t1e4.csv
    omega=4.0100850320255668e-8
    atmosphere_reference_radius=7.0479108661544016e13
    surface_polar_radius=6.3573075498874664e13
    surface_match_radius=6.4517279030778633e13
    surface_match_polar_radius=5.9356292186235469e13
    rotate_atmosphere=true
    gas_mass=2.8801452052462170e34
    ;;
  nonrotating)
    profile=data/ic/rsg21_mesa_scf_corecut7p00_g5p3_nr16384_taper_rho3e-9_gamma5_3.csv
    omega=0.0
    atmosphere_reference_radius=6.4852294460342625e13
    surface_polar_radius=6.4852294460342625e13
    surface_match_radius=6.0562880106787742e13
    surface_match_polar_radius=6.0562880106787742e13
    rotate_atmosphere=false
    gas_mass=2.8798328945271988e34
    ;;
  *) echo "WOR182_ROTATION must be rotating or nonrotating" >&2; exit 2 ;;
esac

case "$eos_mode" in
  material)
    eos_file=data/eos/rsg_tomida2013_material_eos.tab
    eos_file_type=ascii
    atmosphere_pressure=2.224714707e1
    density_floor=1.0e-8
    ;;
  lte_radiation)
    eos_file=data/eos/rsg_tomida2013_lte_radiation_eos.data
    eos_file_type=binary
    # Material pressure at 1e4 K plus a*T^4/3.
    atmosphere_pressure=4.746614707e1
    # Keep a >=0.5 dex guard above the combined table's rho=1e-12 cgs edge.
    # This is still 5.3 times below the 1.7e-11 cgs atmosphere.
    density_floor=3.2e-4
    ;;
  *) echo "WOR182_EOS_MODE must be material or lte_radiation" >&2; exit 2 ;;
esac

model_tag=${rotation}_${eos_mode}
problem_id=mesa_scf_rsg_corecut7p00_${model_tag}_hd4wz_l4
run=/work/mizutaniks/wor182/${model_tag}_hd_gate_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$input"
test -s "$profile"
test -s "$eos_file"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$input" "$profile" "$eos_file" > "$run/provenance.sha256"
printf 'rotation=%s\neos_mode=%s\nprofile=%s\nomega_rad_s=%s\neos_file=%s\neos_file_type=%s\natmosphere_pressure_dyn_cm2=%s\ndensity_floor_code=%s\ntlim=%s\ncfl=%s\nnlim=%s\nrestart_dcycle=%s\n' \
  "$rotation" "$eos_mode" "$profile" "$omega" "$eos_file" \
  "$eos_file_type" "$atmosphere_pressure" "$density_floor" \
  "$tlim" "$cfl" "$nlim" "$restart_dcycle" > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -i "$input" -d "$run" \
  job/problem_id="$problem_id" \
  time/tlim="$tlim" time/cfl_number="$cfl" time/nlim="$nlim" \
  time/integrator=rk4 time/xorder=4wz time/ncycle_out=1 \
  hydro/eos_file_name="$eos_file" hydro/eos_file_type="$eos_file_type" \
  hydro/eos_include_lte_radiation=false hydro/dfloor="$density_floor" \
  problem/scf_profile_file="$profile" \
  problem/angular_velocity_rad_s="$omega" \
  problem/atmosphere_pressure_dyn_cm2="$atmosphere_pressure" \
  problem/atmosphere_reference_radius_cm="$atmosphere_reference_radius" \
  problem/surface_polar_radius_cm="$surface_polar_radius" \
  problem/surface_match_radius_cm="$surface_match_radius" \
  problem/surface_match_polar_radius_cm="$surface_match_polar_radius" \
  problem/atmosphere_planck_taper_radius_cm="$surface_match_radius" \
  problem/atmosphere_planck_taper_polar_radius_cm="$surface_match_polar_radius" \
  problem/rotate_atmosphere="$rotate_atmosphere" problem/gas_mass_g="$gas_mass" \
  problem/core_luminosity_lsun=0.0 \
  problem/rebuild_radiative_surface=false \
  problem/pointwise_fld_surface_cells=false \
  output1/dcycle=1 output2/dcycle="$restart_dcycle" \
  output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

if grep -Eq 'FATAL ERROR|^scf_rsg:' "$run/stdout.log" "$run/stderr.log"; then
  status=1
fi
grep -E '^cycle=|FATAL ERROR|^scf_rsg:|scf_rsg profile=' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
tail -n 3 "$run"/*.hst > "$run/history_tail.log" 2>/dev/null || true
exit "$status"
