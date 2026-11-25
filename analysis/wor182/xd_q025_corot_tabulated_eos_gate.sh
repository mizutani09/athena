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
input=inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4_tabulated
nlim=${WOR182_NLIM:-0}
tlim=${WOR182_TLIM:-1.0e-3}
integrator=${WOR182_INTEGRATOR:-rk4}
cfl=${WOR182_CFL:-0.1}
surface_mach_diagnostic_threshold=${WOR182_SURFACE_MACH_DIAGNOSTIC_THRESHOLD:--1.0}
relaxation_velocity_tau=${WOR182_RELAXATION_VELOCITY_TAU:--1.0}
relaxation_taper_start=${WOR182_RELAXATION_TAPER_START:--1.0}
relaxation_taper_end=${WOR182_RELAXATION_TAPER_END:--1.0}
run=/work/mizutaniks/wor182/q025_corot_tabulated_eos_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$input"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$input" data/eos/rsg_tomida2013_material_eos.tab \
  data/opacity/rsg_gray_opacity_t7.h5 > "$run/provenance.sha256"
printf 'integrator=%s\nnlim=%s\ntlim=%s\ncfl=%s\nsurface_mach_diagnostic_threshold=%s\nrelaxation_velocity_tau=%s\nrelaxation_taper_start=%s\nrelaxation_taper_end=%s\n' \
  "$integrator" "$nlim" "$tlim" "$cfl" \
  "$surface_mach_diagnostic_threshold" "$relaxation_velocity_tau" \
  "$relaxation_taper_start" "$relaxation_taper_end" \
  > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -i "$input" -d "$run" \
  time/integrator="$integrator" time/nlim="$nlim" time/tlim="$tlim" \
  time/cfl_number="$cfl" time/ncycle_out=1 \
  problem/surface_mach_diagnostic_threshold="$surface_mach_diagnostic_threshold" \
  problem/relaxation_velocity_tau="$relaxation_velocity_tau" \
  problem/relaxation_taper_start="$relaxation_taper_start" \
  problem/relaxation_taper_end="$relaxation_taper_end" \
  output1/dcycle=1 output2/dcycle=1000 \
  output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

grep -E '^cycle=|NR.*status=|FATAL ERROR|EOS|scf_rsg profile=' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
exit "$status"
