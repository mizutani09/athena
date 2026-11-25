#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=00:30:00
#SBATCH --job-name=wor182_q025_fld
#SBATCH --output=/work/mizutaniks/wor182/q025_corot_fld_%j.out

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
profile=${WOR182_PROFILE:-data/ic/rsg21_mesa_scf_corecut7p00_q025_corot_nr16384_taper_rho3e-9_gamma5_3.csv}
atmosphere_density=${WOR182_ATMOSPHERE_DENSITY:-}
rotate_atmosphere=${WOR182_ROTATE_ATMOSPHERE:-}
nlim=${WOR182_NLIM:-110}
tlim=${WOR182_TLIM:-1.0e-2}
integrator=${WOR182_INTEGRATOR:-ssprk5_4}
slice_output_dt=${WOR182_SLICE_OUTPUT_DT:--1}
case ${CORE_LEVEL:-l4} in
  l3)
    input=inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l3
    ;;
  l4)
    input=inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_q025_corot_fld4wz_l4
    ;;
  *) echo "CORE_LEVEL must be l3 or l4" >&2; exit 2 ;;
esac
run=/work/mizutaniks/wor182/q025_corot_fld_${CORE_LEVEL:-l4}_${SLURM_JOB_ID}

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$input"
test -s "$profile"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$input" "$profile" > "$run/provenance.sha256"
problem_arguments=(problem/scf_profile_file="$profile")
if [[ -n "$atmosphere_density" ]]; then
  problem_arguments+=(
    problem/atmosphere_density_g_cm3="$atmosphere_density")
fi
if [[ -n "$rotate_atmosphere" ]]; then
  problem_arguments+=(problem/rotate_atmosphere="$rotate_atmosphere")
fi
printf 'core_level=%s\nintegrator=%s\nnlim=%s\ntlim=%s\nprofile=%s\natmosphere_density=%s\nrotate_atmosphere=%s\nslice_output_dt=%s\n' \
  "${CORE_LEVEL:-l4}" "$integrator" "$nlim" "$tlim" "$profile" \
  "$atmosphere_density" "$rotate_atmosphere" "$slice_output_dt" \
  > "$run/run_config.txt"

set +e
/usr/bin/time -f 'wall_s=%e maxrss_kb=%M' -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -i "$input" -d "$run" \
  time/integrator="$integrator" time/nlim="$nlim" time/tlim="$tlim" \
  time/ncycle_out=1 \
  output1/dcycle=1 output2/dcycle=1000 \
  output3/dt="$slice_output_dt" output4/dt="$slice_output_dt" \
  "${problem_arguments[@]}" \
  > "$run/stdout.log" 2> "$run/stderr.log"
status=$?
set -e

grep -E '^cycle=|NR.*status=|FATAL ERROR|Er_old=|source_delta=|outgoing_delta=' \
  "$run/stdout.log" "$run/stderr.log" > "$run/summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 100 "$run/summary.log"
exit "$status"
