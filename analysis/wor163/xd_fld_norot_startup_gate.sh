#!/bin/bash
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=3
#SBATCH --ntasks=168
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=01:00:00
#SBATCH --job-name=wor163_fld0_start
#SBATCH --output=/work/mizutaniks/wor163/fld_norot_startup_%j.out

set -euo pipefail
source /etc/profile.d/modules.sh
export MODULEPATH=/work/opt/intel/modulefiles:/work/opt/local_oneapi/modulefiles:/work/opt/local/modulefiles
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
export SLURM_MPI_TYPE=pmi2
export LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=1
ulimit -c 0

repository=$HOME/simulation/athena-rsg-corecut7p00-fld
input=inputs/rsg/athinput.mesa_scf_rsg_corecut7p00_fld_norot_l3
profile=data/ic/rsg21_mesa_scf_corecut7p00_g5p3_nr16384_taper_rho3e-9_gamma5_3.csv
run=/work/mizutaniks/wor163/fld_norot_startup_${SLURM_JOB_ID}
problem_id=mesa_scf_rsg_corecut7p00_fld_norot_l3

mkdir -p "$run"
cd "$repository"
test -x bin/athena
test -s "$profile"
git rev-parse HEAD > "$run/git_commit.txt"
git status --short > "$run/git_status.txt"
sha256sum bin/athena "$input" "$profile" data/opacity/rsg_gray_opacity_t7.h5 \
  src/pgen/scf_rsg.cpp src/pgen/scf_axisymmetric_profile.hpp \
  > "$run/provenance.sha256"

/usr/bin/time -f 'launcher_wall_s=%e launcher_maxrss_kb=%M' \
  -o "$run/time.log" \
  srun -n"$SLURM_NTASKS" --ntasks-per-node=56 \
  bin/athena -i "$input" -d "$run" \
  time/nlim=300 time/tlim=1.0e-2 time/ncycle_out=10 \
  output1/dcycle=10 output2/dcycle=1000 \
  output3/dt=-1 output4/dt=-1 \
  > "$run/stdout.log" 2> "$run/stderr.log"

grep -q 'Terminating on time limit' "$run/stdout.log"
if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
    "$run/stdout.log" "$run/stderr.log"; then
  echo 'nonrotating FLD startup gate failed' >&2
  exit 1
fi
test -s "$run/${problem_id}.final.rst"
grep -E '\[NR\] status=|\[LINEAR_FGMRES\]|FLD_OPACITY_DIAGNOSTICS' \
  "$run/stdout.log" > "$run/solver_summary.log" || true
sacct -j "${SLURM_JOB_ID}" -n -P \
  --format=JobID,State,Elapsed,TotalCPU,MaxRSS,MaxVMSize,AllocCPUS \
  > "$run/slurm_accounting.txt" || true
cat "$run/time.log"
tail -n 30 "$run/stdout.log"
