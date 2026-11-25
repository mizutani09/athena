#!/bin/bash
#SBATCH --partition=M-test-a
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=wor182_ssp_fld
#SBATCH --output=/work/mizutaniks/wor182/ssprk54_transport_%j.out

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

repository=${WOR182_SSP_REPOSITORY:-$HOME/simulation/athena-wor182-ssptest}
input=inputs/radiation/athinput.nrfld_fourth_order_transport
root=/work/mizutaniks/wor182/ssprk54_transport_${SLURM_JOB_ID}
mkdir -p "$root"
cd "$repository"
test -x bin/athena
test -s "$input"
git rev-parse HEAD > "$root/git_commit.txt"
sha256sum bin/athena "$input" > "$root/provenance.sha256"

for n in 16 32 64; do
  run="$root/n${n}"
  mkdir -p "$run"
  srun -n8 --ntasks-per-node=4 bin/athena -i "$input" -d "$run" \
    time/integrator=ssprk5_4 time/xorder=4 time/cfl_number=0.2 \
    mesh/nx1="$n" mesh/nx2="$n" mesh/nx3="$n" \
    meshblock/nx1=8 meshblock/nx2=8 meshblock/nx3=8 \
    output1/dcycle=1 \
    > "$run/stdout.log" 2> "$run/stderr.log"
  if grep -Eiq 'nan|fatal error|segmentation fault|bad_alloc|out of memory' \
      "$run/stdout.log" "$run/stderr.log"; then
    echo "N=$n failed" >&2
    exit 1
  fi
  hst="$run/nrfld_fourth_order_transport.hst"
  test -s "$hst"
  printf 'N=%s final_history=' "$n"
  tail -n 1 "$hst"
done
