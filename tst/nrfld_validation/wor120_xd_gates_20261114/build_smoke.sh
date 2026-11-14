#!/bin/bash
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --job-name=wor120_build
#SBATCH --output=/work/mizutaniks/wor120/build_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5 >/dev/null 2>&1
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0
root=/work/mizutaniks/wor120/athena
run=/work/mizutaniks/wor120/smoke64
cd "$root"
module list 2> /work/mizutaniks/wor120/modules_build.txt
python3 configure.py --prob=rsg_star_in_box --coord=cartesian --eos=general/eos_table --flux=hllc -nrmgfld -mpi -hdf5 --nghost=4 --cxx=g++ --mpiccmd=mpiicpx > /work/mizutaniks/wor120/configure.log
make clean > /work/mizutaniks/wor120/make_clean.log 2>&1
make -j16 > /work/mizutaniks/wor120/make.log 2>&1
cp -p bin/athena bin/athena_wor120_e4233143
sha256sum bin/athena_wor120_e4233143 data/eos/rsg_tomida2013_material_eos.tab data/opacity/rsg_gray_opacity_wor55.h5 data/ic/st35gm03n07_wor55_lte_fld_trial.dat > /work/mizutaniks/wor120/source_assets.sha256
mkdir -p "$run"
srun -n8 bin/athena_wor120_e4233143 -i inputs/rsg/athinput.st35gm03n07_fld_wor55_64_lte_fld_trial -d "$run" time/nlim=1 time/tlim=3.318764 > "$run/stdout.log" 2> "$run/stderr.log"
grep -q 'Terminating on cycle limit' "$run/stdout.log"
