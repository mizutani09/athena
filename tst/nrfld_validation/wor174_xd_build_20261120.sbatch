#!/usr/bin/env bash
# Build the clean WOR-174 solar binary on an XD compute node.
# The worktree must already be detached at the recorded source commit.
#SBATCH --job-name=wor174_solar_build
#SBATCH --account=a
#SBATCH --partition=M-debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/work/mizutaniks/wor174/xd_ab_20261120/build_%j.out

set -euo pipefail
repo=/work/mizutaniks/wor174/athena-solar-topface-clean-20261120
result=/work/mizutaniks/wor174/xd_ab_20261120
commit=f3bfffa4f31d9589ebb2660bed30ba2da02304a8
reference=/home/mizutaniks/simulation/athena-wor91-topface-20261117

source /etc/profile.d/modules.sh
module use /work/opt/intel/modulefiles /work/opt/local_oneapi/modulefiles
module load umf/1.0.2 PrgEnv-oneapi/2025.1 hdf5-parallel/1.14.5
export PATH="${I_MPI_ROOT}/bin:${PATH}"
for var in CPATH LIBRARY_PATH LD_LIBRARY_PATH; do
  IFS=: read -r -a entries <<< "${!var:-}"
  filtered=()
  for entry in "${entries[@]}"; do
    case "$entry" in /work/opt/local/miniconda3/*) ;; *) filtered+=("$entry") ;; esac
  done
  export "$var=$(IFS=:; printf '%s' "${filtered[*]}")"
done

mkdir -p "$result"
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$repo" status --short --untracked-files=no)"
cd "$repo"
/usr/bin/python3 configure.py \
  --prob=simple_convection_lhllc_fld --coord=cartesian \
  --eos=general/eos_table --flux=lhllc -nrmgfld -mpi -hdf5 \
  --hdf5_path=/work/opt/local_oneapi/hdf5-parallel/1.14.5/INTEL/2024.0 \
  --cxx=icpx --mpiccmd=mpiicpx > "$result/configure_${SLURM_JOB_ID}.log"
# The current configure adds -qopenmp-simd, which rejects the table-EOS
# exception path.  Reuse the generated files from the validated solar build;
# their only differences from this configuration are mpiicpx and that flag.
diff -u "$reference/src/defs.hpp" src/defs.hpp > "$result/generated_defs_diff_${SLURM_JOB_ID}.txt" || true
diff -u "$reference/Makefile" Makefile > "$result/generated_makefile_diff_${SLURM_JOB_ID}.txt" || true
test "$(sha256sum Makefile | awk '{print $1}')" = 5ceb1552e79f09ef4d14606abe9bc12766fd7b143af83eb85b68e9e31da6d458
test "$(sha256sum src/defs.hpp | awk '{print $1}')" = 7a650eaceed54f407c122662539083b8b61de8b2c350eaae5c1bccf6c6b3d8d8
test "$(sha256sum "$reference/Makefile" | awk '{print $1}')" = 5b27d00e1d338bda020476161b96451d0d42557694e02f894e97cd02dc3137bb
test "$(sha256sum "$reference/src/defs.hpp" | awk '{print $1}')" = a634b77a19f11c5c6b2d460b5d3cd796922e66c2c701a6cc74e4a60781a1e8b3
cp "$reference/Makefile" Makefile
cp "$reference/src/defs.hpp" src/defs.hpp
make clean > "$result/make_${SLURM_JOB_ID}.log" 2>&1
make -j"$SLURM_CPUS_PER_TASK" >> "$result/make_${SLURM_JOB_ID}.log" 2>&1
test -x bin/athena
ldd bin/athena > "$result/ldd_${SLURM_JOB_ID}.txt"
! grep -q 'not found' "$result/ldd_${SLURM_JOB_ID}.txt"
{
  printf 'job_id=%s\nsource_commit=%s\nsrc_tree=%s\n' \
    "$SLURM_JOB_ID" "$commit" "$(git rev-parse HEAD:src)"
  sha256sum bin/athena Makefile src/defs.hpp \
    src/pgen/simple_convection_lhllc_fld.cpp \
    src/pgen/solar_top_face_opacity.hpp
  module list 2>&1
} > "$result/build_${SLURM_JOB_ID}.txt"
printf 'PASS\n' > "$result/build_${SLURM_JOB_ID}.result"
