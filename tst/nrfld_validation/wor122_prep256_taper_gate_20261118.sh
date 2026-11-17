#!/bin/bash
# Paired short continuation from the same t=110 restart. Heavy files stay in WORK.
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --job-name=wor122_prep256_taper_gate
#SBATCH --output=/work/mizutaniks/wor122/prep256_taper_gate_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5 >/dev/null 2>&1
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0

base=/work/mizutaniks/wor122
code="$base/athena_prep_thermal"
source_rst="$base/prep256_fgmres_r40_t110/wor122_prep256_fgmres_r40_t110.final.rst"
test -s "$source_rst"
test -x "$code/bin/athena"
cd "$code"

for mode in hold taper; do
  run="$base/prep256_${mode}_t120"
  prefix="wor122_prep256_${mode}_t120"
  mkdir -p "$run"
  if [[ "$mode" == taper ]]; then
    taper_args=(problem/prep_thermal_taper_start=110.0
                problem/prep_thermal_taper_end=120.0)
  else
    taper_args=(problem/prep_thermal_taper_start=-1.0
                problem/prep_thermal_taper_end=-1.0)
  fi
  srun -n64 bin/athena -r "$source_rst" -d "$run" \
    "${taper_args[@]}" job/problem_id="$prefix" \
    time/tlim=120.0 time/nlim=4000 problem/short_test_max_dt=0.1 \
    nrfld/fgmres_restart=40 nrfld/fgmres_max_iterations=500 \
    nrfld/fgmres_mg_vcycles=1 \
    output1/dt=1.0 output2/dt=5.0 output3/dt=5.0 output4/dt=1000.0 \
    > "$run/stdout.log" 2> "$run/stderr.log"
  grep -q 'Terminating on time limit' "$run/stdout.log"
  grep -q 'FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0' "$run/stdout.log"
  /home/mizutaniks/.conda/envs/py311env/bin/python3 - "$run/$prefix.hst" "$mode" <<'PY'
import sys
import numpy as np
data = np.atleast_2d(np.loadtxt(sys.argv[1]))
last = data[-1]
assert last[0] >= 119.99 and np.isfinite(last).all()
print('RSG_TAPER_GATE', sys.argv[2], 'time', last[0],
      'Mach_max', last[16], 'Pprep_over_Lheat', last[29]/last[15],
      'Lrad_over_Lheat', last[22:28].sum()/last[15],
      'mass', last[2], flush=True)
PY
done
