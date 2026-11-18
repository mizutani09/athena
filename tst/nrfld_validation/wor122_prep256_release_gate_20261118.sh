#!/bin/bash
# Matched 256^3 control and genuinely unsupported continuation to t=130.
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --job-name=wor122_prep256_release_gate
#SBATCH --output=/work/mizutaniks/wor122/prep256_release_gate_%j.out
set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5 >/dev/null 2>&1
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0

base=/work/mizutaniks/wor122
code="$base/athena_prep_thermal"
test -x "$code/bin/athena"
cd "$code"

for mode in hold off; do
  if [[ "$mode" == hold ]]; then
    rst="$base/prep256_hold_t120/wor122_prep256_hold_t120.final.rst"
    source_args=(problem/prep_thermal_tau=0.1)
  else
    rst="$base/prep256_taper_t120/wor122_prep256_taper_t120.final.rst"
    source_args=(problem/prep_thermal_tau=-1.0)
  fi
  test -s "$rst"
  run="$base/prep256_${mode}_t130"
  prefix="wor122_prep256_${mode}_t130"
  mkdir -p "$run"
  srun -n64 bin/athena -r "$rst" -d "$run" \
    "${source_args[@]}" job/problem_id="$prefix" \
    time/tlim=130.0 time/nlim=4000 problem/short_test_max_dt=0.1 \
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
assert last[0] >= 129.99 and np.isfinite(last).all()
print('RSG_RELEASE_GATE', sys.argv[2], 'time', last[0],
      'Mach_max', last[16], 'Pprep_over_Lheat', last[29]/last[15],
      'Lrad_over_Lheat', last[22:28].sum()/last[15],
      'mass', last[2], flush=True)
PY
done
