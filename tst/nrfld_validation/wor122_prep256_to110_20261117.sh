#!/bin/bash
# Continue the supported 256^3 RSG through one optimistic surface eddy time.
# Copy to $WORK/wor122 and submit there; all large data stay under $WORK.
#SBATCH --partition=M-bulk-a
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=110G
#SBATCH --time=11:30:00
#SBATCH --job-name=wor122_prep256_to110
#SBATCH --output=/work/mizutaniks/wor122/prep256_to110_%j.out

set -euo pipefail
module purge >/dev/null 2>&1
module load tbb/2022.1 compiler-rt/2025.1.1 umf/0.10.0
module load PrgEnv-oneapi/2025.1
module load hdf5-parallel/1.14.5 >/dev/null 2>&1
export SLURM_MPI_TYPE=pmi2 LD_LIBRARY_PATH=/lib64:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=1
ulimit -c 0

base=/work/mizutaniks/wor122
code="$base/athena_prep_thermal"
source="$base/prep256_fgmres_r40_t52"
source_rst="$source/wor122_prep256_fgmres_r40_t52.final.rst"
test -s "$source_rst"
test -x "$code/bin/athena"

gate() {
  local hist="$1" target="$2" initial_mass="$3"
  /home/mizutaniks/.conda/envs/py311env/bin/python3 - "$hist" "$target" "$initial_mass" <<'PY'
import math
import sys
from pathlib import Path

rows = [[float(x) for x in line.split()] for line in Path(sys.argv[1]).read_text().splitlines()
        if line.strip() and not line.startswith('#')]
assert rows, 'empty history'
last = rows[-1]
time_target, initial_mass = float(sys.argv[2]), float(sys.argv[3])
time, mass, mach = last[0], last[2], last[16]
luminosity_ratio = sum(last[22:28]) / last[15]
mass_drift = mass / initial_mass - 1.0
print('RSG_GATE', 'time', time, 'Mach_max', mach,
      'Lrad_over_Lheat', luminosity_ratio, 'mass_drift', mass_drift, flush=True)
assert math.isfinite(time) and time >= time_target - 0.01
assert math.isfinite(mach) and mach < 0.5
assert math.isfinite(mass_drift) and abs(mass_drift) < 1e-4
assert math.isfinite(luminosity_ratio) and 0.5 < luminosity_ratio < 1.5
PY
}

initial_mass=$(/home/mizutaniks/.conda/envs/py311env/bin/python3 - "$source/wor122_prep256_fgmres_r40_t52.hst" <<'PY'
import sys
from pathlib import Path
rows = [line for line in Path(sys.argv[1]).read_text().splitlines()
        if line.strip() and not line.startswith('#')]
last = [float(x) for x in rows[-1].split()]
assert last[0] >= 51.99 and last[16] < 0.3
print(last[2])
PY
)

run80="$base/prep256_fgmres_r40_t80"
mkdir -p "$run80"
cd "$code"
srun -n64 bin/athena -r "$source_rst" -d "$run80" \
  job/problem_id=wor122_prep256_fgmres_r40_t80 \
  time/tlim=80.0 time/nlim=4000 problem/short_test_max_dt=0.1 \
  nrfld/fgmres_restart=40 nrfld/fgmres_max_iterations=500 \
  nrfld/fgmres_mg_vcycles=1 output1/dt=1.0 \
  output2/dt=10.0 output3/dt=10.0 output4/dt=14.0 \
  > "$run80/stdout.log" 2> "$run80/stderr.log"
grep -q 'Terminating on time limit' "$run80/stdout.log"
grep -q 'FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0' "$run80/stdout.log"
gate "$run80/wor122_prep256_fgmres_r40_t80.hst" 80.0 "$initial_mass"

run110="$base/prep256_fgmres_r40_t110"
mkdir -p "$run110"
rst80="$run80/wor122_prep256_fgmres_r40_t80.final.rst"
test -s "$rst80"
srun -n64 bin/athena -r "$rst80" -d "$run110" \
  job/problem_id=wor122_prep256_fgmres_r40_t110 \
  time/tlim=110.0 time/nlim=4000 problem/short_test_max_dt=0.1 \
  nrfld/fgmres_restart=40 nrfld/fgmres_max_iterations=500 \
  nrfld/fgmres_mg_vcycles=1 output1/dt=1.0 \
  output2/dt=10.0 output3/dt=10.0 output4/dt=15.0 \
  > "$run110/stdout.log" 2> "$run110/stderr.log"
grep -q 'Terminating on time limit' "$run110/stdout.log"
grep -q 'FLD_OPACITY_DIAGNOSTICS out_of_domain=0 clamped=0' "$run110/stdout.log"
gate "$run110/wor122_prep256_fgmres_r40_t110.hst" 110.0 "$initial_mass"
