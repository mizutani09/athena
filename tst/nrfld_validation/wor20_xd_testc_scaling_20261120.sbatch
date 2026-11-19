#!/usr/bin/env bash
# XD2000 conservative Test-C AMR multi-node scaling pilot (WOR-20).
# Submit with: sbatch --nodes=N --export=ALL,WOR20_REPEAT=R this_file
# N is 1, 2, or 4; one repeat per allocation.
#SBATCH --job-name=wor20_amr_scale
#SBATCH --account=a
#SBATCH --partition=M-test-a
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/work/mizutaniks/wor20/xd_scaling_20261120/slurm_%x_%j.out

set -euo pipefail
: "${WOR20_REPEAT:=0}"
case "$SLURM_JOB_NUM_NODES" in 1|2|4) ;; *) echo "nodes must be 1, 2, or 4"; exit 2;; esac
[[ "$WOR20_REPEAT" =~ ^[0-9]+$ ]]
expected_ranks=$((32*SLURM_JOB_NUM_NODES))
test "$SLURM_NTASKS" -eq "$expected_ranks"

root=/work/mizutaniks/wor20/xd_scaling_20261120
repo=/work/mizutaniks/wor20/athena-performance-20261120
binary=$repo/bin/athena
wrapper=$root/assets/rank_resource_wrapper.py
wrapper_sha=d30afe06b6b70e72686051cc3479b82dcf43fc5468a2e0ead497877b38776d3c
python=/work/opt/cray/pe/python/3.11.7/bin/python3.11
commit=6ff44924b64d955be27fd240269ca4cbd488b982
binary_sha=f42869088d82a3294b35679380a06bac4140163bf0cdbc94df47f4779d38aca6
run=$root/run_${SLURM_JOB_NUM_NODES}n_r${WOR20_REPEAT}_${SLURM_JOB_ID}
test ! -e "$run"
mkdir -p "$run"
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test "$(sha256sum "$binary" | awk '{print $1}')" = "$binary_sha"
test "$(sha256sum "$wrapper" | awk '{print $1}')" = "$wrapper_sha"
test -x "$python"

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
export OMP_NUM_THREADS=1 I_MPI_PMI_LIBRARY=/lib64/libpmi2.so

cat > "$run/input.used" <<'INPUT'
<job>
problem_id = wor20_xd_testc

<time>
cfl_number = 0.3
nlim = 12
tlim = 0.0012
integrator = vl2
xorder = 2
ncycle_out = 12

<mesh>
refinement = static
nx1 = 128
x1min = 0
x1max = 1
ix1_bc = periodic
ox1_bc = periodic
nx2 = 128
x2min = 0
x2max = 1
ix2_bc = periodic
ox2_bc = periodic
nx3 = 128
x3min = 0
x3max = 1
ix3_bc = periodic
ox3_bc = periodic

<meshblock>
nx1 = 8
nx2 = 8
nx3 = 8

<refinement1>
level = 1
x1min = 0.25
x1max = 0.75
x2min = 0.25
x2max = 0.75
x3min = 0.25
x3max = 0.75

<hydro>
gamma = 1.6666666666666667
rho_unit = 1e-7
egas_unit = 1e7
leng_unit = 4e10
mu = 1
active = fixed

<problem>
density_low = 1
density_high = 1.15
slab_left = 0.25
slab_right = 0.75
interface_width = 0
erad_background = 1
erad_amplitude = 0.1
pulse_center = 0.25
pulse_width = 0.08
pressure = 100

<performance>
enabled = true
warmup = 2
linear_detail = true

<fld>
is_couple = false
only_rad = true
cut_diff = false
include_radiation_force = false
include_mixed_frame_terms = false
fixed_flux_limiter = true
const_opacity = 0.4
stage_diagnostics = false

<nrfld>
mgmode = MGI
fas = false
smoother = jacobi-rb
linear_solver = fgmres
fgmres_threshold = 1e-10
fgmres_restart = 20
fgmres_max_iterations = 200
fgmres_mg_vcycles = 1
coarse_operator = rediscretized
prolongation = trilinear
npresmooth = 2
npostsmooth = 2
omega = 1
threshold = 1e-10
nr_threshold = 1e-8
nr_max_iterations = 100
failure_policy = abort
diagnostic_verbosity = 0
coarse_solver = direct
coarse_direct_max_cells = 64
coarse_diagnostics = false
ix1_bc = periodic
ox1_bc = periodic
ix2_bc = periodic
ox2_bc = periodic
ix3_bc = periodic
ox3_bc = periodic

<output1>
file_type = hst
dcycle = 12
data_format = %.16e
INPUT

cp "$0" "$run/job.sbatch"
{
  printf 'job_id=%s\nnodes=%s\nranks=%s\nrepeat=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_NUM_NODES" "$SLURM_NTASKS" "$WOR20_REPEAT"
  printf 'source_commit=%s\nbinary_sha256=%s\nwrapper_sha256=%s\ninput_sha256=%s\n' \
    "$commit" "$binary_sha" "$wrapper_sha" \
    "$(sha256sum "$run/input.used" | awk '{print $1}')"
  module list 2>&1
} > "$run/provenance.txt"

cd "$run"
set +e
srun --mpi=pmi2 --ntasks="$SLURM_NTASKS" --cpu-bind=cores \
  "$python" "$wrapper" "$run" "$binary" -i input.used > run.log 2>&1
raw_status=$?
set -e

"$python" - "$run" "$raw_status" <<'PY'
import json, math, re, statistics, sys
from pathlib import Path
root=Path(sys.argv[1]); raw_status=int(sys.argv[2])
resources=[json.loads(p.read_text()) for p in sorted(root.glob('resource_rank*.json'))]
timings=[json.loads(p.read_text()) for p in sorted(root.glob('performance_rank*.json'))]
log=(root/'run.log').read_text(errors='replace')
prefix='[LINEAR_FGMRES_TIMING] '
detail=[]
for line in log.splitlines():
    if line.startswith(prefix):
        detail.append({k:float(v) for k,v in (token.split('=',1) for token in line[len(prefix):].split())
                       if k.endswith('_seconds')})
fgmres=[int(x) for x in re.findall(r'^\[LINEAR_FGMRES\].*?\biterations=(\d+)',log,re.M)]
history=[]
for line in (root/'wor20_xd_testc.hst').read_text().splitlines():
    if line and not line.startswith('#'):
        history.append([float(x) for x in line.split()])
total=sum(row['total_seconds'] for row in detail)
summary={
 'status':'passed', 'raw_status':raw_status, 'nodes':int(__import__('os').environ['SLURM_JOB_NUM_NODES']),
 'ranks':int(__import__('os').environ['SLURM_NTASKS']), 'repeat':int(__import__('os').environ['WOR20_REPEAT']),
 'meshblocks':7680, 'leaf_cells':3932160, 'measured_steps':10,
 'seconds_per_step':max(row['wall_seconds'] for row in timings)/10,
 'max_rank_rss_kib':max(row['max_rss_kib'] for row in resources),
 'sum_rank_peak_rss_kib':sum(row['max_rss_kib'] for row in resources),
 'newton_iterations':sorted({step['newton_iterations'] for row in timings for step in row['steps']}),
 'fgmres_iterations_max':max(fgmres), 'fgmres_iterations_total':sum(fgmres),
 'mass_relative_drift':(history[-1][2]-history[0][2])/history[0][2],
 'linear_solves':len(detail),
 'fractions':{key:sum(row[key] for row in detail)/total for key in
  ('preconditioner_seconds','matvec_seconds','reduction_seconds','vector_and_other_seconds')},
}
valid=(raw_status==0 and len(resources)==summary['ranks'] and len(timings)==summary['ranks']
 and all(r['returncode']==0 for r in resources)
 and all(len(r['steps'])==10 and all(s['newton_reason']=='converged' for s in r['steps']) for r in timings)
 and len(detail)==12 and len(fgmres)==12 and summary['fgmres_iterations_max']>0
 and abs(summary['mass_relative_drift'])<=1e-12
 and '[NRFLD_INTERFACE_MODE] path=conservative selection=supported_default' in log
 and 'FATAL' not in log and '### WARNING' not in log
 and 'Terminating on cycle limit' in log)
summary['status']='passed' if valid else 'failed'
(root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
(root/'result.txt').write_text('PASS\n' if valid else 'FAIL\n')
if not valid: raise SystemExit(1)
PY

cat "$run/result.txt"
