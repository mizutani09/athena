# NR-FLD RSG commissioning and HPC runs

This note records the 2026-09-01 commissioning state.  One code time is
43.6499 days, the dynamical time is about 1.538 code times, and the radial
sound-crossing time of the initial RSG is about 2.675 code times.

## Current conclusion

The NR/MG solver is numerically stable at CFL=0.3 and the fast tolerances in
the supplied inputs.  A strict/fast comparison agreed at roughly 1e-5 in the
fields while the strict solve was about eleven times slower.  No NR/MG
convergence warnings occurred in the long 128^3 experiments.

The remaining limitation is spatial resolution and relaxation of the mapped
stellar core, not the implicit tolerance.  Releasing the numerical core over
two code times produced envelope max/RMS Mach 0.172/0.033 at t=6.2.  A
four-code-time release reduced this to 0.0995/0.0186 at the same time, but the
outgoing core-boundary wave later reached max Mach 0.229.  At t=10.2 it had
fallen to 0.118, while r50 had moved from about 0.257 to 0.302.  Thus 128^3 is
useful for commissioning but has not demonstrated a clean freely evolving
low-Mach star.

A broad increase of the discrete-HSE correction in the core was tested and
rejected: at 64^3 it increased envelope max Mach from 0.0407 to 0.114 at
t=6.5.  That experimental option is not retained in the production pgen.

The profile generator now evaluates dE_rad/dr analytically from the same HSE
and EOS equations used to construct the star.  This removes roundoff-driven
central opacity oscillations without changing density, pressure, temperature,
mass, or the target luminosity.  On the current grids the oscillations were
inside the innermost cell and therefore were not the cause of the observed
expansion, but the analytic form is the correct one for higher resolution.

## Resolution and rank choices

The domain is [-1.28,1.28]^3 and MeshBlock size is 32^3.

| root grid | blocks | suggested MPI ranks | role |
|---:|---:|---:|---|
| 256^3 | 512 | 256 or 512 | pilot and parameter validation |
| 512^3 | 4096 | 512 or 1024 | production-resolution relaxation |

Benchmark 256 and 512 ranks for the 256^3 pilot before choosing.  For 512^3,
1024 ranks gives four MeshBlocks per rank.  The 512^3 paired prim+uov dumps are
about 7.5 GB each with the present variable set, so output cadence must be
chosen with storage in mind.

## Build

Run on the target CPU because `g++-simd` enables native optimization.

```bash
cd /path/to/athena
python3 configure.py --prob=nrfld_conv_star --eos=adiabatic \
  -nrmgfld -mpi -hdf5 --cxx=g++-simd
make -j 32
```

For a portable rather than CPU-native build, use `--cxx=g++`.

## Isolated-star sequence

First run the convectively stable n=3 control on 256^3.  `$PWD` below must be
the repository root; use a filesystem local to the compute nodes for output.

```bash
repo=$PWD
run_dir=$SCRATCH/nrfld_rsg/stable_n3_256
mkdir -p "$run_dir"
srun --mpi=pmix -n 256 --cpu-bind=cores "$repo/bin/athena" \
  -i "$repo/inputs/radiation/athinput.nrfld_conv_star_hpc" \
  -d "$run_dir" -t 23:50:00 \
  problem/stellar_profile_file="$repo/inputs/radiation/rsg10_stable_n3.dat" \
  problem/velocity_perturb_fraction=0.0 \
  2>&1 | tee "$run_dir/run.log"
```

Then run the mildly convectively unstable target profile with the same grid.

```bash
run_dir=$SCRATCH/nrfld_rsg/mild_conv_256
mkdir -p "$run_dir"
srun --mpi=pmix -n 256 --cpu-bind=cores "$repo/bin/athena" \
  -i "$repo/inputs/radiation/athinput.nrfld_conv_star_hpc" \
  -d "$run_dir" -t 23:50:00 \
  problem/stellar_profile_file="$repo/inputs/radiation/rsg10_modified_le_mild.dat" \
  2>&1 | tee "$run_dir/run.log"
```

If both pass, repeat at 512^3 on 1024 ranks:

```bash
run_dir=$SCRATCH/nrfld_rsg/mild_conv_512
mkdir -p "$run_dir"
srun --mpi=pmix -n 1024 --cpu-bind=cores "$repo/bin/athena" \
  -i "$repo/inputs/radiation/athinput.nrfld_conv_star_hpc" \
  -d "$run_dir" -t 23:50:00 \
  mesh/nx1=512 mesh/nx2=512 mesh/nx3=512 \
  problem/stellar_profile_file="$repo/inputs/radiation/rsg10_modified_le_mild.dat" \
  2>&1 | tee "$run_dir/run.log"
```

The input ends at t=16: core-anchor release is complete at t=8, global
velocity damping is complete at t=14, and the final two code times are free
evolution.  Continue a successful relaxed run to t=50 first, then t=100:

```bash
next_dir=$SCRATCH/nrfld_rsg/mild_conv_512_t50
mkdir -p "$next_dir"
srun --mpi=pmix -n 1024 --cpu-bind=cores "$repo/bin/athena" \
  -r "$run_dir/nrfld_conv_star.final.rst" \
  -i "$repo/inputs/radiation/athinput.nrfld_conv_star_hpc" \
  -d "$next_dir" -t 23:50:00 \
  mesh/nx1=512 mesh/nx2=512 mesh/nx3=512 \
  time/tlim=50.0 output2/dt=2.0 output3/dt=2.0 output4/dt=5.0 \
  problem/stellar_profile_file="$repo/inputs/radiation/rsg10_modified_le_mild.dat" \
  2>&1 | tee "$next_dir/run.log"
```

If the site uses Open MPI directly rather than Slurm PMIx, replace `srun ...`
with `mpirun --bind-to core --map-by core -np N ...`.

## Acceptance checks

After t=14, require no NaN or NR/MG convergence warnings, mass drift below
about 0.5%, stable mass radii, and no coherent radial expansion.  For the n=3
control, a practical target is envelope max Mach below 0.05 and mass-weighted
RMS below 0.01.  For the mildly unstable target, subsonic local convection is
expected, but the mass-weighted radial velocity should remain much smaller
than the fluctuating velocity and the mass radii should not drift secularly.

```bash
python3 "$repo/utils/diagnose_nrfld_conv_star.py" "$run_dir" \
  --inner 0.25 --output "$run_dir/diagnostics.txt"
python3 "$repo/utils/plot_nrfld_conv_star.py" "$run_dir" \
  --dest "$run_dir/plots" --repeat 1
```

## CE commissioning

Build the CE pgen separately and run the supplied CE input.  This setup uses a
prescribed circular companion and indirect acceleration; it does not yet
include gas backreaction or dynamical inspiral.

```bash
cd "$repo"
python3 configure.py --prob=nrfld_conv_star_ce --eos=adiabatic \
  -nrmgfld -mpi -hdf5 --cxx=g++-simd
make -j 32

run_dir=$SCRATCH/nrfld_rsg/ce_256
mkdir -p "$run_dir"
srun --mpi=pmix -n 480 --cpu-bind=cores "$repo/bin/athena" \
  -i "$repo/inputs/radiation/athinput.nrfld_conv_star_ce" \
  -d "$run_dir" -t 23:50:00 \
  problem/stellar_profile_file="$repo/inputs/radiation/rsg10_modified_le_mild.dat" \
  2>&1 | tee "$run_dir/run.log"
```

The CE input uses a root domain [-2.56,2.56]^3 plus one static-refinement level
inside [-1.28,1.28]^3.  Its 256^3 root grid has 960 MeshBlocks; the 512^3 root
grid has 7680, so 480 and 960 ranks respectively give exactly two and eight
MeshBlocks per rank.  The inner stellar resolution matches the corresponding isolated-star
run.  The companion starts at t=16 and reaches full strength at t=20.  Promote
to a 512^3 root grid with `-n 960` and
`mesh/nx1=512 mesh/nx2=512 mesh/nx3=512` only after the isolated-star 512^3
relaxation passes.
