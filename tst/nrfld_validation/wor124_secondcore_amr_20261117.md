# WOR-124 second-core preparation, full-domain AMR gate

## State at 2026-11-17 20:24 JST

The original worktree has unrelated uncommitted changes. This experiment uses
the isolated branch `exp/wor124-second-core-amr-gate-20261117`, based on the
completed cutout audit commit `9db57282d1b83f612702bba8a53b7b3e9c1ea40f`.
No main-branch merge or source-code change is part of this gate.

The previous central cutout is not a valid second-core production setup:
its fluid, MHD, and FLD boundaries remained periodic, and the crop's gravity
excluded external mass. At cycle 2250, matched-cell radiation energy across
the ±750 au crop exceeded the full-domain value by 160.5%. The center itself
remained close, which supports a diagnostic use only.

## XD job progress

- Rotation-only high-AMR job `341013` was RUNNING on M-large-a, two nodes and
  128 MPI ranks. Its latest completed segment was cycle 4390, with 5146
  blocks, `rho_max=2.11970e-11 g cm^-3`, and minimum Jeans resolution 27.61
  cells. Its 24-hour limit ends at about 2026-11-17 22:12 JST. This job was
  left untouched.
- Full-domain five-cycle jobs `341626` (`mesh/numlevel=11` control) and
  `341627` (`mesh/numlevel=12` trial) completed with PASS on M-test-a from the
  same cycle-2250 weak-field plus rotation restart. Both used source commit
  `7dccd1af`, binary SHA256 `2a8b93e1...`, 64 MPI ranks, one node, CFL=0.3,
  and identical NR-FLD settings. The exact job script is
  [`wor124_secondcore_amr_gate.sbatch`](wor124_secondcore_amr_gate.sbatch).
- The checkpoint had 4572 blocks, `rho_max=1.01188e-10 g cm^-3`,
  `Tgas_max=338.34 K`, and a 16.13-cell minimum Jeans resolution. The gate
  required an intact cycle-2255 restart, no opacity-domain event, positive
  density and temperature, and at least 16 Jeans cells.

### AMR-gate result

| job | levels | wall | blocks | Jeans min | rho max | T max | MPI MaxRSS | result |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 341626 | 11 | 139 s | 4572 | 16.0846 | 1.02140e-10 | 339.72 K | 43,625,476 KiB | PASS |
| 341627 | 12 | 148 s | 4852 | 32.1235 | 1.02615e-10 | 340.34 K | 46,334,848 KiB | PASS |

The additional level doubled the minimum Jeans-cell count for 6.12% more
blocks, 6.47% more elapsed time, and 6.21% more MPI-step MaxRSS. Both cases
used three Newton iterations per cycle. The level-11 FGMRES mean/max was
9.8/10 iterations and the level-12 mean/max was 9.47/10; the largest reported
true relative residuals were `1.96e-8` and `1.87e-8`, respectively. The
largest final Newton L2 residual was `2.43e-10` in both. Level 12 recorded a
localized final-max residual `5.12e-6` immediately after regridding while its
global L2 remained `1.20e-10`; all calls reported converged.

Across the stored post-restart history, relative mass drift was `-7.15e-9`
at level 11 and `-2.94e-9` at level 12; Lz drift was `1.54e-7` and `7.13e-8`.
Opacity diagnostics were identically zero for out-of-domain, clamp, negative,
zero, and non-finite events. This is a successful spatial-AMR gate, not yet a
second-core endpoint demonstration. Machine-readable provenance and metrics
are in
[`wor124_secondcore_amr_gate_results_20261117.json`](wor124_secondcore_amr_gate_results_20261117.json).

## Local table feasibility audit

The current material-only Tomida-family EOS table spans `1e-21` to
`1e-4 g cm^-3`, with temperatures about 5 to 100,000 K. The current
Semenov-based opacity table spans `1e-21` to `1e-7 g cm^-3` and about 5
to 9900 K; its source opacity itself is only valid down to `2e-18 g cm^-3`.
The initial second-core scale is much smaller than the current finest
cell, so neither table extension nor spatial refinement alone is sufficient.

The existing EOS generator produced a finite, positive, monotone *trial*
table through `1e-2 g cm^-3` with the same H/He model, at 107 density and
256 temperature grid points. Its common inverse-energy axis contracted from
`log10 x2=[8.41578,13.29615]` to `[8.41578,13.25155]`. At 30 overlapping
sample coordinates, the largest relative interpolation changes were
`4.67e-4` in `P/e`, `3.64e-4` in temperature, `1.46e-4` in entropy, and
`7.64e-3` in `nabla_ad`. The trial table and audit are under
`/home/kosuke/simulation/test/collapse_rotmhd_20261117/secondcore_amr_gate_20261117/`.
This is only a generator audit. It is not a validated physical second-core
EOS/opacity package; the high-density gas opacity and smooth restart
transition still need independent work.

Tomida et al. (2013), Appendix 2, combined Semenov dust with Ferguson
low-temperature gas and Opacity Project gas tables, and explicitly noted
that the dense protostellar region lies outside the available opacity
coverage. They extrapolated the nearest high-density values there while
describing the uncertainty. Their Appendix 1 also warns that their ideal
chemical-equilibrium EOS becomes invalid above roughly `0.1 g cm^-3`.
Our `1e-2 g cm^-3` trial stays below that stated EOS limit.

## OPTAB high-temperature opacity bridge

The available OPTAB gray table is the appropriate high-temperature gas
source, not a replacement for low-temperature dust. Its qualified axes are
`log10(rho)=[-23.3000,-1.19314]` and `log10(T)=[2.5,7.0]`; its composition is
`X=0.70, Y=0.28, Z=0.02`. The Planck mean is line plus continuum absorption
without pure scattering, while the Rosseland mean includes absorption and
scattering. Its metadata records OPTAB commit
`2d95b7c1a944e15d80605afee783c22eed441ae1`.

Direct overlap comparison explains why it is needed. At `rho=1e-10 g cm^-3`,
Semenov/OPTAB Planck ratios are about 0.0194 at 3000 K and `5.06e-5` at
5000 K. Semenov's own source describes its gas table as the 1500--10,000 K
component, while its strength is the dust and dust-evaporation model.

[`make_protostellar_hybrid_opacity.py`](../../tools/collapse/make_protostellar_hybrid_opacity.py)
therefore retains Semenov through the end of its density-dependent final dust
evaporation interval (`T_evap+100 K`), blends the logarithmic opacities with a
cubic smoothstep, and becomes pure OPTAB at `T_evap+300 K`. This corresponds
to pure OPTAB above 1202--2074 K across the density grid. OPTAB is never
extrapolated. Below the bridge, Semenov density is held at its nearest source
row outside the qualified range, and that limitation remains explicit.

The trial grid is uniform, as required by Athena's HDF5 reader, with 256
density points over `log10(rho)=[-21,-2]` and 2048 temperature points over
5--100,000 K. It is 8.1 MiB and has SHA256 `3a704854...`. A deterministic
second-interpolation audit gives Planck/Rosseland 99th-percentile source
agreement below the bridge of `6.40e-4/7.71e-4 dex` and above it of
`8.27e-4/8.92e-4 dex`. The worst low-temperature Rosseland point is
0.140 dex at a sharp dust feature; the high-temperature maxima are only
0.00486 and 0.00275 dex.

A static transition-width bracket is recorded in
[`wor124_optab_transition_sensitivity_20261117.json`](wor124_optab_transition_sensitivity_20261117.json).
Changing the pure-OPTAB endpoint from the 300 K default to 200 or 400 K has
zero median effect over 900--2300 K because most cells lie outside the narrow
handoff, but its 99th-percentile Planck difference is 0.291 or 0.165 dex and
its Rosseland difference is 0.205 or 0.155 dex. Thus the choice is localized
but not numerically negligible while a trajectory crosses dust evaporation.

XD job `341645` used
[`wor124_secondcore_opacity_gate.sbatch`](wor124_secondcore_opacity_gate.sbatch)
to compare this table against level-12 control job `341627` from exactly the
same cycle-2250 restart for five cycles and completed with PASS on M-test-a.
It reached cycle 2255 in 140 s with 4852 blocks, a 32.1235-cell minimum Jeans
resolution, `rho_max=1.02615e-10 g cm^-3`, `Tmax=340.339 K`, and no opacity
domain or value event. MPI-step MaxRSS was 51,598,952 KiB, 11.36% above the
control's single-run watermark; wall time was 5.4% lower, so neither difference
is treated as a stable performance measurement.

The hybrid and control used identical Newton and FGMRES iteration sequences:
three Newton iterations per cycle and FGMRES mean/max 9.47/10. Their final
rho, temperature, Jeans resolution, and radiation-energy histories agree to
approximately `1e-11` relative. `divB_max` differed by 5.5% but remained only
`3.14e-10` in absolute value. The duplicate M-debug fallback job `341648` was
canceled while pending once the M-test job started.

Because the checkpoint is only 340 K, this gate validates table reading,
low-temperature equivalence, restart, Newton/FGMRES, AMR, conservation, and
memory; it does not exercise the OPTAB-dominated high-temperature regime. A
continuation that actually crosses dust evaporation remains required before
choosing a transition width and using the bridge for a second-core production
claim.
