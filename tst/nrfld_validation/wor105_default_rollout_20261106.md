# WOR-105: validated AMR interface defaults

## Selection and scope

The common-face conservative operator is now selected by default for
multilevel Cartesian Test C and Mach 5 radiative-shock problems. Test C
requires periodic faces, cubic cells, and the existing NR-FLD restrictions;
Mach 5 requires coupled fixed-limiter FLD, user x1 boundaries, periodic
transverse boundaries, and cubic cells. The fixed-limiter Test C default uses
the distance-weighted series-resistance coefficient on all faces. Test C with
a variable limiter uses the existing CASTRO-like face opacity and one shared
fine-face flux; the limiter is frozen within each Newton solve. Mach 5 uses
the shared CASTRO-like face coefficient. Coupled solves use assembled Schur
Jv and FGMRES/MG, not the expensive difference Jv. Tangential reconstruction
is on for the supported defaults.

`ATHENA_NRFLD_LEGACY_INTERFACE=1` explicitly selects the old interface
operator for an A/B run. Setting it together with
`ATHENA_NRFLD_CONSERVATIVE_PROTOTYPE=1` is an error. On other multilevel
NR-FLD problem generators, the code stops with an error unless the legacy
path is explicitly selected. Uniform grids do not have coarse/fine faces and
retain the existing solver path. The log prints `[NRFLD_INTERFACE_MODE]` and
the active conservative options. The cost-bearing tangential reconstruction,
series resistance, and difference Jv remain switchable with their existing
environment flags set to `0` or `1`; the default difference Jv is off.

This is a narrow rollout, not a claim that all FLD physics, coordinates, or
meshes have been validated. Noncubic cells and curvilinear coordinates are
outside the project scope and remain rejected by the geometry guard. The
N4096 full-length convergence study remains WOR-100. A different AMR
problem must explicitly use the legacy route until its conservative path is
validated.

## Local regressions

The double-HDF5 coupled restart regression from WOR-106 passes all six
static/adaptive, legacy/shared, and 4→4/1→4 rank cases after the default
change. Continuous and restart cycle-5 times are identical; the maximum
active-field difference is `8.53e-14`. The same local Test C with actual
derefinement runs five cycles with no prototype flags. Both fixed and variable
limiter coupled defaults converge at final Newton L2 at most `4.35e-11`.
The fixed case logs `series_resistance=1`; the variable case logs `0`.
The common-face periodic diffusion ledger remains at roundoff.

The opt-in series mode and the new default produced **byte-identical VTK
outputs** in the fixed-limiter N32 cosine test. The baseline, shared, and
series errors against the analytic mode were respectively `8.885e-7`,
`4.917e-7`, and `4.917e-7`, with the conservative global diffusion ledger
`~9e-21`. The double-HDF5 two-material eigenmode at N64, density ratio
0.1, and a material jump *inside* the fine region reproduced the earlier
L1 ranking: legacy `3.269e-5`, shared `3.268e-5`, and distance-weighted
series `8.789e-7`; the latter conserved the periodic integral to roundoff.
This supports the fixed-limiter thick-interface default. Explicit legacy,
optional correction-off, conflicting-option, and noncubic-geometry guards
were also exercised.

## XD2000: late Mach 5 physical and flux comparison

Jobs `336882` (legacy) and `336883` (shared face) completed 100 cycles on
32 ranks from the same N2048-equivalent AMR Mach 5 checkpoint at cycle
36517, `t=0.04 s`. Both used the **same binary** from commit `1e9e0b6d`
with the WOR-106 restart/boundary correction. The checkpoint and source-input
SHA-256 values match the pinned WOR-102 A/B inputs. Each retained 263 blocks
and all 100 Newton solves converged. The final times differ by only
`3.25e-9 s`; the spatial leaf grid is identical.

| Quantity | Legacy | Shared face |
| --- | ---: | ---: |
| Max relative coarse/fine flux mismatch | 7.279% | 5.63e-13 |
| Sum of coarse/fine face imbalance | -11.9457 | -2.59e-12 |
| Domain implicit diffusion | -3.08818 | 8.85567 |
| Measured physical-boundary diffusion | 8.85753 | 8.85567 |
| FGMRES iterations, 100 steps | 5400 | 2700 |
| Max Newton final L2 | 2.05e-12 | 2.53e-15 |
| Job wall time | 26.60 s | 16.05 s |

At the shock-aligned leaf centers within 300 cm of the density jump (106
samples), the mean absolute temperature error against the existing Mach 5
semi-analytic reference is `1.784→1.666 K` for gas and `1.058→0.923 K`
for radiation. The largest A/B difference in that sample is `0.225 K` in
gas and `0.187 K` in radiation. All density and temperature values remain
positive. Thus the corrected conservative path improves this late,
same-checkpoint profile as well as interface conservation. The near-shock
comparison is not a full-run N4096 convergence claim.

Remote artifacts: `/work/mizutaniks/wor105/{build,output}`. The two final
HDF profiles are copied to `/tmp/wor105_xd_profile_20261106`. Exact job
scripts are `wor105_xd_predefault_build_20261106.sbatch` and
`wor105_xd_predefault_ab_20261106.sbatch` in this directory. This XD run
compared the two explicit modes before changing the default selector; the
conservative operator itself is the same. Final no-flag job `336891` used
the default-selector commit `f4f6ec41`, completed 100 cycles on 32 ranks,
and logged `path=conservative selection=supported_default`. Its final HDF
leaf locations and all user fields are **exactly equal** to the explicit
candidate job, and its 100-cycle face/energy audit is numerically identical:
maximum relative coarse/fine mismatch `5.63e-13`, physical-boundary
diffusion `8.855674579318485`, and domain diffusion `8.855674579315895`.
The final no-flag profile is copied beside the others to
`/tmp/wor105_xd_profile_20261106/default`. The job scripts are
`wor105_xd_default_build_20261106.sbatch` and
`wor105_xd_default_run_20261106.sbatch`.
