# WOR-10: Mach 5 radiation-temperature peak audit (2026-11-04)

## What is measured

The earlier `diagnose_mach5_convergence.py` `bump_height_K` is the positive
**residual against the semi-analytic profile** near the density jump. It is not
the small **spatial local maximum and subsequent minimum of numerical T_rad**.
`mach5_trad_peak_audit.py` reports both. The spatial metric is the largest fall
from a local maximum to its following minimum in the first 300 cm downstream of
the density-jump center. Translating the x origin to the jump cannot change this
temperature difference. The residual comparison does use that density-jump
alignment; it does not optimize the T_rad shift.

All runs reach 0.04 s with strict Newton/FGMRES convergence. The full
radiative-shock validation marks these runs `accuracy_not_met` because of the
separate gas-temperature spike criterion (WOR-67). This report does not override
that verdict. The comparison baseline is the supplied Mach 5 semi-analytic CSV.

| Case | Shock dx [cm] | Spatial wiggle [K] | Residual bump [K] | T_gas peak near jump [K] |
| --- | ---: | ---: | ---: | ---: |
| Current uniform N64 | 93.75 | 0.034607 | 0 | 861.248 |
| Current uniform N128 | 46.875 | 0.108032 | 0.205333 | 866.658 |
| Current uniform N256 | 23.4375 | 0.039368 | 0.236807 | 878.803 |
| Earlier one-level AMR, finest dx=N256 | 23.4375 | 0.025818 | 0 | 878.920 |
| N128 dx, right boundary 1250 cm | 46.875 | 0 | 0.145214 | 867.113 |
| N128 dx, right boundary 2750 cm | 46.875 | 0.019226 | 0.934215 | 866.850 |

The baseline domain is [-4000, 2000] cm. For the right-boundary controls,
`x1min=-4000` and dx=46.875 cm are unchanged: only `nx1/x1max` become
112/1250 or 144/2750, compared with 128/2000. Both boundaries of the earlier
whole-domain translation controls changed, so those are deliberately excluded
from the main conclusion. Both right-boundary controls used the current
N64-run executable; the baseline N128 executable is a separate build of the
same source and configuration. The source is the isolated `codex/wor25-28-10-29-23-24` worktree
based on `209e2239` with only output and analysis changes at this point.

The downstream semi-analytic T_rad is monotone in the measured window. The
uniform-grid spatial wiggle is non-monotone across N64/128/256, so these data do
not establish a clean convergence rate or a unique truncation-error mechanism.
Its maximum is ~0.108 K at N128, ~1.3e-4 relative to T_rad~855 K. Moving only
the right boundary changes this small feature substantially, demonstrating
finite-domain sensitivity. The residual bump does not track the spatial
wiggle: right-far has a large residual offset but a smaller spatial wiggle.

Near the jump, the fixed limiter is 1/3. The optical depth per N128 cell is
`chi_R*dx=39.79`, while `kappa_P*dx=0.00184`. The fixed limiter follows the
published semi-analytic comparison setup; a variable-limiter comparison would
test a different physical closure. The paper's finest shock-grid example has
dx~1.5 cm, much smaller than this workstation's N256 dx=23.44 cm. Thus
under-resolution and finite boundary placement remain plausible contributors;
these controls do not prove causality or settle WOR-67's gas-spike accuracy.

## Reproduction and artifacts

The report and all CSV/PNG/JSON analysis artifacts are under
`/mnt/d/output/rad/wor10_trad_peak_20261104_final/index.html`. Current-source
uniform runs and saved effective inputs are in
`/mnt/d/output/rad/wor10_mach5_n{64,128,256}_20261104/`; right-boundary
controls are in `/mnt/d/output/rad/wor10_mach5_right_{near,far}_20261104/`.
The previous AMR reference is in
`/mnt/d/output/rad/validation_radiative_shock_amr1_20261102/`. The two manual
right-boundary runs used `mpirun -np 2 athena -i input.used -d output` with
`mesh/nx1=112 mesh/x1max=1250` and `mesh/nx1=144 mesh/x1max=2750`,
respectively. `input.used` contains the N128 baseline settings; these two
command-line overrides supply the effective domain. `run.log` records solver
convergence.

Primary paper: Zhang et al., *CASTRO: A New Compressible Astrophysical Solver.
II. Gray Radiation Hydrodynamics*, section 6.6,
https://ccse.lbl.gov/Publications/wqzhang/castro2.pdf.
