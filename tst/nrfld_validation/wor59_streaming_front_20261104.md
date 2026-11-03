# WOR-59: streaming-front lag and broadening (2026-11-04)

## Result

The 128×8×8 slab reproduces the previous 128×32×32 streaming profile
**exactly** for `chi_R=1e-7 cm^-1`, `dt=dx/(20c)`. This reduces memory and
run time for an independent time/space sweep without changing the tested
1D-like solution. All cases below converged with FGMRES, reached 2 ns, and
were plotted in `/mnt/d/output/rad/wor59_convergence_20261104/index.html`.
No curve or reference position was shifted.

| Case (`chi_R=1e-7 cm^-1` unless stated) | 50% error [cm] | 10% leading error [cm] | 10–90% width [cm] |
| --- | ---: | ---: | ---: |
| N=128, `dt=dx/(2c)` | -8.429 | +1.808 | 20.504 |
| N=128, `dt=dx/(20c)` | -2.871 | +1.967 | 9.259 |
| N=128, `dt=dx/(100c)` | -1.949 | +2.192 | 7.302 |
| N=256, physical `dt` fixed to N=128 `dx/(20c)` | -2.602 | +1.122 | 7.555 |
| N=512, same fixed physical `dt` | -2.758 | +0.446 | 6.743 |
| N=256, `dt=dx/(20c)` | -1.918 | +1.249 | 6.183 |
| N=512, `dt=dx/(20c)` | -1.469 | +0.619 | 4.162 |
| N=128, `chi_R=1e-4`, `dt=dx/(20c)` | -3.086 | +1.912 | 13.918 |

The fixed-`dt` N=512 position appears worse than N=256 because the initial
step intersects different cells. At `x0=10 cm`, its discrete initial 50%
position is already `-0.039 cm` from x0, while the N=128/256 initial bias is
`+0.156 cm`. A controlled series with the discontinuity on a cell face at
`x0=10.15625 cm` has zero initial interpolation bias for all three grids; its
fixed-`dt` 50% errors improve from `-3.027` to `-2.759` to `-2.719 cm` at
N=128/256/512. This changes the *initial condition*, not a post-hoc shift of
the plotted solutions.

At fixed N, reducing `dt` mainly improves the center lag and width. At fixed
physical `dt`, refining `dx` mainly reduces the 10% leading-edge error and
width; the remaining center lag approaches a time-discretization floor.
Refining both improves all three metrics. The front-zone median `lambda*R`
is `0.99999955` in the thin N=128 `dx/(20c)` case, so the limiter is already
at its streaming asymptote there. Increasing the right-boundary distance by
50 cm changes the 50% position by only `0.0044 cm`. Tightening both Newton
and FGMRES tolerances from `1e-8` to `1e-10` leaves the stored profile
unchanged. These controls point to the finite-step, finite-grid implicit FLD
discretization as the dominant cause of the measured lag and broadening;
they do not identify a solver-convergence or boundary-location defect.

The integrated energy relative to an ideal translated step is lower by
`2.766 cm` in the N=128 thin `dx/(20c)` run, close to its `2.871 cm` center
lag. This is a *transport comparison*, not a violation of the finite-volume
conservation law: the problem imposes a radiation energy at the left boundary,
not an analytic free-streaming flux. The initial FLD flux through a perfectly
flat radiation plateau is zero; at 2 ns the leftmost cell's diagnostic flux
is about `0.976 c E`. This and the front's finite width distinguish the
finite-grid FLD solution from a rigidly translated step. The 10% leading
edge remains ahead of `x0+ct` at finite resolution, but its error drops from
`1.967` to `0.619 cm` on joint N=128→512 refinement.

## Isolated numerical variants

Two deliberately isolated builds were compared to the same N=128 thin
`dx/(20c)` baseline. Replacing arithmetic face radiation energy by the larger
adjacent state reduced 50% lag from `2.871` to `0.937 cm`, but increased the
10–90% width from `9.259` to `19.089 cm`, the 10% leading error from `1.967`
to `8.600 cm`, and sharp-step L1 error from `0.03965` to `0.06053`. This
variant was rejected. Raising the left ghost energy by `3.95e-4` changed the
50% error by only `0.0017 cm`, so a small Dirichlet-value adjustment does not
address the lag. Neither experimental source change entered the repository;
their inputs, logs, outputs, binary hashes, source patches, and comparison
plot are under the sweep report's `experiments/` directory.

## Repeat and interpretation

```text
python3 tst/nrfld_validation/streaming_front_convergence.py --max-cores 4
```

The script builds an isolated streaming executable, runs the controls
sequentially under `/mnt/d/output/rad`, writes per-case input and full log,
and makes `index.html` and `summary.json`. Nine automated trend/control
checks passed on the reported data. A calculation PASS indicates solver and
output integrity; the separate trend result indicates refinement behavior.
The existing four-condition `streaming_validation.py` and its movie remain the
standard visual benchmark. For a sharper 2 ns front on this workstation,
`N=512` with `dt=dx/(20c)` reduces the 50% lag to `1.47 cm` and width to
`4.16 cm`; further accuracy requires further refinement or a different
radiation transport formulation. The sharp translated step is a kinematic
reference, not a complete analytic FLD profile.
