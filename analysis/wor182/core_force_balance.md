# WOR-182 core force-balance audit

## Interpretation

The tabulated-EOS FLD map splits the SCF total pressure into material pressure
and LTE radiation pressure.  Therefore the diagnostic

`material pressure acceleration + gravity + centrifugal acceleration`

is not a hydrostatic residual.  In the optically thick stellar interior it is
expected to be comparable to the omitted radiation-pressure acceleration.
For the present model its mass-weighted L2 ratio is about 0.35.

The physically relevant diagnostic is

`material pressure acceleration + radiation acceleration + gravity +`
`centrifugal acceleration`.

It must be sampled during an evolution step, when the FLD face gradients and
flux limiter are current.  A restart-only (`nlim=0`) slice does not refresh
those face states and must not be used to infer the full force balance.

## Existing long runs

| model | interval | full-force L2, start | full-force L2, end |
|---|---:|---:|---:|
| corotating, core-only relaxation (`350292`) | 300.61--470 | 0.00460 | 0.0261 |
| nonrotating, core-only relaxation (`350293`) | 150.24--300 | 0.00141 | 0.00775 |

The global full-force imbalance grows together with the radial mode but is far
smaller than the material-only number.  The tracer audit also finds a
mass-weighted stellar fraction of exactly one inside 100 Rsun, so tracer
dilution is not driving the core mode.

## Instrumentation decision

History output now includes full-force error and norm restricted to
`r < 1.5 R_support` (97.47 Rsun for the current model).  These quantities are
evaluated during integration and are the acceptance diagnostic for subsequent
core-timescale and core-model comparisons.

Do not construct a more aggressive core cut solely from the material-only
0.35 value.  First compare `relaxation_core_tau = 5, 2, 1` at a common restart
and then use the localized full-force history to decide whether the underlying
equilibrium model requires modification.
