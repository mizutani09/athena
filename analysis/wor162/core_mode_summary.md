# WOR-162 central-flow decomposition

The long surface-L3/central-L6 run (XD job 342602) fails because of a
localized, predominantly axisymmetric spin-up inside 20 Rsun.  It is not
primarily a low-order nonaxisymmetric instability.

At 0.55 dynamical time, the equatorial-slice decomposition gives:

| Region | fitted Omega / target Omega | tangential fraction of perturbation energy | axisymmetric fraction of tangential energy | nonaxisymmetric fraction of tangential energy |
| --- | ---: | ---: | ---: | ---: |
| r < 5 Rsun | 4013.3 | 0.997725 | 0.993252 | 0.006748 |
| r < 10 Rsun | 2120.7 | 0.997666 | 0.993450 | 0.006550 |
| r < 20 Rsun | 925.9 | 0.997664 | 0.993441 | 0.006559 |

The density m=1 amplitude within 20 Rsun grows to 0.00423, and the radial
velocity m=1 coherence grows to 0.160, but these remain secondary to the
axisymmetric rotation profile.  The in-plane translation fraction is below
7e-5.  The annular decomposition closes to an absolute energy-fraction error
of 1.1e-13.

The early response is strongly resolution dependent:

| Comparison | Region | time / t_dyn | coarse excess Omega / target | fine excess Omega / target | suppression |
| --- | --- | ---: | ---: | ---: | ---: |
| central L5 to L6 | r < 20 Rsun | 0.10 | 945.9 | 12.24 | 77x |
| central L6 to L7 | r < 5 Rsun | 0.02127 | 2.171 | 0.03068 | 71x |
| central L6 to L7 | r < 20 Rsun | 0.02127 | 0.2679 | 0.005875 | 46x |

At 0.02 dynamical time, the root64-L6 and surface-L3/central-L6 excess
rotation within 20 Rsun differs by about 20% (0.268 versus 0.222).  This is
small compared with the L6-to-L7 resolution effect, so a root64-L7 extension
to 0.1 dynamical time is an efficient discriminating calculation before a
costly surface-refined L7 run.

Raw decompositions are in `core_mode_342916/` and
`core_resolution_342924/`.  Both analysis jobs completed successfully.
