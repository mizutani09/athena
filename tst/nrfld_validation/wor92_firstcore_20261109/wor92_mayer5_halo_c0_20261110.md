# Single-binary Mayer §5 ideal EOS: roundoff halo diagnostic

The physical setup is the [pressure-equilibrium gas and 14 K background-radiation test](wor92_mayer5_coldrad_c0_20261110.md), with the background gas/radiation reset after implicit FLD so the accepted state retains its prescribed temperature. In the failed controls, all reported negative radiation values were first-layer MeshBlock halos of magnitude at most about `3e-17` code; the c1 active 3D radiation minimum was `0.29064393` code. A negative *active* state remains a fatal error.

The isolated source change in `NewtonRaphson::ProlongateBoundary` extends the existing opt-in `ATHENA_WOR92_POSITIVE_NR_GHOST=2` mode. After boundary exchange, it clamps only **halo** radiation values in `[−64 ε max(1,max_active|Er|),0)` to `TINY_NUMBER`, counts them, and logs the most negative value and tolerance. Larger negative values and non-finite values are still caught by the original NR-FLD state check. The change is intended to handle roundoff-sized interpolation undershoot without masking physical negativity. It is not a general positivity scheme.

XD M-debug isolated build **339312** completed `0:0` in 1:48. Its binary SHA-256 is `26e6191629662816427eb144bcf609d2249bc0cf653ead4c06b6c658c8391bb9`; the pgen and NR boundary source hashes are in the build provenance. M-test-a preflight **339313** completed cycle 10 with Newton/FGMRES convergence, no accepted pressure floor, and Jeans cells ≥16. Its per-cell halo log showed values near `−3e-17` against a block-local roundoff tolerance around `5e-14`. The mass audit below invalidated this checkpoint for the long run; the corrected path is documented in [the post-reset sync report](wor92_mayer5_postsync_c0_20261110.md).
# 2026-11-10: mass-conservation audit

The 10-cycle halo-roundoff preflight (job 339313) completed with converged
Newton solves and positive gas pressure, but **failed the physical mass audit**:
gas mass fell from 2.936555728050246e33 g to 2.858871852399791e33 g
(2.6454%) while the mesh coarsened from 960 to 120 blocks. Do not use that
restart for the first-core run. The no-derefinement control 339318 still lost
0.4538% by c2, and the fixed-grid control 339319 lost 0.9135% by c2. The
control without the post-implicit background sink (339327) conserved mass to
15 digits over the same two steps. Radiation-pressure-free Riemann control
339320 still lost 0.917%. Thus derefinement and the Riemann radiation-pressure
term do not explain the loss. The accepted-state background reset occurs after
the existing FLD boundary synchronization in main.cpp; a dedicated post-reset
boundary synchronization is being tested.
