# WOR-120: fourth-order radiation transport and force faces (local validation)

Branch: `wor-120-radiation-high-order`. Validation used the local PC only; XD2000 was not contacted.

## Scope

For `time/xorder=4`, the HLLC-FLD radiation face state now follows the same two-sweep face-average/face-point construction as hydro. A transverse Laplacian converts face-averaged `Er`, limiter `lambda`, and transport factor `a_r` to point states for the second Riemann sweep. A transverse correction then forms the face-averaged radiation advection flux and face `Er` used by the explicit radiation force. The implicit FLD diffusion operator and temporal radiation/hydro split remain at their existing orders; the results below do not claim fourth-order convergence of the complete FLD system.

The general-EOS fourth-order conserved-to-primitive conversion now retains the second-order cell average where the de-averaged point density drops below half of the cell-averaged density. This is a local density-cliff fallback, not a general positivity guarantee.

The fourth-order hydro face-point conversion also reverts the **complete** gas state to the face average when point density or pressure differs by more than a factor of two from its face average. This prevents a component-wise floor from pairing atmosphere density with stellar pressure in the HLLC solver. It applies to hydro+FLD; MHD is outside this change.

## Reproduction

Configure an isolated copy with `python configure.py --prob=nrfld_adv --eos=adiabatic -nrmgfld -hdf5 -mpi --nghost=4 --cxx=g++ --mpiccmd=/usr/bin/mpicxx`, then build and run `inputs/radiation/athinput.nrfld_fourth_order_transport`. Override `mesh/nx1`, `mesh/nx2`, `mesh/nx3` together for resolution and `time/xorder` for the comparison. The test initializes exact cell averages of a smooth 3D periodic radiation mode and compares to analytic transport at speed `4 v0/3`, with diffusion and gas-radiation coupling disabled and a fixed limiter.

| Grid | xorder=2 L1 | xorder=4 L1 |
| --- | ---: | ---: |
| 16³ | 7.39478e-5 | 9.46959e-7 |
| 32³ | 1.77773e-5 | 5.33324e-8 |
| 64³ | 4.34320e-6 | 3.34480e-9 |

The xorder=4 rates are 4.150 and 3.995; xorder=2 rates are 2.056 and 2.033. The final radiation integral in the history output is constant to the printed precision. The 16³ case gives the same L1 error, `9.46959e-7`, with one or two MPI ranks. Rebuilding after the density-cliff fallback gave the same 16³ xorder=4 error.

An independent static gas+radiation pressure equilibrium test (`nrfld_advect_radiation_pulse`, fixed limiter, `rk4`) retained a near-zero force imbalance. At 64², the initial force residual for the selected core block was `3.38e-17` code units for both spatial orders. After a short step, xorder=4 had maximum speed `4.88e-8` cm/s, against `2.06e-10` cm/s for xorder=2; both are physically negligible for this test. This checks discrete cancellation, not force convergence order. Very high test opacity (`1e20`) can fail the implicit Newton solve with either order after multiple steps.

## RSG gate and limitation

An isolated local 64³ RSG build used `--prob=rsg_star_in_box --coord=cartesian --eos=general/eos_table -nrmgfld -hdf5 -mpi --nghost=4`, the local Tomida material EOS, WOR-55 opacity table, and HSE profile. The production 64³ input was overridden to `integrator=rk4`, `nlim=1`, and `xorder=4`. Before the density-cliff fallback, the initial primitive output contained velocities up to `1.66e5` code units near the stellar surface (`1.89e-4` for xorder=2), shrinking the first CFL step to `9.36e-5` rather than the imposed `0.1`. The local fallback eliminated velocities above 1 code unit in the initial output and restored the imposed first step of `0.1`.

With only the cell-average fallback, a single xorder=4 step of `1e-4` code time completes, but a `0.1` step makes a third-layer ghost cell negative (`Er=-1.03e-4`). Disabling the newly added fourth-order radiation flux correction does not remove this failure; replacing the table with constant opacity also leaves the negative `Er`. Instrumenting the explicit radiation flux showed a representative face with gas density at its `1e-13` floor on both sides, pressures `7.5e-6` and `1.03e-4`, gas velocities near `1e-8`, yet HLLC contact speed `-1.39e4` and radiation flux `-0.115`. The resulting RK stage drained a cell with initial `Er=7.75e-6` to `-4.28e-4`. The source was the *gas* face-point de-averaging, not the new radiation face correction.

With the complete gas face-state fallback, the production-table RSG 64³ run completes a `0.1` step, a two-cycle run to code time `5.0`, and a ten-cycle run to code time `33.0545` with `correct_ic=true`. Newton converges on all ten cycles; opacity diagnostics report zero domain excursions. The ten-cycle history gives mass `1.77791e9` initially and `1.77790e9` finally at printed precision, and maximum Mach number `0.534` at the end. The xorder=2 control completes its `0.1` step. A two-rank xorder=4 run without `correct_ic` also completes to code time `5.0`. These are startup smoke tests, not evidence of long-time RSG stability or improved core anisotropy. No production-duration xorder=4 RSG run has been validated.

The two-rank xorder=4 RSG case **with** `correct_ic=true` aborts in `MPI_Start` with `MPI_ERR_REQUEST`; the same build/input with `correct_ic=false` completes and an xorder=2 control also completes. This is a separate initial-condition-correction/MPI interaction that must be resolved before using that option in a parallel RSG run.

Next gate: quantify where the local fallback activates, then compare initial force residuals and axis/diagonal velocities at 64³ and higher resolution before any long calculation. Resolve `correct_ic`/MPI interaction or keep that option disabled with a justified initial-data error estimate. Check RSG behavior beyond startup.
