# WOR-120: fourth-order radiation transport and force faces (local validation)

Branch: `wor-120-radiation-high-order`. Validation used the local PC only; XD2000 was not contacted.

## Scope

For `time/xorder=4`, the HLLC-FLD radiation face state now follows the same two-sweep face-average/face-point construction as hydro. A transverse Laplacian converts face-averaged `Er`, limiter `lambda`, and transport factor `a_r` to point states for the second Riemann sweep. A transverse correction then forms the face-averaged radiation advection flux and face `Er` used by the explicit radiation force. The implicit FLD diffusion operator and temporal radiation/hydro split remain at their existing orders; the results below do not claim fourth-order convergence of the complete FLD system.

The general-EOS fourth-order conserved-to-primitive conversion now retains the second-order cell average where the de-averaged point density drops below half of the cell-averaged density. This is a local density-cliff fallback, not a general positivity guarantee.

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

With the fallback, a single xorder=4 step of `1e-4` code time completes. At `0.1`, the run still fails: a third-layer ghost cell has `Er=-1.03e-4`. Disabling the newly added fourth-order radiation flux correction does not remove this failure. Disabling radiation transport changes the first error to an opacity-table domain violation at a ghost cell. Replacing the table with constant opacity still gives a negative `Er` after the ordinary step. Thus the surface/ghost-cell positivity issue predates or extends beyond the new radiation face correction. The xorder=2 RSG control completes one `0.1` step. No production-duration xorder=4 RSG run has been validated.

Next gate: identify the specific RK stage and face flux that makes `Er` negative, add a conservative positivity-preserving fallback across unresolved surfaces, and repeat one-step, MPI boundary, and short RSG runs before any long calculation.
