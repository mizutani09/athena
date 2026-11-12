# WOR-120: fourth-order radiation transport and force faces (local validation)

Branch: `wor-120-radiation-high-order`. Validation used the local PC only; XD2000 was not contacted.

## Scope

For `time/xorder=4`, the HLLC-FLD radiation face state now follows the same two-sweep face-average/face-point construction as hydro. A transverse Laplacian converts face-averaged `Er`, limiter `lambda`, and transport factor `a_r` to point states for the second Riemann sweep. A transverse correction then forms the face-averaged radiation advection flux and face `Er` used by the explicit radiation force. The implicit FLD diffusion operator and temporal radiation/hydro split remain at their existing orders; the results below do not claim fourth-order convergence of the complete FLD system.

Radiation transport continues to use the contact-upwind `Er`. The explicit force uses the arithmetic mean of the left and right radiation face states. The gas momentum flux subtracts the matching mean of the left and right radiation pressures, preserving the discrete gas+radiation pressure balance for constant limiter. The previous upwind choice shifted the force gradient by a cell near a stationary smooth peak.

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

### Independent radiation-force convergence

`inputs/radiation/athinput.nrfld_fourth_order_force` uses a one-dimensional smooth temperature pulse in a thin 3D periodic mesh, exact eight-point quadrature for the initial cell-averaged `Er`, `lambda=1/3`, and gas pressure chosen to keep total pressure uniform. The gas temperature equals `(Er/a)^(1/4)` so the initial implicit thermal solve is at equilibrium. `FLD_FORCE_PROFILE` is emitted on the first explicit RK stage. The reference force density is the analytic **cell average** `-[Er(x+h/2)-Er(x-h/2)]/(3h)`, which avoids confusing the cell-average force with a point derivative. `check_fourth_order_radiation_force.py` uses 16-point quadrature for the reference density and measures relative L1 error over the same `0<x<64 cm` interval at every resolution.

The base input is 64×16×16 with 16³ MeshBlocks. For `N=32,64,128,256`, override `mesh/nx1=N`, `mesh/x2min=mesh/x3min=-8192/N`, `mesh/x2max=mesh/x3max=8192/N`, `fld/force_diagnostic_gid=N/32`, and `time/xorder=2` or `4`. All eight local runs completed one step; each comparison samples the central positive-x MeshBlock.

| x1 cells | xorder=2 relative L1 | xorder=4 relative L1 |
| ---: | ---: | ---: |
| 32 | 2.86702e-1 | 2.58093e-1 |
| 64 | 8.67810e-2 | 1.77054e-2 |
| 128 | 2.28441e-2 | 1.71343e-4 |
| 256 | 5.78651e-3 | 1.08425e-5 |

The 128→256 measured rates are 1.981 and 3.982 for xorder=2 and xorder=4, respectively. The 64→128 xorder=4 pair exhibits stronger cancellation (rate 6.691); the finest pair is the useful asymptotic check. The gas+radiation force residual in these pressure-balanced runs remains near floating-point roundoff. This verifies the **spatial** force operator for a fixed limiter, not the variable-limiter or implicit FLD order.

## RSG gate and limitation

An isolated local 64³ RSG build used `--prob=rsg_star_in_box --coord=cartesian --eos=general/eos_table -nrmgfld -hdf5 -mpi --nghost=4`, the local Tomida material EOS, WOR-55 opacity table, and HSE profile. This build also overlaid pre-existing, uncommitted workspace changes needed by the local RSG setup, so it is not a clean-commit RSG integration test. The production 64³ input was overridden to `integrator=rk4`, `nlim=1`, and `xorder=4`. Before the density-cliff fallback, the initial primitive output contained velocities up to `1.66e5` code units near the stellar surface (`1.89e-4` for xorder=2), shrinking the first CFL step to `9.36e-5` rather than the imposed `0.1`. The local fallback eliminated velocities above 1 code unit in the initial output and restored the imposed first step of `0.1`.

With only the cell-average fallback, a single xorder=4 step of `1e-4` code time completes, but a `0.1` step makes a third-layer ghost cell negative (`Er=-1.03e-4`). Disabling the newly added fourth-order radiation flux correction does not remove this failure; replacing the table with constant opacity also leaves the negative `Er`. Instrumenting the explicit radiation flux showed a representative face with gas density at its `1e-13` floor on both sides, pressures `7.5e-6` and `1.03e-4`, gas velocities near `1e-8`, yet HLLC contact speed `-1.39e4` and radiation flux `-0.115`. The resulting RK stage drained a cell with initial `Er=7.75e-6` to `-4.28e-4`. The source was the *gas* face-point de-averaging, not the new radiation face correction.

With the complete gas face-state fallback, the production-table RSG 64³ run completes a `0.1` step, a two-cycle run to code time `5.0`, and a ten-cycle run to code time `33.0545` with `correct_ic=true`. Newton converges on all ten cycles; opacity diagnostics report zero domain excursions. The ten-cycle history gives mass `1.77791e9` initially and `1.77790e9` finally at printed precision, and maximum Mach number `0.534` at the end. The xorder=2 control completes its `0.1` step. A two-rank xorder=4 run without `correct_ic` also completes to code time `5.0`. These are startup smoke tests, not evidence of long-time RSG stability or improved core anisotropy. No production-duration xorder=4 RSG run has been validated.

The two-rank xorder=4 RSG case **with** `correct_ic=true` initially aborted in `MPI_Start` with `MPI_ERR_REQUEST`. `Mesh::CorrectMidpointInitialCondition` started a second FLD ghost receive through `StartReceivingSubset`, but skipped the matching `u_rad_fldbvar` send and receive completion. Restoring both calls lets the two-rank case complete to code time `5.0`; its printed mass, `Erad`, and maximum Mach number match the one-rank run (`1.77790e9`, `3.64241e9`, `0.13514`). After the symmetric force-face change, the two-rank case again completes to `5.0`, with maximum Mach `0.135147`.

Next gate: quantify where the local fallback activates, then compare initial force residuals and axis/diagonal velocities at 64³ and higher resolution before any long calculation. Check variable-limiter force behavior and RSG stability beyond startup.
