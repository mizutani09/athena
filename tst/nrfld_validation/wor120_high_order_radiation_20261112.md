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

### Variable-limiter force and full-scheme order

The separate `athinput.nrfld_variable_limiter_force` repeats the first-stage force measurement with the Levermore-Pomraning variable limiter, `opacity_mass_coeff=0.01 cm² g⁻¹`, and the RSG choice `radiation_pressure_in_riemann=false`. Its local limiter spans approximately 0.132–1/3. The reference is a 16-point quadrature of the analytic cell-averaged **product** `-lambda(R) dEr/dx`, where `R=|dEr/dx|/(kappa rho Er)` uses the analytic continuum profile. The test's initial `p_g+Er/3` profile is deliberately not a hydrostatic solution for variable lambda; only the first-stage radiation force is compared. All cases completed their one-step local runs. The N=512 case uses cubic 32³ MeshBlocks to keep the selected block covering the same `0<x<64 cm` comparison interval as the smaller cases.

| x1 cells | xorder=2 relative L1 | xorder=4 relative L1 |
| ---: | ---: | ---: |
| 64 | 1.21456e-1 | 3.15914e-2 |
| 128 | 3.88763e-2 | 1.08948e-2 |
| 256 | 1.18409e-2 | 2.74073e-3 |
| 512 | 3.33488e-3 | 6.74637e-4 |

The xorder=4 rates are **1.991 and 2.022** over the finest two refinement pairs; xorder=2 gives 1.715 and 1.828. Repeating xorder=4 with radiation pressure enabled in the Riemann solver gave identical first-stage force errors. Thus the higher-order Er face gradient improves the error constant but does not make the variable-limiter force fourth order. The cell-centered limiter uses a second-order centered Er gradient. In addition, multiplying one cell-centered lambda by the face-Er difference is only a second-order cell-average approximation to the product `lambda(x) dEr/dx`, even if lambda itself were exact. Achieving fourth-order variable-limiter force requires a high-order closure evaluation **and** high-order quadrature of that product; simply differentiating `lambda Er` would add an unphysical `Er grad(lambda)` term.

To reproduce the force table, use the same N-dependent mesh bounds and diagnostic gid listed for the fixed-limiter test with `athinput.nrfld_variable_limiter_force`. At N=512 additionally set `mesh/nx2=mesh/nx3=32`, `meshblock/nx1=meshblock/nx2=meshblock/nx3=32`, transverse bounds to ±32, and `fld/force_diagnostic_gid=8`. Pass the resulting logs as `--case N:ORDER:PATH` to `check_fourth_order_radiation_force.py` with `--input inputs/radiation/athinput.nrfld_variable_limiter_force`.

#### Candidate without more active cells or radiation unknowns

`prototype_fourth_order_variable_force.py` demonstrates a possible one-dimensional cell-average operator. A quartic reconstructed from the five existing Er cell averages `i-2..i+2` supplies point Er and its derivative at two Gauss points in cell `i`. The limiter is evaluated from those values and the local opacity, then the product `-lambda(R) dEr/dx` is integrated with the same positive Gauss weights. Active-cell count and the one-Er-unknown-per-cell formulation stay unchanged. For the same smooth Gaussian profile with variable density and limiter, the relative L1 error on `0<x<64 cm` is `1.55131e-6`, `9.97614e-8`, `6.26904e-9` at N=512, 1024, 2048; the final observed rates are **3.959, 3.992**. Coarser pairs are not uniformly asymptotic. This is an analytic Python prototype, not a modification or validation of the Athena++ FLD solver; it excludes tabulated opacity, 3D face integration, boundaries, positivity fallback, and nonlinear implicit iterations.

The five-cell stencil needs neighbor information two cells away. The current hydro `xorder=4` path already requires `NGHOST>=4`, so this candidate should fit within that allocated halo, subject to a complete 3D stencil audit. A build upgraded from `NGHOST=2` to 4 still uses more ghost-inclusive storage: for a 16³ MeshBlock, `(16+8)^3/(16+4)^3=1.728` per cell-centered array. For fourth-order implicit diffusion, keep one Er unknown and evaluate a high-order conservative residual matrix-free; the existing seven-point matrix/MG can be explored as a *preconditioner*, not as the final high-order operator. That requires a Krylov/Newton interface and may add temporary vectors, communication, and iterations. Keeping the present seven-point residual unchanged cannot yield fourth-order diffusion.

For the **complete smooth FLD system**, `xorder=4` applies to the explicit gas and Er transport where its local fallbacks do not activate. The implicit FLD diffusion flux in `NRFLD.cpp` uses adjacent-cell Er differences and face coefficients, so its spatial discretization is second order. The variable limiter also enters the explicit Er transport factor through the second-order cell-centered closure. Therefore the coupled spatial scheme is at best second order when these terms matter; it has not been shown to converge uniformly at second order for the full RSG model. A whole-run error-versus-resolution study would need a common physical time, controlled time error, and a converged 3D reference. The FLD solve is one backward-Euler update after all explicit RK stages, so full-run time error can be first order when the CFL timestep scales with cell width. Steep stellar surfaces and the local positive-state fallbacks can lower observed convergence further.

An isolated build from committed source `25df2c2d` with `--prob=nrfld_diff --eos=adiabatic -nrmgfld -hdf5 -mpi --nghost=4` ran the existing Gaussian diffusion convergence suite at `--xorder 4` (RK4 explicit stages). For N=16, 32, 64, the fixed-diffusion-number relative L1 errors were `9.13529e-3`, `2.36019e-3`, `6.04948e-4`, giving orders **1.953, 1.964**. After removing the leading backward-Euler time error at each grid using two timestep sequences and their measured `sum(dt²)`, the spatial errors were `6.28970e-3`, `1.61835e-3`, `4.18716e-4`, giving orders **1.958, 1.950**. At fixed N=32, successive timestep differences gave temporal orders **0.989, 0.993**. All solves reached the common final time and passed the Newton tolerance checks. This is direct evidence for second-order *diffusion-only* spatial accuracy and first-order implicit time accuracy on the current branch; it is not a measured whole-RSG convergence rate. The reusable runner now accepts `--xorder 4` and allows the small actual RK CFL timestep drift in its time-error extrapolation.

Reproduction after building the `nrfld_diff` MPI binary with `--nghost=4`: `python3 tst/nrfld_validation/diffusion_convergence.py --binary /path/to/bin/athena --xorder 4 --max-cores 2 --no-plots`.

Local raw artifacts: `/tmp/wor120_diffusion_xorder4_v2_20261112/` for the diffusion suite and `/tmp/wor120-clean-final/variable_n*_o*.log` for the variable-limiter force profiles. They were generated without contacting XD2000.

## RSG gate and limitation

An isolated local 64³ RSG build used `--prob=rsg_star_in_box --coord=cartesian --eos=general/eos_table -nrmgfld -hdf5 -mpi --nghost=4`, the local Tomida material EOS, WOR-55 opacity table, and HSE profile. This build also overlaid pre-existing, uncommitted workspace changes needed by the local RSG setup, so it is not a clean-commit RSG integration test. The production 64³ input was overridden to `integrator=rk4`, `nlim=1`, and `xorder=4`. Before the density-cliff fallback, the initial primitive output contained velocities up to `1.66e5` code units near the stellar surface (`1.89e-4` for xorder=2), shrinking the first CFL step to `9.36e-5` rather than the imposed `0.1`. The local fallback eliminated velocities above 1 code unit in the initial output and restored the imposed first step of `0.1`.

With only the cell-average fallback, a single xorder=4 step of `1e-4` code time completes, but a `0.1` step makes a third-layer ghost cell negative (`Er=-1.03e-4`). Disabling the newly added fourth-order radiation flux correction does not remove this failure; replacing the table with constant opacity also leaves the negative `Er`. Instrumenting the explicit radiation flux showed a representative face with gas density at its `1e-13` floor on both sides, pressures `7.5e-6` and `1.03e-4`, gas velocities near `1e-8`, yet HLLC contact speed `-1.39e4` and radiation flux `-0.115`. The resulting RK stage drained a cell with initial `Er=7.75e-6` to `-4.28e-4`. The source was the *gas* face-point de-averaging, not the new radiation face correction.

With the complete gas face-state fallback, the production-table RSG 64³ run completes a `0.1` step, a two-cycle run to code time `5.0`, and a ten-cycle run to code time `33.0545` with `correct_ic=true`. Newton converges on all ten cycles; opacity diagnostics report zero domain excursions. The ten-cycle history gives mass `1.77791e9` initially and `1.77790e9` finally at printed precision, and maximum Mach number `0.534` at the end. The xorder=2 control completes its `0.1` step. A two-rank xorder=4 run without `correct_ic` also completes to code time `5.0`. These are startup smoke tests, not evidence of long-time RSG stability or improved core anisotropy. No production-duration xorder=4 RSG run has been validated.

The two-rank xorder=4 RSG case **with** `correct_ic=true` initially aborted in `MPI_Start` with `MPI_ERR_REQUEST`. `Mesh::CorrectMidpointInitialCondition` started a second FLD ghost receive through `StartReceivingSubset`, but skipped the matching `u_rad_fldbvar` send and receive completion. Restoring both calls lets the two-rank case complete to code time `5.0`; its printed mass, `Erad`, and maximum Mach number match the one-rank run (`1.77790e9`, `3.64241e9`, `0.13514`). After the symmetric force-face change, the two-rank case again completes to `5.0`, with maximum Mach `0.135147`.

The committed source was also built independently from `git archive HEAD` with MPI enabled. The N=64, 128, 256 force cases at both spatial orders reproduced the table above exactly. A two-rank, xorder=4, `correct_ic=true` **uniform** FLD case completed one step without an MPI error. Applying `correct_ic` to the very high-opacity Gaussian force-test initial condition instead failed the initial nonlinear gas solve (`backtracking_exhausted`) on both ranks; that case does not test the communication fix. The nonuniform production-table RSG startup above is the relevant MPI integration check.

Next gate: quantify where the local fallback activates, then compare initial force residuals and axis/diagonal velocities at 64³ and higher resolution before any long calculation. Check variable-limiter force behavior and RSG stability beyond startup.

## Follow-up: three-dimensional variable-limiter force

Branch `wor-120-force-3d` adds an **opt-in** `fld/fourth_order_radiation_force=true` source for smooth, uniform, three-dimensional Cartesian meshes with `xorder=4`, at least four ghost cells, and `radiation_pressure_in_riemann=false`. It reconstructs the existing cell averages of `Er`, density, and mass opacity on a 5³ stencil, evaluates the Levermore-Pomraning limiter and `-lambda grad(Er)` at eight cell quadrature points, and falls back to the old face force if a reconstructed state is invalid. There are no new active cells or persistent radiation unknowns. A fourth-order-small face-gradient correction anchors the constant-limiter component to the gas pressure face flux and preserves its exact discrete cancellation.

`athinput.nrfld_force_3d` and `check_fourth_order_force_3d.py` compare the first-stage force to an independent 8³ Gauss reference. For opacity `0.00025 cm²/g`, the limiter spans a strong variable regime. The profile has a three-dimensional periodic Er mode; the second series adds an exactly averaged 30% density mode, making the extinction coefficient vary spatially too.

| Grid | Fixed density, high-order relative L1 | Variable density, old relative L1 | Variable density, high-order relative L1 |
| ---: | ---: | ---: | ---: |
| 16³ | 1.38179e-3 | 3.92118e-2 | 1.37606e-3 |
| 32³ | 3.39933e-5 | 9.90273e-3 | 3.42617e-5 |
| 64³ | 2.32125e-6 | 2.46729e-3 | 2.35665e-6 |

The 32→64 orders are 3.872 for fixed density and **3.862** for variable density, against **2.005** for the old variable-density force. The high-order runs had zero reconstruction fallbacks. Every one of the 4096 force rows for a selected 32³ block was bitwise identical with one and two MPI ranks. All one-step states remained finite with positive density and gas pressure; the implicit solve converged. With a fixed limiter and `p_g+Er/3=constant`, the 32³ and 64³ discrete gas-plus-radiation force residuals are 5.08e-14 and 9.65e-14 relative L1, respectively. The 16³ pressure-balance test has a small 1.19e-7 coarse-grid residual.

Reproduction: build with `--prob=nrfld_advect_radiation_pulse --eos=adiabatic -nrmgfld -hdf5 -mpi --nghost=4`, run `athinput.nrfld_force_3d` at 16³, 32³, and 64³ with MeshBlocks of half the domain width. For the variable-density series, set `problem/force_3d_density_amplitude=0.3` and `problem/opacity_mass_coeff=0.00025`; pass the resulting logs to `check_fourth_order_force_3d.py --density-amplitude 0.3 --opacity-mass-coeff 0.00025`. Local logs: `/tmp/wor120_force3d_density_n*.log` and `/tmp/wor120_force3d_legacy_density_n*.log`.

### RSG startup and gravity balance

An isolated local RSG build with the existing Tomida EOS, WOR-55 opacity, and HSE profile completed a 64³ RK4/xorder=4 single step to code time 0.1 with and without the new force. The new-force run also completed on two MPI ranks. Its selected central MeshBlock (`gid=42`) gave 4096 bitwise-identical initial vector-force diagnostic rows between one and two ranks, with 10 of 4096 high-order force evaluations using the safe fallback. Diagnostic coordinates put all ten at radii 811.7–847.8, across the stellar surface at radius 830, and **none in the core**. The final one-step state retained positive density, gas pressure, and radiation energy; the Newton solve converged in five iterations. The one-rank wall time rose from 20.37 s to 25.94 s in this small run (about 27%). This includes initialization and I/O, so it is not a kernel-only timing.

Adding the RSG potential's analytic cell-center gravity acceleration to the recorded gas-flux and radiation forces gives a central-block, inside-`Rcore` vector-force relative L1 residual of 3.9824e-3 for both schemes. The largest absolute residual there is 2.4477e-3 code force-density units at radius 82.0 code length units, directed radially outward. The residual is essentially unchanged because this block is already close to the diffusion limit (`lambda≈1/3`): the summed relative difference between old and high-order radiation force is 6.1e-14, and the largest pointwise relative difference is 2.9e-9. Thus the new variable-limiter force fixes the smooth spatial-order defect but does **not** cure the present RSG core imbalance. This gravity audit uses the single-precision HDF5 initial density and the exact pgen potential formula; it is an initial-stage diagnostic, not a long-time stability result. It is reproducible with `check_rsg_initial_force_balance.py` and the logs in `/tmp/wor120_rsg_force_{baseline,high4}.log`.

A two-rank high-order RSG startup continued for ten cycles to code time 33.0540 in 105.75 s wall time. All ten nonlinear steps converged; the printed mass stayed at 1.77790e9 code units and maximum Mach at the final step was 0.533642. This is consistent with the earlier old-force ten-cycle startup (maximum Mach about 0.534), and does not establish longer-term stability or reduction of the axial flow. Initial force imbalance and longer directional velocity growth remain the RSG gates.

At 128³, a single four-MPI-rank step to code time `1e-4` also completed in 49.06 s, with Newton converged in three iterations and no opacity-domain excursions. Initial and final HDF5 states had positive density, gas pressure, and Er. The corresponding positive-x/negative-y/positive-z core-octant block (`gid=338`) contained 597 cells inside `Rcore`, all with zero high-order force fallbacks. Its gas+radiation+gravity relative L1 residual was **5.3442e-4**, versus 3.9824e-3 at 64³ (78 core cells); the ratio is 7.45 for a factor-two grid refinement, an apparent order of 2.90 over this one pair. The largest absolute core residual fell from 2.4477e-3 to 3.7124e-4. Both blocks cover the complete same core octant. This supports a resolution-sensitive discrete core imbalance, while one refinement pair is insufficient to establish an asymptotic order or isolate gas pressure from pointwise gravity as the sole cause.
