# WOR-28: NR-FLD radiation output (2026-11-04)

The existing `Er`/`Fr` branch in `src/outputs/outputs.cpp` belongs to
`NRRadiation`, not to the separately enabled NR-FLD solver. A saved NR-FLD
`variable=prim` HDF5 file previously contained only `rho`, `press`, and three
velocities. This change adds the current `FLD::u_rad` to `prim` and `cons` as
cell-centred scalar `Er`; `variable=Er` also writes just that field. The HDF5
`prim`/`cons` dataset variable count is incremented explicitly. The output is
a shallow view of the current state, so it allocates no additional full-grid
array and does not modify the solver or restart format.

`Er` is the evolved radiation energy density in **code energy-density units**.
The FLD source and explicit transport use the mixed-frame formulation through
first order in velocity/light speed: `u_rad` is the lab-frame evolved energy
variable, while opacity/closure represent comoving material properties. The
output is the stored `u_rad` without a frame or unit conversion. In the static
equilibrium pgen, physical `Erad = Er * egas_unit` (the test input uses
`egas_unit=4.28129270484549e14 erg cm^-3`). The existing radiation transport
module's `Er`/`Fr` output should not be confused with this FLD field.

## Flux decision

Do **not** add a generic `Fr` output yet. `FLD::u_rad_flux[3]` is the
face-centred **explicit Godunov transport** flux written during a hydro stage;
it is neither the comoving diffusive FLD flux nor a complete synchronized
lab-frame radiation-energy flux at output time. The implicit diffusion flux
is computed from `c lambda/sigma_R` and neighbouring `Er` values in the NR
operator but is not stored as a complete time-aligned face field. Giving
`u_rad_flux` the name `Fr` would misidentify its physics and time location.

A future opt-in flux output should expose separately named quantities:

1. `Fr_com_diff_x/y/z`: face-centred comoving-closure diffusion flux,
   reconstructed with the same limiter, opacity and boundary-face formula as
   the implicit solve, with its stage/step time recorded.
2. `Fr_lab_x/y/z`: the full lab-frame energy transport only after the explicit
   Godunov flux, implicit diffusion, and O(v/c) work/enthalpy terms can be
   combined without double counting. Test zero-velocity diffusion, uniform
   `Er` advection, and Marshak boundary luminosity against an energy ledger.

An additional float HDF5 `Er` costs approximately four bytes per active cell
before HDF5 metadata/compression (eight bytes when built with `-h5double`).
The shallow OutputData view adds no persistent or per-output field copy.

## Verification

Built with `--prob nrfld_advect_radiation_pulse --eos adiabatic --flux lhllc
-nrmgfld -hdf5 -mpi`. Single-rank, one-step 16x16x4 static equilibrium and
64x4x4 moving radiation pulse completed. Both `prim` and standalone `Er`
HDF5 contain `Er`; the standalone field is bitwise equal to `prim/Er`.
`Erad/Er` from the pgen's independent user output agrees with `egas_unit`
within single-precision output rounding (relative scatter below `1.7e-7`).
Static user-output datasets match the no-`Er` output control bitwise for all
three saved snapshots. Raw inputs, commands and logs:
`/mnt/d/output/rad/wor28_primitive_20261104/` (`verified`, `advect`,
`er_only`; root `control.log` is the no-`Er` comparison).

For example, `variable=prim` now includes `Er` in `VariableNames`; a smaller
`variable=Er` HDF5 output puts it in the `hydro` dataset, following the
existing single-variable Athena++ writer convention.
