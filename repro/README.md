# st35gm03n07 Athena++/FLD reproducibility

The production entry point is `src/pgen/rsg_star_in_box.cpp`. It refuses to
start without the general EOS, an imported HSE profile, a user opacity table,
and `eos_table_clamp=false`.

1. Check `data/eos/MANIFEST` and place or regenerate the EOS at `data/eos/rsg_nata_eos.tab`.
   The current snapshot is numerical-validation only because it contains photon terms.
2. Check `data/opacity/MANIFEST` and place the checked gray opacity at
   `data/opacity/rsg_gray_opacity.h5`.
3. Generate the commissioning-grid initial state:

   ```bash
   python3 tools/rsg/make_hse_model.py \
     --eos data/eos/rsg_nata_eos.tab \
     --output data/ic/st35gm03n07_hse.dat \
     --grid 128 --density-floor 1e-12
   python3 tools/rsg/verify_hse_model.py \
     data/ic/st35gm03n07_hse.dat --grid 128 \
     --output data/ic/st35gm03n07_hse.verify.json
   ```

4. Build with `BUILD_JOBS=16 scripts/configure_rsg.sh`.
5. Commission with the 128³ variant:

   ```bash
   NPROC=27 scripts/run_rsg.sh \
     inputs/rsg/athinput.st35gm03n07_fld_128 runs/st35gm03n07_C0
   ```

6. Run the 256³ baseline only after the 128³ acceptance gates in
   `repro/acceptance.yaml` pass. Use the 512³ variant for factor-two spatial
   convergence after the baseline run is relaxed.

The compact 128³ and 512³ files are expanded by `scripts/run_rsg.sh`; the 256³
file is a normal Athena++ input. The historical odd-sized grids cannot initialize
the current NR-MG solver because its MeshBlocks must have power-of-two dimensions.
Every run captures the source state, build
configuration, input, and physics-table checksums before launching MPI.

The present radiation ghost-cell bath is still an engineering boundary. A
face-flux Marshak/vacuum condition and a production-quality adiabatic-interior
plus gray-atmosphere model remain acceptance blockers, as recorded in
`doc/freytag_star_in_box_validation.md`.
