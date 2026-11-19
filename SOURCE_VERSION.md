# FLD publication source version

Status: **review candidate, not an official release**  
Recorded: 2026-11-20 JST

## Immutable solver identity

- Solver `src/` tree: `c4ab200383c7317a2feccc50aa9578d5cf71478d`
- Full source used by the selected collapse execution:
  `ff60f8f07842acb00e7753932bcfe30867e3a66e`
- Collaborator branch: `collab/protostellar-collapse-first-core-v1`
- Immutable collaborator tag: `protostellar-firstcore-v1`
- Publication review branch: `codex/wor114-collapse-candidate-20261116`
- Review PR: GitHub Draft PR #4

Documentation-only commits on the review branch may change its `HEAD` without
changing the immutable solver tree above.  Cite both a pinned full commit and
the `src/` tree.  Do not cite fork `master` as the FLD paper source.

No official publication tag has been created and no merge to `master` is
authorized by this file.

## Selected application

The primary paper application is the non-rotating, non-magnetized,
three-dimensional Cartesian-AMR protostellar collapse with tabulated H/He EOS
and variable Semenov opacity through a first-core candidate.  Its input,
external-table, binary, output, and figure hashes are in
`tst/nrfld_validation/publication_application_manifest_20261120.json`.

Start with `COLLABORATOR_START_HERE.md` for a portable one-step acceptance
test and the fixed input.  The exact input SHA-256 is
`aad81db83087d27351db59ef70b9b14a68e8a2dc3e6fc164a4cb73e8d413e7e6`.
External tables are not bundled in public Git.

The acceptance test was repeated from review commit `2fe5102c` on
2026-11-20 with two MPI ranks.  It reproduced the recorded binary hash,
converged one Newton solve with two bounded-inexact FGMRES accepts, and had a
gas-mass relative change of `-3.22e-15`.  The compact machine-readable record
is `examples/protostellar_firstcore_v1/review_head_repeat_20261120.json`.

The selected paper figures, caption drafts, and claim boundary are collected
in `tst/nrfld_validation/publication_applications_20261120.html` and
`doc/fld_application_selection_20261120.md`.

## Qualified numerical path

The recorded collapse configuration uses Newton--FGMRES with MG
preconditioning, the conservative supported AMR interface, shared face
coefficients, assembled global Jv, table EOS/opacity, regrid/restart, and a
bounded-inexact linear true-residual cap of `1e-7`.  Outer Newton acceptance is
separate and remains strict.  The exact source has evidence for diffusion,
Jeans AMR, local exchange, periodic conservation, streaming, coupled AMR
restart, and the mixed-frame fast matrix, summarized in
`doc/fld_collapse_source_delta_20261116.md`.

## Known limits and separate branches

- The application does not demonstrate a matched Mayer et al. reproduction,
  second-core formation, high-temperature opacity validity, or resolution
  convergence of its Cartesian angular imprint.
- The ambient reset is a deliberate energy source/sink; full-domain total
  energy is not conserved by design.
- Semenov Planck extinction is used as an absorption proxy.
- The solar local-convection result is a secondary limited application.  Its
  historical evidence and the clean default-off WOR-174 option are tracked
  separately.  The clean option passed its mature same-restart A/B in XD job
  343705, but neither it nor the historical solar diff is silently part of
  this collapse source.
- WOR-123 inexact-stagnation, WOR-124 table-MHD/strong-field work, and
  WOR-120/122 RSG high-order/SMR work are separate opt-in research lines and
  are not automatically integrated.
- A new problem generator, boundary condition, geometry, EOS/opacity table,
  non-cubic mesh, rotation, or magnetic field requires a new scoped gate.
- Do not enable `ATHENA_NRFLD_LEGACY_INTERFACE=1` for a supported conservative
  AMR claim.

## Finalization checklist

Before an official paper release, the author must review Draft PR #4, decide
whether the now-qualified clean WOR-174 solar option belongs in the release
source, select the full commit to tag, and approve the tag and any merge.  The
WOR-174 result recommends the option only for the tested solar application,
not as a global FLD default.  Future upstream patch preparation is WOR-117 and
is not a paper release prerequisite.
