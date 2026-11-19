# FLD publication baseline status (2026-11-20)

This note supersedes the operational conclusions, but not the historical
measurements, in the 2026-11-09 Draft PR #3 documents. It records which source
is fixed for the validated protostellar-collapse application and which later
experiments must remain separate.

## Fixed collapse source

The following commits have exactly the same `src/` tree object,
`c4ab200383c7317a2feccc50aa9578d5cf71478d`:

| Purpose | Ref / commit |
| --- | --- |
| Source used by the c0-to-first-core XD run | `ff60f8f07842acb00e7753932bcfe30867e3a66e` |
| Reviewed solver and regression candidate | `codex/wor114-collapse-candidate-20261116`, `4f92c1fa877d26508c74f1d945c2d5d01a1e1693` |
| Collaborator handoff | `collab/protostellar-collapse-first-core-v1`, `6e658312bb504a90b2e385604193213c2bb07702` |
| Paper-figure source | `paper/protostellar-collapse-figures-v1`, `59ff33be1f9352cd5fa33c0f25f85105f941550b` |

These commits also have identical `configure.py`, `Makefile.in`, and
`inputs/radiation/athinput.protostellar_core_fld_mayer5` blobs. The immutable
collaborator tag is `protostellar-firstcore-v1` (peeled commit `6e658312...`).
The handoff branch adds instructions and smoke evidence without changing the
solver. The figure branch adds analysis products without changing the solver.

The production input used by the table-EOS run has SHA-256
`aad81db83087d27351db59ef70b9b14a68e8a2dc3e6fc164a4cb73e8d413e7e6`.
The EOS and opacity SHA-256 values are
`742a3456923d4c7056c3705accbcc793af3e0c79408a2f597a0cd9271b13ba43` and
`6c42445452bca6aeca1253f31ab049b2ff107dea3601f946b4c61c93db022f94`.
The exact binary and output hashes remain in WOR-115 and the paper-figure
manifest; this note does not relabel a documentation commit as a rerun.

## Qualification and claim boundary

The exact solver source has passed the WOR-95 diffusion and Jeans-AMR gates,
the WOR-106 six-way AMR restart gate, closed-box conservation, streaming, and
the mixed-frame fast matrix documented in
`fld_collapse_source_delta_20261116.md`. The full table-EOS, variable-opacity
AMR trajectory reached the first-core candidate with finite positive accepted
states, no final table-domain violation, bounded linear solves, and strict
outer Newton acceptance.

The publication-safe application claim is limited to stable three-dimensional
AMR evolution with the specified tabulated EOS and variable opacity through a
first-core candidate. It does not claim a matched Mayer et al. reproduction,
second-core formation, high-temperature opacity validation, or resolution
convergence of the observed Cartesian directional imprint. The radial and
central-evolution figures are suitable with those limitations; the final
Cartesian slice remains diagnostic unless a shape-accuracy claim is added.

## Later branches not folded into this source

| Work | Decision for the fixed non-magnetic collapse source |
| --- | --- |
| WOR-120 / WOR-122 RSG high-order and SMR work | Separate application and numerical-policy line. Do not merge into the collapse release merely because files overlap. |
| WOR-123 plateau-based inexact FGMRES | The implementation is opt-in and default-off. It is not required by the validated collapse trajectory and is not a reason to change this fixed source. |
| WOR-124 rotation/table-MHD work | Separate opt-in extension. Its magnetic-energy accounting and table-MHD gate are required for that extension, while the fixed application deliberately uses `B=0` and rejects table-MHD by default. Strong-field AMR-corner diagnostics do not invalidate the non-magnetic run. |
| WOR-115 post-AMR ambient-reset trial | Not used by the production trajectory and not imported. |

This separation prevents successful application-specific fixes from silently
changing the numerical method behind already qualified figures. A user who
needs table-EOS MHD must use a separately identified WOR-124 descendant and
its narrower validation record, not the `protostellar-firstcore-v1` tag.

## Solar application evidence and source boundary

WOR-116 has now closed the scientific application gate with **limited
inclusion**.  The supported claim is stable non-magnetic three-dimensional
radiative-convective transport with tabulated EOS, physical Rosseland
opacity, and grey FLD, including controlled 96 km states that bracket one
solar flux.  The completed 48 km branch does not reproduce the same flux, so
resolution convergence, a quantitatively observed solar atmosphere, and
non-grey spectral agreement are explicitly outside the claim.  The fixed
numbers, proposed caption, and report generator are in
`solar_application_gate_20261120.md` and
`tst/nrfld_validation/wor116_solar_application_gate.py`.

This scientific decision does not by itself qualify a solar release source.
The XD evidence used a hashed detached checkout with an uncommitted four-file
solar-boundary diff.  WOR-174 re-expresses the top-face Rosseland-opacity
reconstruction as a default-off, solar-PGen option on branch
`exp/wor174-solar-top-face-optin-20261120` (validated implementation commit
`2670facd11ea872d45565d091ad8da8b8709c771`; provenance correction
`f3bfffa4f`).  Its pure-math and 1/2-rank real-solver A/B tests pass, and the
omitted and explicit-false settings are byte-identical.  It has not yet passed
a long same-restart clean-source A/B equivalent to the selected XD evidence,
and it does not automatically certify every other file in the historical
detached diff.  It therefore remains a reviewable opt-in descendant rather
than part of the fixed collapse source or a new global FLD default.

## Remaining publication integration decisions

The collapse application source and figure provenance are fixed, and the
solar-convection scientific claim is now bounded by the limited WOR-116 gate.
The application choice, selected collapse figure pair, and caption boundaries
are fixed in `fld_application_selection_20261120.md`.  The audited figure
artifacts and regeneration scripts are now present on this review branch as
documentation-only descendants; the `src/` tree remains exactly
`c4ab200383c7317a2feccc50aa9578d5cf71478d`.
The corresponding execution, table, figure, and source hashes are collected
in `tst/nrfld_validation/publication_application_manifest_20261120.json`.
The broader FLD-paper baseline is not yet declared final because the author
must review the release candidate and decide whether the clean solar-boundary
opt-in is included after its remaining long A/B check.  That source decision
is distinct from accepting the already-audited solar result as a limited
application.
Draft PR #3 remains a historical 2026-11-09 audit at `6ad84f78`; it predates
the verified collapse-source alignment. It must not be merged or cited as the
collaborator release. A successor review should use the clean `4f92c1fa`
candidate (or a documentation-only descendant), retain the source hashes
above, and avoid importing RSG or table-MHD experiments without a separately
stated paper claim and regression matrix.

No fork `master` merge or official publication tag is authorized by this
status note. The collaborator tag is already the stable, supported collapse
entry point while author review of the full paper baseline continues.
