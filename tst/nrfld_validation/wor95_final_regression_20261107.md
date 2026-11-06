# WOR-95 local regression, 2026-11-07 JST

Source: `209e22398` (`nr_mgrad_dev` snapshot), isolated worktree
`/home/kosuke/simulation/athena_wor95_validation`, branch
`codex/wor95-validation-20261107`. No main or development branch was changed.

## Jeans AMR

Two one-cycle, two-rank runs used the same isolated `protostellar_core_fld`
binary (`c4619ec2630088837696b1da2cb3d3612c040c4dd06a8426b693da47ae7cbab4`),
the same Mayer5 input except `problem/N_J` and output ID, a 32³ root, 16³
MeshBlocks, one permitted refinement level, and `refinement_mode=jeans`.
The root cell width was 375 au. At the unrefined initial state the minimum
Jeans length was 15.2334 root cells, so the code's `cells < N_J` condition
predicts no refinement for 4 and refinement for 32. The upper level bound
prevents further refinement even though the refined minimum is 30.4667 cells.

| N_J | Leaf blocks | Created | Actual max level | Finest cell (au) | Solve dt | Newton max iterations | FGMRES max iterations / true relative residual |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8 | 0 | 1 | 375 | 4.1446358997 | 4 | 9 / 1.1752e-8 |
| 32 | 64 | 56 | 2 | 187.5 | 2.0723179499 | 4 | 11 / 1.3878e-8 |

All stage-0 Newton records said `converged`; maximum final L2/max norms were
`3.57e-9/9.98e-9` and `1.11e-8/1.99e-8`. All FGMRES records had
`breakdown=0` and true `||b-Ax||/||b|| <= 2e-8`. The preliminary input failed
before evolution because output6 had neither positive `dt` nor `dcycle`;
the failure log is retained and the input was corrected before these runs.

## Gaussian FLD diffusion

The existing `diffusion_convergence.py` was run against the same source in
isolated builds, first with MG and then with the added `--linear-solver fgmres`
validation option, reusing the verified binary SHA-256
`13f6152e41802df5e7798316e5c09a1d941a7d16c7f22e64446689231c118a56`.
Both modes produced the same reported errors and passed:

| Measure | Orders |
| --- | --- |
| Fixed diffusion number, dx 16→32→64 | 1.9702, 2.0007 |
| Time-extrapolated spatial error | 1.9851, 2.0048 |
| Fixed N=32, dt sweep 40→80→160→320 steps | 0.9890, 0.9945 |

The fixed-diffusion-number series also gives 0.9851 and 1.0004 when
interpreted against dt, but coupled dx/dt refinement alone does not prove
temporal order. The independent fixed-grid sweep does. The requested dt values
were observed: at N=16, 40 steps `5e-4`; N=32, 160 steps `1.25e-4`;
N=64, 640 steps `3.125e-5`. One N=64, 1280-step run had a final
`4.72e-16` roundoff step after 1280 full `1.5625e-5` steps; the script
checks its total elapsed time. Every active stage-0 Newton solve met
`final_l2, final_max <= 1e-11`. In FGMRES mode, 5,280 solves were checked
against their reported true residual and `breakdown=0`; maximum iterations
was 6 and maximum true relative residual was `9.9852e-11` against `1e-10`.
This agrees with the earlier approximate second spatial and first temporal
orders recorded in WOR-95; no numeric old-run artifact was available for a
pointwise difference.

## Reproduction and artifacts

```bash
cd /home/kosuke/simulation/athena_wor95_validation
python3 tst/nrfld_validation/diffusion_convergence.py --max-cores 2 \
  --output-root /home/kosuke/simulation/test/rad_nr/wor95_20261107 \
  --run-name diffusion
python3 tst/nrfld_validation/diffusion_convergence.py --max-cores 2 \
  --binary /home/kosuke/simulation/test/rad_nr/wor95_20261107/diffusion/build_source/bin/athena \
  --linear-solver fgmres \
  --output-root /home/kosuke/simulation/test/rad_nr/wor95_20261107 \
  --run-name diffusion_fgmres
cd /home/kosuke/simulation/test/rad_nr/wor95_20261107/amr
python3 make_inputs.py
mpirun -np 2 build_source/bin/athena -i nj4/input.used -d nj4/output
mpirun -np 2 build_source/bin/athena -i nj32/input.used -d nj32/output
cd ..
python3 analyze.py
```

The AMR build uses `--prob=protostellar_core_fld --coord=cartesian
--eos=adiabatic --grav=mg -nrmgfld --flux=hlld_fld --nscalars=1 -b
-hdf5 -mpi --cflag=-O3`; its exact command is in `amr/configure.log` and
`amr/build_source/config.log`. The diffusion build command and source tree
hash are in `diffusion/manifest.json`.

Artifact root: `/home/kosuke/simulation/test/rad_nr/wor95_20261107/`.
`summary.json`, `amr_summary.csv`, `diffusion_samples.csv`,
`amr_blocks.png`, both `diffusion*/convergence.png`, manifests, exact
inputs, and raw logs are retained there.

## Scope and outstanding provenance

The checked-out current source does not contain a parameter or implementation
named flux-form matvec or precision-floor, although WOR-93 reports those
options for job 335063. Its source snapshot and complete input were not found
in this checkout. Thus these regressions verify `209e22398` and the current
FGMRES code, but do not establish binary identity with the successful
long-running WOR-93 job. Reconcile that source provenance before marking the
"final solver" part of WOR-95 complete.
