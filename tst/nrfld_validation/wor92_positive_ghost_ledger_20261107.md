# WOR-92: energy and AMR-face audit of the positive ghost trial

The mean-preserving positive NR radiation ghost trial was rerun from the saved cycle-74 checkpoint through cycle 99 at CFL=0.3, with both rank energy and block-face ledgers enabled. The numerical `cycle`, FGMRES and Newton lines, and the full 27-row history file are **identical** to the prior run without ledgers. All 26 new Newton solves converged. Each of the 26 implicit stages has both rank ledgers and 120 unique block records. The [audit script](wor92_positive_ghost_ledger_20261107.py) geometrically pairs all matching block faces in three dimensions and finds 24 coarse-fine face groups per stage.

| Metric over cycles 74–99 | Measured maximum | Location |
| --- | ---: | --- |
| Weighted coarse-fine face-flux mismatch | `3.901784936e-10` | Cycle 75 |
| Rank signed energy-balance fraction | `2.695645178e-6` | Cycle 92 |
| Rank absolute-term energy-balance fraction | `2.701417008e-6` | Cycle 92 |

At the two cycles where the ghost limiter activates, the weighted coarse-fine mismatch is `3.209e-12` (87) and `4.109e-12` (88). The rank signed energy-balance fractions are `3.185e-11` and `7.371e-11`. The later increase to about `2.7e-6` begins at cycle 92, after the limiter stops activating. A [same-checkpoint Newton gate audit](wor92_newton_energy_gate_20261107.md) shows that one additional Newton iteration lowers the cycle-92 signed balance to `3.685e-11`, although the stricter global gate fails at cycle 0. The ledger measures the implicit FLD energy equation at committed stages; it does not by itself establish full hydro, gravitational, and AMR-regrid conservation or a science accuracy budget. The original no-limiter CFL=0.3 run ends at the start of cycle 87, so there is no same-trajectory reference for later cycles.

The initial MPI output capture interleaved some rank lines, so it was **not used** for the numerical ledger. A [rank-isolating wrapper](wor92_ledger_rank_wrapper_20261107.sh) wrote each rank's stdout directly to a distinct file. The audit then required 120 unique blocks, both ranks, and 24 coarse-fine groups for every stage, as well as the interior-flux sum matching the paired face sum. The exact rank logs, history, launcher log, [stage CSV](wor92_positive_ghost_ledger_results_20261107/stages.csv), [face CSV](wor92_positive_ghost_ledger_results_20261107/faces.csv), [JSON](wor92_positive_ghost_ledger_results_20261107/summary.json), and [figure](wor92_positive_ghost_ledger_results_20261107/ledger.png) are in the [evidence archive](wor92_positive_ghost_ledger_results_20261107/evidence.tar.gz). Regenerate them with:

```bash
python3 tst/nrfld_validation/wor92_positive_ghost_ledger_20261107.py
```

For reproduction, use the exact input and checkpoint hashes in the [continuation report](wor92_domain_guard_followup_20261107.md), the isolated binary SHA-256 `e7854e44e35a3ed85f9ab1963a801c1b09a6736db7356c6ff53dd2c0519217da`, and a fresh output directory. From the data bundle directory, run the [wrapper](wor92_ledger_rank_wrapper_20261107.sh) under `mpirun -np 2` with the seven `ATHENA_NRFLD_*` flags, `ATHENA_WOR92_INEXACT_TRUE_CAP=1e-7`, `ATHENA_WOR92_POSITIVE_NR_GHOST=2`, `ATHENA_NRFLD_ENERGY_LEDGER=1`, `ATHENA_NRFLD_INTERFACE_LEDGER=1`, `WOR92_ATHENA_BIN=/path/to/isolated/bin/athena`, and `WOR92_LEDGER_RANK_DIR=/path/to/fresh/output`; pass `-r` the cycle-74 checkpoint, `-i` the guard input, and `time/nlim=100 time/tlim=10000.0`.

A clean, single-method, opt-in integration was subsequently committed as `14633c31`. Its isolated binary SHA-256 is `0e6af159f3c545d765667ab45a3b7feaad7a3ddbb0c05e8897633822c86f5c1b`; a cycle-74-to-90 replay matched all cycle, Newton and FGMRES lines and the full history file of the original mean-preserving trial. Post-regrid opacity domain, higher-precision state differences, later energy behavior, and first-core physics remain open. WOR-92 remains **In Progress**. No XD2000 use or merge occurred.
