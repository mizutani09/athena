# WOR-14: 256³ matched taper gate at t=120

XD job **341811** completed normally (2026-11-18 02:00–05:14 JST,
03:14:07 elapsed). Starting from the same t=110 restart, its two serial
64-rank M-bulk-a calculations either kept the nonphysical spherical-shell
thermal support or linearly tapered it from t=110 to t=120. Both ended
at the requested time, with zero opacity-domain failures and empty
stderr. A single M-debug analysis job, **341960**, completed in 28 s.

| Diagnostic at t=120 (30.56 days) | Support held | Support tapered |
|---|---:|---:|
| Global history maximum Mach | 0.4953 | 0.4899 |
| 800–860 R☉ snapshot maximum Mach proxy | 0.1926 | 0.1958 |
| Six-face Lrad / Lheat | 0.9570 | 1.0423 |
| Final preparation power / Lheat | short-step 0.714 | 0 |
| 813–817 R☉ Tgas / t=110 Tgas | 1.0005 | 0.9102 |
| 813–817 R☉ Erad / t=110 Erad | 0.9942 | 0.6864 |
| 813–817 R☉ gas pressure / t=110 pressure | 0.9990 | 0.9013 |
| 813–817 R☉ density / t=110 density | 0.9984 | 0.9992 |
| 830–850 R☉ Tgas / t=110 Tgas | 1.0000 | 1.0207 |
| 800–830 R☉ convective enthalpy proxy / Lstar | 0.00172 | 0.00224 |

The maximum Mach remains at r≈1107 R☉ in the density-floor exterior
for both cases. In the held case the ordinary-step preparation power
averaged 0.970 Lheat. In the tapered case it averaged 0.779 Lheat over
the same interval and reached exactly zero by t=120. The t=120 held
power value of 0.714 reflects its short final step (dt=0.0393); it is
not representative of the ordinary source rate.

The controlled difference is sharply localized at 813–817 R☉:
less support produces a 9% gas-temperature drop and 31% radiation-energy
drop in 2.55 days, before a significant density response. The adjacent
outer shells warm by ~2%, and the escaping radiative luminosity rises
by ~9% relative to the held case. Thus the taper has exposed a thermal
imbalance, not a hydrodynamic blow-up in this short interval. The
convective transport proxy is still ≪1 Lstar. The 4-R☉ affected
shell is narrower than the 256³ cell width of ~7.9 R☉; its detailed
shape and amplitude are resolution sensitive.

The time history and shell comparison are in
`doc/figures/wor122_prep256_taper_gate_20261118.png` (reproduce with
`python3 tst/nrfld_validation/wor122_plot_taper_gate_20261118.py`).
Small XD outputs are under
`tst/nrfld_validation/wor122_prep_thermal_20261118/` locally; large
restarts and Athena HDF outputs remain under `$WORK/wor122/` on XD.

To determine whether this thermal deficit drives later surface motion,
**job 341970** was submitted on M-large-a. It compares t=120→130 with
the source held versus fully off, each continuing its matched t=120
state. The single job requests 64 ranks and 4 h, and writes to
`$WORK/wor122/prep256_{hold,off}_t130/`. Script:
`tst/nrfld_validation/wor122_prep256_release_gate_20261118.sh`,
SHA-256 `85a7efcafe3af574a1fe4355500e9c6d2ad623a0cadf941566d51880b5cfe28d`.
The first M-bulk-a submission, 341964, was canceled while pending after
the M-large-a test-only estimate indicated an earlier start. No duplicate
calculation was left queued. The actual M-large-a job was still pending
at 10:57 JST, with Slurm's tentative start at 12:00 JST.

Until sustained source-free thermal equilibrium and reasonable surface
Mach are shown, no year-scale production run is justified. If the off
case keeps cooling, a new 1D mean state with a consistent luminosity
budget and a resolved surface transition is the next physical task.
