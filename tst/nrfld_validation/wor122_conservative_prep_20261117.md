# WOR-122 / WOR-14: conservative preparation heat transport

The supported 256³ RSG continuation reached code time 52 (13.24 d) with
converged solves, but its artificial shell support still supplied 0.945 of the
central luminosity on average from t=36 to 52. The six-face radiation loss
was about 0.957 of that luminosity. Over the same interval, the sum of gas,
radiation, and potential energy grew by 3.63e6 code energy units. The source
ledger predicts approximately 3.63e6: the supported run retains almost one
additional stellar luminosity. The rising core speed may be related to this
excess, but this correlation does not establish causation.

`problem/prep_thermal_conservative=true` makes the preparation source an
energy transfer. After the implicit FLD solve, it adds the existing shell
correction and removes the same global energy from the central heating kernel.
The transfer is capped at `Lheat * dt`, so the central heating is never made
negative by this operation. Negative shell corrections transfer energy toward
the core. The gas pressure, cached FLD gas energy, and opacity are updated in
both regions. The ordinary model keeps the option disabled. New history
columns `Eprep_redirected` and `Pprep_redirected` equal the shell values when
the transfer option is on and are zero otherwise. The existing seven-element
meshblock restart field is preserved, allowing old restarts to be read.

Local 64³ one-step PPM checks, with the 790–900 R_sun support window and
`prep_thermal_tau=0.1`:

| Case | Shell energy | Redirected energy | Maximum Mach |
| --- | ---: | ---: | ---: |
| Additional shell heating | 19374.343987793 | 0 | 0.00537704 |
| Conservative transfer | 19374.343987793 | 19374.343987793 | 0.00537704 |

The change in total gas+radiation+potential energy between the paired runs
was -19374.343993, agreeing with the redirected amount within 5.4e-6 code
units (roundoff in the much larger stored energy). A deliberately reduced
central luminosity triggered the cap: redirected energy was 229.929466,
equal to `Lheat*dt`. A restart made with the previous seven-element code was
read by the revised executable, both with the feature off and with `-r` plus
`-i` to enable it; both one-step continuations terminated normally and had
zero opacity-domain errors. Athena requires `-i` when setting a new parameter
absent from the old restart's embedded input.

XD short gate: job **341276**, M-debug, 128³ to code time 8, using the
conservative option. Run directory `$WORK/wor122/prep128_transfer_t8`; script
`$WORK/wor122/prep128_transfer_t8.sh`; executable source copy
`$WORK/wor122/athena_prep_transfer`. It is compared against the existing
matched 128³ additional-heating run `$WORK/wor122/prep128_outer_t8`.

The job completed normally in 4 min 35 s with zero opacity-domain errors.
At t=8, maximum Mach was **0.12170**, versus **0.14942** in the matched
additional-heating control. Integrated shell and redirected energies each
equal 0.98761 Lheat times elapsed time. However, the six-face radiative
luminosity rose to **1.37086 Lheat**, versus **0.97119** in the control.
Thus energy redirection reduces the early maximum speed but lets the outer
radiation field drain appreciably faster. This is not evidence of a stable,
source-free convective equilibrium, and no longer run was submitted.

The XD trial used the physically identical first version of the transfer
implementation, which stored redirected energy in two extra meshblock restart
fields. The committed implementation derives the two new history columns from
the shell correction and retains the existing seven-element restart field;
its local old-restart continuation passed. The XD trial restart should not be
used with the final executable without regenerating it.

The initial 1D radiative transition was also tested with temperature-smooth
opacity trials. A 0.1-dex Gaussian in log T slightly widened the transition
but the local 64³ one-step maximum Mach increased from 0.00505 to 0.00528,
so it was rejected. A 0.2-dex smooth or tapered table could not produce the
required LTE/FLD streaming exterior before 850 R_sun; a 32-R_sun direct flux
blend required changing the photospheric optical depth and left a 22% 1D HSE
residual. These are not qualified initial models. The temporary profile
generator edit was reverted.

This transfer does not by itself supply the missing initial convective flux
or balance the discrete surface gas force and FLD face flux. It is a short
gate on the excess-energy hypothesis, not a source-free RSG solution.
