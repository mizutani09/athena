"""Declared physical accuracy gates for completed NR-FLD validation reports.

These gates complement the runners' final-time, finite-field, and strict-solver
checks.  A completed calculation can therefore have an accuracy gap without
being mistaken for a solver or infrastructure failure.
"""

from __future__ import annotations

from html import escape
import math


def _check(case: str, quantity: str, value: float, limit: float,
           relation: str = "<=") -> dict:
    passed = (math.isfinite(value) and
              (value <= limit if relation == "<=" else value >= limit))
    return {"case": case, "quantity": quantity, "value": value,
            "relation": relation, "limit": limit, "passed": passed}


def _result(checks: list[dict], rationale: str) -> dict:
    return {"passed": bool(checks) and all(row["passed"] for row in checks),
            "checks": checks, "rationale": rationale}


def radiative_shock(summary: dict) -> dict:
    """Require bulk agreement, a located stationary shock, and its gas spike."""
    checks = []
    for layout, cases in (("uniform", summary["cases"]),
                          ("amr", summary.get("amr_attachment", {}).get("cases", {}))):
        if not cases:
            continue
        for case in summary["selected_cases"]:
            data = cases[case]
            analysis = data["analysis"]
            final = analysis["final"]
            name = f"{layout}/{case}"
            for field in ("rho", "tgas", "trad"):
                checks.append(_check(name, f"aligned {field} relative L1 [%]",
                    final["agreement"][field]["global_relative_L1_percent"], 2.0))
            domain_cm = 1500.0 if case == "mach2" else 6000.0
            checks.append(_check(name, "shock position error / domain width",
                abs(final["shock_alignment_shift_cm"])/domain_cm, 0.05))
            checks.append(_check(name, "previous aligned snapshot change",
                max(analysis["aligned_change_since_previous"].values()), 0.01))
            peak = analysis["gas_temperature_peak"]
            checks.append(_check(name, "gas spike peak relative error",
                peak["relative_peak_error"], 0.10))
            checks.append(_check(name, "gas spike amplitude recovery",
                peak["spike_amplitude_recovery"], 0.50, ">="))
    return _result(checks,
        "The resolved steady shock must agree with the semi-analytic bulk profile "
        "within 2% L1, lie within 5% of the domain width, change by at most 1% "
        "between the final snapshots, and recover at least half the gas-temperature "
        "spike with peak error at most 10%. Alignment is used only for the shape L1.")


def advection_pulse(summary: dict) -> dict:
    """Compare moving and static solutions after only the prescribed v*t shift."""
    n = int(summary["cases"]["static"]["nx1"])
    metric = summary["analysis"]["metrics"]
    # Coarse fast runs are screening checks; the regular 128+ cell case has a
    # stricter bound on the background-subtracted pulse, where error is visible.
    contrast_limit = 0.10 if n < 128 else 0.01
    velocity_limit = 0.05 if n < 128 else 0.01
    checks = [_check("pulse", f"{field} pulse-contrast relative L1",
                     metric[field]["pulse_contrast_L1"], contrast_limit)
              for field in ("rho", "Tgas", "Erad")]
    checks.append(_check("pulse", "max velocity mismatch / imposed speed",
                         metric["vel1"]["max_abs"] / 1.0e6, velocity_limit))
    return _result(checks,
        f"At N={n}, the background-subtracted density, gas-temperature, and "
        f"radiation-energy pulse contrasts must each agree within "
        f"{100*contrast_limit:g}%; the velocity perturbation must be within "
        f"{100*velocity_limit:g}% of the imposed 10 km/s. The only shift is v0*t.")


def static_equilibrium(summary: dict) -> dict:
    """Bound pointwise equilibrium drift and spurious fluid motion."""
    final = summary["analysis"]["final"]
    checks = [_check("equilibrium", f"max {field} pointwise relative drift",
                     final[field]["max_pointwise_relative_drift"], 1e-4)
              for field in ("rho", "Tgas", "Trad", "Erad", "Ptot")]
    checks.extend((
        _check("equilibrium", "max speed [cm/s]", final["velocity_max_cm_s"], 1e-5),
        _check("equilibrium", "total-pressure spatial range / mean",
               final["total_pressure_spatial_range_over_mean"], 1e-4),
        _check("equilibrium", "mass relative drift", final["mass_relative_drift"], 1e-5),
        _check("equilibrium", "total-energy relative drift",
               final["total_energy_relative_drift"], 1e-5)))
    return _result(checks,
        "Over the full target time, every pointwise state drift and the total-pressure "
        "spatial variation must stay below 0.01%, mass and total energy below "
        "0.001%, and spurious speed below 1e-5 cm/s. HDF5 scalar rounding "
        "limits sensitivity below about 1e-7 relative.")


def acceptance_html(assessment: dict | None) -> str:
    if not assessment:
        return ""
    rows = "".join(
        f'<tr><th>{escape(row["case"])}</th><td>{escape(row["quantity"])}</td>'
        f'<td>{row["value"]:.4g}</td><td>{escape(row["relation"])} '
        f'{row["limit"]:.4g}</td><td>{"PASS" if row["passed"] else "NOT MET"}</td></tr>'
        for row in assessment["checks"])
    verdict = "PASS" if assessment["passed"] else "ACCURACY NOT MET"
    return (f'<section><h2>Physical accuracy criteria: {verdict}</h2>'
            f'<p>{escape(assessment["rationale"])}</p>'
            '<div class="scroll"><table><thead><tr><th>Case</th><th>Quantity</th>'
            '<th>Measured</th><th>Criterion</th><th>Result</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div></section>')
