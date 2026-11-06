#!/usr/bin/env python3
"""Audit the solar-box initial column and a temporary dark-top control.

Interior radiation faces reconstruct the same three-dimensional FLD opacity
cap and Levermore-Pomraning limiter as NRFLD.cpp. Boundary fluxes come from
Athena++ HST; a full dump cannot recover the lagged Marshak face coefficient.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_initial_flux_audit import load_profile
from solar_convection_report import (input_values, parse_log, read_history,
                                     total_bottom_flux, total_top_flux)

import sys

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from plot_simple_convection_surface3d import read_volume  # noqa: E402
from plot_simple_convection_tau import opacity_interpolator, units_from_input  # noqa: E402


F_SUN = 6.3e10
C_LIGHT = 2.99792458e10


def average(time: np.ndarray, values: np.ndarray,
            start: float, end: float) -> float:
    if start < time[0] or end > time[-1] or start >= end:
        raise ValueError(f"Window {start}..{end} outside {time[0]}..{time[-1]}")
    interior = (time > start) & (time < end)
    x = np.r_[start, time[interior], end]
    y = np.r_[np.interp(start, time, values), values[interior],
              np.interp(end, time, values)]
    return float(np.trapezoid(y, x)/(end-start))


def pair_at(run: Path, target: float) -> tuple[np.ndarray, np.ndarray,
                                                tuple[np.ndarray, ...], float]:
    pairs = []
    uov = {p.name.split(".uov_full.")[-1]: p
           for p in run.glob("*.uov_full.*.athdf")}
    for prim in run.glob("*.prim_full.*.athdf"):
        suffix = prim.name.split(".prim_full.")[-1]
        if suffix in uov:
            pairs.append((prim, uov[suffix]))
    if not pairs:
        raise ValueError(f"No paired full-volume dumps in {run}")
    candidates = []
    for prim, other in pairs:
        import h5py
        with h5py.File(prim) as source:
            time = float(source.attrs["Time"])
        candidates.append((abs(time-target), prim, other))
    _, prim, other = min(candidates)
    a, coordinates, time = read_volume(prim, "prim")
    b, other_coordinates, other_time = read_volume(other, "user_out_var")
    if (abs(time-target) > 0.2 or abs(time-other_time) > 1e-6 or
            any(not np.array_equal(x, y) for x, y in
                zip(coordinates, other_coordinates))):
        raise ValueError(f"Missing full-volume pair near t={target} in {run}")
    return a, b, coordinates, time


def column(prim: np.ndarray, uov: np.ndarray,
           coordinates: tuple[np.ndarray, ...], units: dict[str, float],
           opacity, gravity: float) -> dict[str, list[float] | float]:
    x, y, z = coordinates
    dx, dy, dz = ((a[1]-a[0])*units["leng_unit"] for a in (x, y, z))
    if not np.allclose((dx, dy), dz, rtol=1e-9):
        raise ValueError("NR-FLD column audit requires cubic cells")
    rho = prim[0]*units["rho_unit"]
    pressure = prim[1]*units["egas_unit"]
    erad = uov[1]
    temp = uov[2]
    sigma = 10**opacity(np.stack((np.log10(rho), np.log10(temp)), axis=-1))*rho
    if np.any(sigma <= 0) or np.any(erad <= 0):
        raise ValueError("Positive opacity and radiation energy required")
    lower, upper = erad[:-1], erad[1:]
    sl, su = sigma[:-1], sigma[1:]
    arithmetic = 0.5*(sl+su)
    harmonic = 2*sl*su/(sl+su)
    sigma_face = np.minimum(arithmetic, np.maximum(harmonic, 4/(3*dz)))
    gx = (np.roll(lower, -1, axis=2)-np.roll(lower, 1, axis=2)
          + np.roll(upper, -1, axis=2)-np.roll(upper, 1, axis=2))/(4*dx)
    gy = (np.roll(lower, -1, axis=1)-np.roll(lower, 1, axis=1)
          + np.roll(upper, -1, axis=1)-np.roll(upper, 1, axis=1))/(4*dy)
    gz = (upper-lower)/dz
    gradient = np.hypot(gx, np.hypot(gy, gz))
    eface = 0.5*(lower+upper)
    ratio = gradient/(sigma_face*eface)
    limiter = np.empty_like(ratio)
    small = ratio <= 1
    limiter[small] = (2+ratio[small])/(6+3*ratio[small]+ratio[small]**2)
    inverse = 1/ratio[~small]
    limiter[~small] = (inverse+2*inverse**2)/(1+3*inverse+6*inverse**2)
    face_flux = -C_LIGHT*limiter*gz/sigma_face
    zface = 0.5*(z[1:]+z[:-1])
    rho_mean = np.mean(rho, axis=(1, 2))
    pressure_mean = np.mean(pressure, axis=(1, 2))
    hse = (np.diff(pressure_mean)/dz
           + 0.5*(rho_mean[1:]+rho_mean[:-1])*gravity)
    hse_fraction = np.abs(hse)/(0.5*(rho_mean[1:]+rho_mean[:-1])*gravity)
    energy_density = np.mean(uov[0]+uov[1], axis=(1, 2))
    # Includes gas and radiation internal energy, but not gravitational work.
    energy_above = np.cumsum((energy_density*dz)[::-1])[::-1]
    return {
        "z_code": z.tolist(), "face_z_code": zface.tolist(),
        "face_frad_over_solar": (np.mean(face_flux, axis=(1, 2))/F_SUN).tolist(),
        "face_limiter_mean": np.mean(limiter, axis=(1, 2)).tolist(),
        "face_reduced_gradient_mean": np.mean(ratio, axis=(1, 2)).tolist(),
        "T_mean_K": np.mean(temp, axis=(1, 2)).tolist(),
        # Gas-radiation coupling is very stiff and cancels between the two
        # energy equations. Use the face-flux divergence for total cooling.
        "energy_above_over_solar_seconds": (energy_above/F_SUN).tolist(),
        "hse_fraction": hse_fraction.tolist(),
        "max_abs_vz_code": float(np.max(np.abs(prim[4]))),
    }


def load_history(run: Path, header: Path | None = None) -> dict[str, np.ndarray]:
    return read_history(next(run.glob("*.hst")), header)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--opacity-table", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root/"comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = {name: root/name for name in
             ("normal_t0_40", "dark_t0_10", "dark_then_normal_t10_40")}
    for name, run in cases.items():
        log = parse_log(run/"run.log")
        if (not log["completed"] or log["fatal_count"] or
                log["slow_mg_warning_count"] or
                set(log["nr_status_counts"])-{"converged", "dt_zero_initialization"}):
            raise ValueError(f"{name}: solver did not pass: {log}")
    manifest = {name: json.loads((run/"manifest.json").read_text())
                for name, run in cases.items()}
    for key in ("binary_sha256", "base_input_sha256", "data_sha256"):
        if len({json.dumps(m[key], sort_keys=True) for m in manifest.values()}) != 1:
            raise ValueError(f"Matched runs differ in {key}")
    controls = [{item for item in m["problem_overrides"]
                 if not item.startswith("rad_top_alpha=")}
                for m in manifest.values()]
    if any(control != controls[0] for control in controls[1:]):
        raise ValueError("Matched runs differ in non-radiative controls")
    alphas = [next(float(item.split("=", 1)[1])
                   for item in manifest[name]["problem_overrides"]
                   if item.startswith("rad_top_alpha=")) for name in cases]
    if not np.allclose(alphas, [0.5, 1e-6, 0.5]):
        raise ValueError(f"Unexpected top boundary controls: {alphas}")
    if (manifest["normal_t0_40"]["restart"] is not None or
            manifest["dark_t0_10"]["restart"] is not None or
            Path(manifest["dark_then_normal_t10_40"]["restart"]).resolve() !=
            (cases["dark_t0_10"] /
             "simple_convection_general_6Mm_box.final.rst").resolve()):
        raise ValueError("Expected t=0 paired runs and restart from dark t=10")
    headers = next(cases["dark_t0_10"].glob("*.hst"))
    histories = {name: load_history(run, headers if name.startswith("dark_then")
                                    else None) for name, run in cases.items()}
    normal = histories["normal_t0_40"]
    dark = histories["dark_t0_10"]
    restored = histories["dark_then_normal_t10_40"]
    if normal["time"][0] != 0 or dark["time"][0] != 0:
        raise ValueError("The matched startup must include t=0")
    if restored["time"][0] > 10.2 or restored["time"][-1] < 39.9:
        raise ValueError("The restored case must cover t=10..40")
    units = units_from_input(cases["normal_t0_40"]/"athinput.generated")
    physical = input_values(cases["normal_t0_40"]/"athinput.generated")
    opacity = opacity_interpolator(args.opacity_table.resolve())
    initial = {}
    for name in ("normal_t0_40", "dark_t0_10"):
        prim, uov, coords, time = pair_at(cases[name], 0)
        initial[name] = column(prim, uov, coords, units, opacity,
                               physical["grav_acc"])
    # Same initial thermodynamic state; the top radiation boundary alone
    # changes. The t=0 output follows Athena++'s dt=0 initialization.
    for key in ("T_mean_K", "face_frad_over_solar"):
        a = np.asarray(initial["normal_t0_40"][key])
        b = np.asarray(initial["dark_t0_10"][key])
        if not np.allclose(a, b, rtol=1e-6, atol=1e-8):
            raise ValueError(f"Initial {key} differs between matched cases")
    profile = load_profile(args.profile.resolve())
    z = np.asarray(initial["normal_t0_40"]["z_code"])
    face_z = np.asarray(initial["normal_t0_40"]["face_z_code"])
    initial_flux = np.asarray(initial["normal_t0_40"]["face_frad_over_solar"])
    energy_seconds = np.asarray(
        initial["normal_t0_40"]["energy_above_over_solar_seconds"])
    dz_code = z[1]-z[0]
    cell_faces = np.r_[z[0]-0.5*dz_code, z+0.5*dz_code]
    energy_density = -np.diff(energy_seconds)*F_SUN / (
        dz_code*units["leng_unit"])
    # The cumulative energy array includes each cell from its center upward.
    # Recover a partial-cell integral for a geometric layer from its density.
    energy_density = np.r_[energy_density, energy_seconds[-1]*F_SUN/
                           (dz_code*units["leng_unit"])]
    def layer_hours(depth_cm: float) -> float:
        threshold = cell_faces[-1]-depth_cm/units["leng_unit"]
        lengths = np.clip(cell_faces[1:]-np.maximum(cell_faces[:-1], threshold),
                          0, dz_code)*units["leng_unit"]
        return float(np.dot(energy_density, lengths)/F_SUN/3600)
    def near_face(height: float) -> dict[str, float]:
        index = int(np.argmin(np.abs(face_z-height)))
        return {"z_code": float(face_z[index]),
                "F_rad_over_solar": float(initial_flux[index])}
    result = {
        "method": "interior 3D face flux from saved t=0 state, NRFLD face opacity cap, full gradient and limiter; exact boundary face flux from HST",
        "reference_flux_cgs": F_SUN,
        "source_code": "src/nr_multigrid/NRFLD.cpp; src/fld/fld.hpp; src/pgen/simple_convection_lhllc_fld.cpp",
        "runs": {name: str(run) for name, run in cases.items()},
        "solver": {name: parse_log(run/"run.log") for name, run in cases.items()},
        "initial": initial,
        "initial_boundary": {
            name: {"top_over_solar": float(total_top_flux(hist)[0]/F_SUN),
                   "bottom_over_solar": float(total_bottom_flux(hist)[0]/F_SUN)}
            for name, hist in (("normal", normal), ("dark", dark))},
        "initial_column": {
            "face_near_z0": near_face(0),
            "face_near_z2": near_face(2),
            "top_interior_face": {"z_code": float(face_z[-1]),
                                  "F_rad_over_solar": float(initial_flux[-1])},
            "model_convective_flux_at_z0_over_solar": float(np.interp(
                0, profile["z"], profile["convective_flux_fraction"]
                * 6.3e10)/F_SUN),
            "initial_max_abs_vz_code": initial["normal_t0_40"]["max_abs_vz_code"],
            "max_centered_hse_fraction": float(np.max(
                initial["normal_t0_40"]["hse_fraction"])),
            "upper_half_Mm_energy_over_solar_hours": layer_hours(0.5e8),
            "upper_one_Mm_energy_over_solar_hours": layer_hours(1e8),
        },
        "windows": {},
        "surfaces": {},
    }
    combined_time = np.r_[dark["time"], restored["time"][restored["time"] > dark["time"][-1]]]
    combined_top = np.r_[total_top_flux(dark), total_top_flux(restored)[
        restored["time"] > dark["time"][-1]]]
    combined_bottom = np.r_[total_bottom_flux(dark), total_bottom_flux(restored)[
        restored["time"] > dark["time"][-1]]]
    if np.any(np.diff(combined_time) <= 0):
        raise ValueError("Combined delayed history is not increasing")
    # Interior faces use the same closure as the NR operator. The two
    # physical boundary values use HST's accepted face-flux diagnostics.
    full_faces = np.r_[result["initial_boundary"]["normal"]["bottom_over_solar"],
                       initial_flux,
                       result["initial_boundary"]["normal"]["top_over_solar"]]
    result["initial_column"]["radiative_flux_divergence_per_cell_over_solar"] = (
        np.diff(full_faces).tolist())
    result["initial_column"]["whole_box_radiative_cooling_over_solar"] = float(
        full_faces[-1]-full_faces[0])
    for label, t, top, bottom in (
        ("normal", normal["time"], total_top_flux(normal), total_bottom_flux(normal)),
        ("dark_then_restored", combined_time, combined_top, combined_bottom),
    ):
        result["windows"][label] = {}
        for first, last in ((0, 2), (2, 10), (10.2, 20), (30, 40)):
            result["windows"][label][f"{first:g}_{last:g}"] = {
                "top_over_solar": average(t, top, first, last)/F_SUN,
                "bottom_over_solar": average(t, bottom, first, last)/F_SUN}
    for name, run in cases.items():
        records = json.loads((run/"surface3d.json").read_text())["records"]
        result["surfaces"][name] = [
            {"time_code": row["time_code"],
             "T_tau23_mean_K": row["T_tau23_mean_K"],
             "z_tau23_mean_code": row["z_tau23_mean_code"]}
            for row in records]
    (output/"summary.json").write_text(json.dumps(result, indent=2)+"\n")

    plt.rcParams.update({"font.size": 16})
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True,
                             constrained_layout=True)
    axes[0].plot(normal["time"], total_top_flux(normal)/F_SUN,
                 label="normal top emission")
    axes[0].plot(combined_time, combined_top/F_SUN,
                 label="dark until t=10, then normal")
    axes[0].axhline(1, ls="--", lw=1, color="black", label="solar")
    axes[0].set_ylabel("Upper flux / solar")
    axes[1].plot(normal["time"], total_bottom_flux(normal)/F_SUN,
                 label="normal")
    axes[1].plot(combined_time, combined_bottom/F_SUN,
                 label="dark until t=10, then normal")
    axes[1].set_ylabel("Lower flux / solar")
    for name, label in (("normal_t0_40", "normal"),
                        ("dark_t0_10", "dark phase"),
                        ("dark_then_normal_t10_40", "after restoration")):
        rows = result["surfaces"][name]
        axes[2].plot([row["time_code"] for row in rows],
                     [row["T_tau23_mean_K"] for row in rows], "o-", label=label)
    axes[2].set(xlabel="Code time", ylabel="3D mean T at τ=2/3 [K]")
    for axis in axes:
        axis.axvspan(0, 10, color="gray", alpha=.08)
        axis.grid(alpha=.25)
        axis.legend()
    fig.savefig(output/"startup_compare.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].plot(face_z, initial_flux, label="actual t=0 FLD interior")
    axes[0, 0].plot(profile["z"], profile["flux_rad"]/F_SUN,
                    label="1D target radiation", ls="--")
    axes[0, 0].plot(profile["z"], profile["convective_flux_fraction"],
                    label="1D target convection", ls=":")
    axes[0, 0].plot([z[0], z[-1]],
                    [result["initial_boundary"]["normal"]["bottom_over_solar"],
                     result["initial_boundary"]["normal"]["top_over_solar"]],
                    "ks", label="exact boundary radiation")
    axes[0, 0].set(xlabel="Height [code]", ylabel="Flux / solar")
    axes[0, 0].legend()
    axes[0, 1].semilogy(z, np.maximum(energy_seconds/3600, 1e-8))
    axes[0, 1].set(xlabel="Height [code]", ylabel="Energy above / solar flux [h]")
    axes[1, 0].semilogy(face_z, initial["normal_t0_40"]["hse_fraction"])
    axes[1, 0].set(xlabel="Height [code]", ylabel="Centered HSE residual / ρg")
    axes[1, 1].plot(z, np.diff(full_faces))
    axes[1, 1].axhline(0, color="black", lw=1)
    axes[1, 1].set(xlabel="Height [code]",
                   ylabel="Radiative flux divergence per cell / solar")
    for axis in axes.flat:
        axis.grid(alpha=.25)
    fig.savefig(output/"initial_column.png", dpi=150)
    plt.close(fig)

    b = result["initial_boundary"]["normal"]
    c = result["initial_column"]
    w = result["windows"]
    rows = "".join(
        f"<tr><td>{html.escape(label)}</td>"
        f"<td>{w[label]['2_10']['top_over_solar']:.3f}</td>"
        f"<td>{w[label]['30_40']['top_over_solar']:.3f}</td>"
        f"<td>{w[label]['30_40']['bottom_over_solar']:.3f}</td></tr>"
        for label in ("normal", "dark_then_restored"))
    (output/"index.html").write_text(
        "<!doctype html><html lang='ja'><meta charset='utf-8'>"
        "<title>太陽対流の初期熱収支</title><style>"
        "body{font:22px/1.55 system-ui;max-width:1250px;margin:2rem auto;"
        "padding:0 1rem;color:#17324a}h1{font-size:2rem}"
        "strong{color:#184c76}table{border-collapse:collapse}"
        "th,td{padding:.55rem .75rem;border-bottom:1px solid #ccc;text-align:right}"
        "th:first-child,td:first-child{text-align:left}img{max-width:100%}"
        "</style><h1>太陽対流の初期熱収支</h1>"
        f"<p><strong>初期上端 {b['top_over_solar']:.3f} Fsun、"
        f"下端 {b['bottom_over_solar']:.2e} Fsun。</strong> "
        "初期速度が0なので、1Dモデルの深部対流流束を3D計算は運べません。"
        f"z≈0 の内部FLD流束も {c['face_near_z0']['F_rad_over_solar']:.2e} Fsun。"
        "上端放射を一時的に止めても、この内部熱輸送欠損は残ります。</p>"
        f"<p><strong>t=30–40の上端平均は通常 {w['normal']['30_40']['top_over_solar']:.3f} "
        f"Fsun、暗い上端から戻した場合 {w['dark_then_restored']['30_40']['top_over_solar']:.3f} "
        "Fsun。</strong> 初期上端を断熱に近づける操作は、少なくともこの短時間では"
        "太陽流束への回復を早めません。長期の定常値を判定する比較ではありません。</p>"
        "<p>上端抑制は α=10⁻⁶ を t=10 まで使い、その後 α=0.5 に戻した"
        "因果対照です。太陽表面の物理境界として推奨する設定ではありません。"
        "内部面は保存3D状態からNR-FLDの不透明度平均・3次元勾配・limiterを再構成。"
        "境界面はAthena++の履歴値を使用します。</p>"
        "<table><tr><th>ケース</th><th>t=2–10 上端/Fsun</th>"
        "<th>t=30–40 上端/Fsun</th><th>t=30–40 下端/Fsun</th></tr>"
        +rows+"</table>"
        "<img src='startup_compare.png' alt='上端放射の短時間対照'>"
        "<img src='initial_column.png' alt='初期列の熱収支'>"
        "<p>各計算: "
        "<a href='../normal_t0_40/index.html'>通常境界</a> / "
        "<a href='../dark_t0_10/index.html'>暗い上端</a> / "
        "<a href='../dark_then_normal_t10_40/index.html'>境界復帰後</a></p>"
        "<p><a href='summary.json'>全数値と方法</a></p></html>\n")
    print(output/"index.html")


if __name__ == "__main__":
    main()
