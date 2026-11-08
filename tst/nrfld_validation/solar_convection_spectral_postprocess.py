#!/usr/bin/env python3
"""LTE monochromatic formal transfer through completed 3D solar-box dumps.

The opacity source is the *same* Optab rho-T grid used to make the grey FLD
table. Scattering is thermalized (S_lambda=B_lambda) in the main synthesis;
the report also quantifies the sensitivity to omitting scattering opacity.
This is a diagnostic synthesis, not a self-consistent non-grey RHD run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from plot_simple_convection_surface3d import read_volume  # noqa: E402
from plot_simple_convection_tau import units_from_input  # noqa: E402

H = 6.62607015e-27
C = 2.99792458e10
KB = 1.380649e-16
SIGMA = 5.670374419e-5
AU = 1.495978707e13
RSUN = 6.957e10
FSUN = 6.3e10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planck_nm(wavelength_nm: float, temperature: np.ndarray) -> np.ndarray:
    """Planck specific intensity in erg s^-1 cm^-2 sr^-1 nm^-1."""
    lam = wavelength_nm * 1e-7
    x = H * C / (lam * KB * temperature)
    return 2 * H * C*C / lam**5 / np.expm1(x) * 1e-7


def prepare_geometry(rho: np.ndarray, temp: np.ndarray,
                     grey_table: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(grey_table) as source:
        points = np.column_stack((np.log10(source["source_points/density"][:]),
                                  np.log10(source["source_points/temperature"][:])))
    if not np.all(np.isfinite(points)):
        raise ValueError("Opacity source points are not finite")
    triangulation = Delaunay(points)
    query = np.column_stack((np.log10(rho.ravel()), np.log10(temp.ravel())))
    simplices = triangulation.find_simplex(query)
    if np.any(simplices < 0):
        raise ValueError(f"{np.count_nonzero(simplices < 0)} cells lie outside "
                         "the Optab rho-T convex hull; refusing extrapolation")
    transforms = triangulation.transform[simplices]
    delta = query - transforms[:, 2]
    first_two = np.einsum("ijk,ik->ij", transforms[:, :2, :], delta)
    weights = np.column_stack((first_two, 1 - np.sum(first_two, axis=1)))
    vertices = triangulation.simplices[simplices]
    reconstructed = np.einsum("ij,ijk->ik",weights,points[vertices])
    if not np.allclose(reconstructed,query,rtol=0,atol=1e-10):
        raise ValueError("Opacity interpolation geometry failed to reproduce states")
    return vertices, weights, np.unique(vertices)


def opacity_index_grid(mono_dir: Path, desired_nm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(mono_dir / "mono_00001.h5") as source:
        wavenumber = source["grd"][:]
    target = 1e7 / desired_nm
    idx = np.searchsorted(wavenumber, target)
    idx = np.clip(idx, 1, len(wavenumber)-1)
    idx -= (np.abs(wavenumber[idx-1]-target) < np.abs(wavenumber[idx]-target))
    return idx, 1e7 / wavenumber[idx]


def load_opacity_samples(mono_dir: Path, used_vertices: np.ndarray,
                         indices: np.ndarray, points: np.ndarray,
                         grey_rosseland_log10: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    abs_sample = np.zeros((len(points), len(indices)))
    sca_sample = np.zeros_like(abs_sample)
    source_files = []
    sorted_order = np.argsort(indices)
    sorted_idx = indices[sorted_order]
    inverse = np.argsort(sorted_order)
    for j in used_vertices:
        path = mono_dir / f"mono_{j+1:05d}.h5"
        if not path.exists():
            raise FileNotFoundError(path)
        with h5py.File(path) as source:
            if not (np.isclose(source["rho"][0], 10**points[j, 0], rtol=1e-10)
                    and np.isclose(source["temp"][0], 10**points[j, 1], rtol=1e-10)):
                raise ValueError(f"Optab state does not match grey source: {path}")
            actual_rosseland = float(source["ros"][0]/source["rho"][0])
            if not np.isclose(np.log10(actual_rosseland),
                              grey_rosseland_log10[j], rtol=0, atol=1e-10):
                raise ValueError(f"Optab Rosseland mean does not match grey table: {path}")
            abs_sample[j] = (source["abs"][sorted_idx] +
                             source["line"][sorted_idx])[inverse]
            sca_sample[j] = source["sca"][sorted_idx][inverse]
        source_files.append({"path": str(path), "sha256": sha256(path)})
    return abs_sample, sca_sample, source_files


def interpolate_extinction(samples: np.ndarray, vertices: np.ndarray,
                           weights: np.ndarray, index: int,
                           shape: tuple[int, ...]) -> np.ndarray:
    values = np.log(np.maximum(samples[:, index], 1e-99))
    log_value = np.einsum("ij,ij->i", weights, values[vertices])
    return np.exp(log_value).reshape(shape)


def formal_ray(temp: np.ndarray, chi: np.ndarray, wavelength_nm: float,
               dz_cm: float, dx_cm: float, dy_cm: float, mu: float,
               azimuth: float, substeps: int = 2) -> np.ndarray:
    """Piecewise-isothermal formal solution along a periodic inclined ray."""
    nz, ny, nx = temp.shape
    yy, xx = np.indices((ny, nx), dtype=float)
    b = planck_nm(wavelength_nm, temp)
    if not 0 < mu <= 1:
        raise ValueError("mu must be in (0, 1]")
    inclination = np.sqrt(1-mu*mu) / mu
    if mu == 1:
        intensity = b[0].copy()
    else:
        bottom_depth_cm = (nz-0.5)*dz_cm
        xb = xx-bottom_depth_cm*inclination*np.cos(azimuth)/dx_cm
        yb = yy-bottom_depth_cm*inclination*np.sin(azimuth)/dy_cm
        intensity = map_coordinates(b[0],[yb,xb],order=1,mode="grid-wrap")
    for iz in range(nz):
        for sub in range(substeps):
            depth_cm = ((nz-1-iz) + 0.5 - (sub+0.5)/substeps + 0.5) * dz_cm
            xray = xx - depth_cm * inclination * np.cos(azimuth) / dx_cm
            yray = yy - depth_cm * inclination * np.sin(azimuth) / dy_cm
            if mu == 1:
                local_b = b[iz]
                local_chi = chi[iz]
            else:
                coords = [yray, xray]
                local_b = map_coordinates(b[iz], coords, order=1, mode="grid-wrap")
                local_chi = map_coordinates(chi[iz], coords, order=1,
                                            mode="grid-wrap")
            transmissivity = np.exp(-np.maximum(local_chi, 0)*dz_cm/
                                    (mu*substeps))
            intensity = intensity*transmissivity + local_b*(-np.expm1(
                -np.maximum(local_chi, 0)*dz_cm/(mu*substeps)))
    return intensity


def exact_slab_test() -> None:
    temperature = np.full((4, 3, 5), 5100.0)
    chi = np.full_like(temperature, 2e-8)
    for mu, phi in ((1.0, 0.0), (0.42, 1.2)):
        result = formal_ray(temperature, chi, 500, 1e8, 1e8, 1e8,
                            mu, phi, substeps=3)
        if not np.allclose(result, planck_nm(500, temperature[0]), rtol=1e-12):
            raise AssertionError("Isothermal formal solution failed")
    nodes,quadrature=np.polynomial.legendre.leggauss(3)
    nodes=(nodes+1)/2
    quadrature=quadrature/2
    blackbody_flux=2*np.pi*sum(
        weight*angle*np.mean(formal_ray(temperature,chi,500,1e8,1e8,1e8,
                                         float(angle),0.3,substeps=2))
        for angle,weight in zip(nodes,quadrature))
    if not np.isclose(blackbody_flux,np.pi*planck_nm(500,temperature[0,0,0]),
                      rtol=1e-12):
        raise AssertionError("Angular flux quadrature failed")
    two_layer = np.stack((np.full((3,5), 6000.),np.full((3,5), 4500.)))
    chi_two = np.full_like(two_layer, 1e-8)
    for mu,phi in ((1.0,0.0),(0.42,1.2)):
        result = formal_ray(two_layer,chi_two,500,1e8,1e8,1e8,
                            mu,phi,substeps=3)
        top_transmission=np.exp(-1/mu)
        exact=(planck_nm(500,two_layer[0])*top_transmission +
               planck_nm(500,two_layer[1])*(1-top_transmission))
        if not np.allclose(result,exact,rtol=1e-12):
            raise AssertionError("Two-layer formal solution failed")


def solar_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    # FITS is only needed to decode the downloaded CALSPEC reference.
    from astropy.io import fits
    with fits.open(path) as source:
        data = source[1].data
        wavelength_nm = np.asarray(data["WAVELENGTH"], float) / 10
        irradiance_per_angstrom = np.asarray(data["FLUX"], float)
    surface_flux_nm = irradiance_per_angstrom * 10 * (AU/RSUN)**2
    return wavelength_nm, surface_flux_nm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--mono-dir", type=Path, required=True)
    parser.add_argument("--grey-table", type=Path, required=True)
    parser.add_argument("--solar-reference", type=Path, required=True)
    parser.add_argument("--dwarf-sequence", type=Path,
                        help="Pecaut & Mamajek dwarf temperature table")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, default=3)
    parser.add_argument("--wavelengths", type=int, default=100)
    parser.add_argument("--substeps", type=int, default=2)
    args = parser.parse_args()
    if args.snapshots < 1 or args.wavelengths < 20 or args.substeps < 1:
        parser.error("Need >=1 snapshot, >=20 wavelengths and >=1 substep")
    exact_slab_test()
    campaign = args.campaign.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    images = output / "images"
    images.mkdir(exist_ok=True)
    state = json.loads((campaign / "state.json").read_text())
    movie_runs=[campaign/row["case"] for row in state["completed"][-48:]]
    (output/"movie_sources.json").write_text(json.dumps({
        "source_runs_relative":[os.path.relpath(run,output) for run in movie_runs],
        "source_runs":[str(run) for run in movie_runs]},indent=2)+"\n")
    completed = state["completed"][-args.snapshots:]
    if len(completed) < args.snapshots:
        raise ValueError("Too few completed segments")
    units = units_from_input(campaign / "athinput.base.validated")
    grey = args.grey_table.resolve()
    mono_dir = args.mono_dir.resolve()
    with h5py.File(grey) as source:
        points = np.column_stack((np.log10(source["source_points/density"][:]),
                                  np.log10(source["source_points/temperature"][:])))
        grey_rosseland_log10 = source["source_points/log10_rosseland"][:]
    requested_nm = np.r_[np.geomspace(100, 299, 24),
                         np.geomspace(300, 2500, args.wavelengths),
                         np.geomspace(2501, 10000, 36)]
    indices, wavelengths_nm = opacity_index_grid(mono_dir, requested_nm)
    optical_band = (requested_nm >= 300) & (requested_nm <= 2500)
    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate Optab spectral samples")
    snapshots = []
    all_vertices = set()
    for row in completed:
        run = campaign / row["case"]
        prim_path = sorted(run.glob("*.prim_full.*.athdf"))[-1]
        uov_path = sorted(run.glob("*.uov_full.*.athdf"))[-1]
        prim, (x, y, z), t = read_volume(prim_path, "prim")
        uov, _, tu = read_volume(uov_path, "user_out_var")
        if abs(t-tu) > 1e-8 or abs(t-row["end_time_code"]) > 1e-6:
            raise ValueError("Snapshot and completed segment times disagree")
        rho = prim[0] * units["rho_unit"]
        temp = uov[2]
        if not (np.all(rho > 0) and np.all(temp > 0)):
            raise ValueError("Nonpositive density/temperature")
        vertices, weights, used = prepare_geometry(rho, temp, grey)
        all_vertices.update(map(int, used))
        snapshots.append(dict(time=t, rho=rho, temp=temp,
                              vertices=vertices, weights=weights, x=x, y=y, z=z,
                              prim=str(prim_path), uov=str(uov_path)))
    abs_sample, sca_sample, source_files = load_opacity_samples(
        mono_dir, np.asarray(sorted(all_vertices)), indices, points,
        grey_rosseland_log10)
    mu, mu_weight = np.polynomial.legendre.leggauss(3)
    mu = (mu+1)/2
    mu_weight = mu_weight/2
    azimuths = np.arange(4)*np.pi/2
    per_snapshot = []
    map_wavelengths = [400, 500, 800, 1600]
    map_indices = [int(np.argmin(abs(wavelengths_nm-w))) for w in map_wavelengths]
    for snapshot in snapshots:
        rho, temp = snapshot["rho"], snapshot["temp"]
        z, x, y = snapshot["z"], snapshot["x"], snapshot["y"]
        dz_cm = float(z[1]-z[0])*units["leng_unit"]
        dx_cm = float(x[1]-x[0])*units["leng_unit"]
        dy_cm = float(y[1]-y[0])*units["leng_unit"]
        flux = np.empty(len(wavelengths_nm))
        mu_intensity = np.empty((len(wavelengths_nm),len(mu)))
        top_tau = np.empty(len(wavelengths_nm))
        total_tau = np.empty(len(wavelengths_nm))
        scattering_fraction = np.empty(len(wavelengths_nm))
        maps = {}
        for il, wavelength in enumerate(wavelengths_nm):
            chi_abs = interpolate_extinction(abs_sample, snapshot["vertices"],
                                             snapshot["weights"], il, rho.shape)
            chi_sca = interpolate_extinction(sca_sample, snapshot["vertices"],
                                             snapshot["weights"], il, rho.shape)
            chi = chi_abs + chi_sca
            if not np.all(np.isfinite(chi)):
                raise ValueError("Nonfinite monochromatic extinction")
            top_tau[il] = float(np.mean(chi[-1]*dz_cm/2))
            total_tau[il] = float(np.mean(np.sum(chi*dz_cm, axis=0)))
            near_surface = np.argmin(abs(np.cumsum((chi*dz_cm)[::-1],axis=0)[::-1]-1),axis=0)
            scatter_surface = np.take_along_axis(chi_sca/chi,
                                                  near_surface[None],axis=0)[0]
            scattering_fraction[il] = np.mean(scatter_surface)
            if il in map_indices:
                maps[il] = {}
            for im, angle in enumerate(mu):
                ray_mean = 0.0
                for az in azimuths:
                    intensity = formal_ray(temp, chi, wavelength, dz_cm, dx_cm,
                                           dy_cm, float(angle), float(az),
                                           args.substeps)
                    ray_mean += float(np.mean(intensity))/len(azimuths)
                    if il in maps and az == 0:
                        maps[il][im] = intensity
                mu_intensity[il, im] = ray_mean
            flux[il] = 2*np.pi*np.dot(mu_weight*mu,mu_intensity[il])
        per_snapshot.append(dict(time=float(snapshot["time"]),
                                 flux=flux, mu_intensity=mu_intensity,
                                 top_tau=top_tau,
                                 total_tau=total_tau,
                                 scattering_fraction=scattering_fraction,
                                 maps=maps, source_prim=snapshot["prim"],
                                 source_uov=snapshot["uov"]))
        print(f"completed t={snapshot['time']:.0f}", flush=True)
    flux_stack = np.stack([s["flux"] for s in per_snapshot])
    flux_mean = np.mean(flux_stack,axis=0)
    flux_std = np.std(flux_stack,axis=0)
    ref_nm, ref_flux = solar_reference(args.solar_reference)
    ref_at_sample = np.interp(wavelengths_nm, ref_nm, ref_flux,
                              left=np.nan,right=np.nan)
    captured_model = float(np.trapezoid(flux_mean[optical_band],
                                        wavelengths_nm[optical_band]))
    quasi_bolometric_model = float(np.trapezoid(flux_mean,wavelengths_nm))
    per_snapshot_band_flux = [float(np.trapezoid(s["flux"][optical_band],
                                                wavelengths_nm[optical_band]))
                              for s in per_snapshot]
    captured_solar = float(np.trapezoid(ref_flux[(ref_nm>=300)&(ref_nm<=2500)],
                                        ref_nm[(ref_nm>=300)&(ref_nm<=2500)]))
    model_teff_lower = (captured_model/SIGMA)**.25
    quasi_bolometric_teff = (quasi_bolometric_model/SIGMA)**.25
    blackbody_flux=np.pi*planck_nm(wavelengths_nm,quasi_bolometric_teff)
    blackbody_band=float(np.trapezoid(blackbody_flux[optical_band],
                                     wavelengths_nm[optical_band]))
    mean_fld_top=float(np.mean([r["top_radiative_flux_over_solar"]
                                for r in completed]))
    fld_teff=(mean_fld_top*FSUN/SIGMA)**.25
    dwarf_reference=None
    if args.dwarf_sequence:
        source=args.dwarf_sequence.resolve()
        values={}
        for line in source.read_text().splitlines():
            fields=line.split()
            if fields and fields[0] in ("K1V","K2V"):
                values[fields[0]]=float(fields[1])
        if set(values)!={"K1V","K2V"}:
            raise ValueError("Dwarf reference lacks K1V/K2V rows")
        dwarf_reference={"path":str(source),"sha256":sha256(source),
                         "url":"https://github.com/emamajek/SpectralType/blob/master/EEM_dwarf_UBVIJHK_colors_Teff.txt",
                         "teff_K":values}
    def band_mean(lo: float, hi: float, values: np.ndarray) -> float:
        selected=(wavelengths_nm>=lo)&(wavelengths_nm<=hi)
        if np.count_nonzero(selected)<2:
            raise ValueError(f"Too few wavelength points in {lo}–{hi} nm")
        return float(np.trapezoid(values[selected],wavelengths_nm[selected])/
                     np.ptp(wavelengths_nm[selected]))
    flux_color_ratio = band_mean(400,600,flux_mean)/band_mean(800,1200,flux_mean)
    solar_color_ratio = band_mean(400,600,ref_at_sample)/band_mean(800,1200,ref_at_sample)
    blackbody_color_ratio=(band_mean(400,600,blackbody_flux)/
                           band_mean(800,1200,blackbody_flux))
    # Scattering sensitivity at 500 nm: use the same geometry and angles.
    last = snapshots[-1]
    il = map_indices[1]
    chi_abs = interpolate_extinction(abs_sample,last["vertices"],last["weights"],il,last["rho"].shape)
    chi_sca = interpolate_extinction(sca_sample,last["vertices"],last["weights"],il,last["rho"].shape)
    dz = float(last["z"][1]-last["z"][0])*units["leng_unit"]
    dx = float(last["x"][1]-last["x"][0])*units["leng_unit"]
    dy = float(last["y"][1]-last["y"][0])*units["leng_unit"]
    no_sca_flux = 0.0
    for im, angle in enumerate(mu):
        for az in azimuths:
            no_sca_flux += (2*np.pi*mu_weight[im]*angle/len(azimuths)
                            * np.mean(formal_ray(last["temp"],chi_abs,
                                                 float(wavelengths_nm[il]),
                                                 dz,dx,dy,float(angle),float(az),
                                                 args.substeps)))
    scatter_sensitivity = float((per_snapshot[-1]["flux"][il]-no_sca_flux)/
                                no_sca_flux)
    def flux_with_substeps(nsub: int) -> float:
        return float(sum(2*np.pi*mu_weight[im]*angle/len(azimuths)
                         *np.mean(formal_ray(last["temp"],chi_abs+chi_sca,
                                             float(wavelengths_nm[il]),dz,dx,dy,
                                             float(angle),float(az),nsub))
                         for im,angle in enumerate(mu) for az in azimuths))
    substep_fluxes = {str(n): flux_with_substeps(n)
                      for n in sorted({1,2,4,8,args.substeps})}
    substep_error = abs(substep_fluxes[str(args.substeps)]-
                        substep_fluxes["8"])/substep_fluxes["8"]
    fine_mu,fine_weight=np.polynomial.legendre.leggauss(5)
    fine_mu=(fine_mu+1)/2
    fine_weight=fine_weight/2
    angular_fine_flux=float(sum(2*np.pi*fine_weight[im]*angle/len(azimuths)
                               *np.mean(formal_ray(last["temp"],chi_abs+chi_sca,
                                                   float(wavelengths_nm[il]),dz,dx,dy,
                                                   float(angle),float(az),8))
                               for im,angle in enumerate(fine_mu) for az in azimuths))
    angular_error=abs(substep_fluxes["8"]-angular_fine_flux)/angular_fine_flux
    band_idx=np.flatnonzero(optical_band)
    sparse_idx = np.unique(np.r_[band_idx[::2],band_idx[-1]])
    sparse_band_flux = float(np.trapezoid(flux_mean[sparse_idx],
                                          wavelengths_nm[sparse_idx]))
    spectral_sampling_difference = abs(sparse_band_flux-captured_model)/captured_model
    with (output/"spectrum.csv").open("w",newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength_nm","model_flux_erg_s_cm2_nm",
                         "snapshot_std_erg_s_cm2_nm","solar_reference_flux_erg_s_cm2_nm",
                         "top_half_cell_tau_mean","scattering_fraction_near_tau1"])
        writer.writerows(zip(wavelengths_nm,flux_mean,flux_std,ref_at_sample,
                             np.mean([s["top_tau"] for s in per_snapshot],axis=0),
                             np.mean([s["scattering_fraction"] for s in per_snapshot],axis=0)))
    np.savez_compressed(output/"angular_intensity.npz",wavelength_nm=wavelengths_nm,
                        mu=mu,time_code=np.array([s["time"] for s in per_snapshot]),
                        intensity=np.stack([s["mu_intensity"] for s in per_snapshot]),
                        flux=flux_stack)
    plt.rcParams.update({"font.size":16,"axes.titlesize":18})
    fig, ax = plt.subplots(figsize=(13,7),layout="constrained")
    ax.plot(wavelengths_nm[optical_band],flux_mean[optical_band],
            label="3D LTE post-process",lw=2.5)
    ax.fill_between(wavelengths_nm[optical_band],
                    (flux_mean-flux_std)[optical_band],
                    (flux_mean+flux_std)[optical_band],
                    alpha=.22,label="snapshot standard deviation")
    ax.plot(ref_nm,ref_flux,label="CALSPEC Sun at photosphere",lw=1.5,alpha=.8)
    top=np.mean([s["top_tau"] for s in per_snapshot],axis=0)
    affected=(top>0.1)&optical_band
    ax.scatter(wavelengths_nm[affected],flux_mean[affected],s=90,marker="x",
               color="crimson",label="top boundary τ > 0.1")
    ax.set(xlim=(300,2500),xlabel="Wavelength [nm]",
           ylabel=r"Surface flux [erg s$^{-1}$ cm$^{-2}$ nm$^{-1}$]",
           title="Emergent spectral flux: solar-box synthesis and observed Sun")
    ax.grid(alpha=.25);ax.legend();fig.savefig(images/"spectral_flux.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6),layout="constrained")
    ax.plot(wavelengths_nm[optical_band],flux_mean[optical_band]/captured_model,
            label="Simulation: normalized 300–2500 nm")
    ax.plot(ref_nm,ref_flux/captured_solar,label="Sun: normalized 300–2500 nm")
    ax.plot(wavelengths_nm[optical_band],
            blackbody_flux[optical_band]/blackbody_band,ls="--",
            label=f"Blackbody {quasi_bolometric_teff:.0f} K")
    ax.set(xlim=(300,2500),xlabel="Wavelength [nm]",ylabel="Normalized spectral shape [nm$^{-1}$]",
           title="Spectral shape comparison (same band integral)")
    ax.grid(alpha=.25);ax.legend();fig.savefig(images/"spectral_shape.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6),layout="constrained")
    for w in map_wavelengths:
        il=int(np.argmin(abs(wavelengths_nm-w)))
        ratio=np.mean([s["mu_intensity"][il] for s in per_snapshot],axis=0)
        ratio/=ratio[-1]
        ax.plot(mu,ratio,"o-",label=f"{wavelengths_nm[il]:.0f} nm")
    ax.set(xlabel=r"View angle $\mu=\cos\theta$",ylabel=r"Mean $I_\lambda(\mu)/I_\lambda(\mu_{\rm max})$",
           title="Center-to-limb intensity variation")
    ax.grid(alpha=.25);ax.legend();fig.savefig(images/"limb_darkening.png",dpi=160);plt.close(fig)
    last_maps=per_snapshot[-1]["maps"]
    fig,axes=plt.subplots(2,2,figsize=(13,11),layout="constrained")
    for ax,w in zip(axes.flat,map_wavelengths):
        il=int(np.argmin(abs(wavelengths_nm-w)))
        image=last_maps[il][2]
        im=ax.imshow(image,origin="lower",cmap="inferno",
                     extent=[last["x"][0]*units["leng_unit"]/1e8,
                             last["x"][-1]*units["leng_unit"]/1e8,
                             last["y"][0]*units["leng_unit"]/1e8,
                             last["y"][-1]*units["leng_unit"]/1e8])
        ax.set(title=f"{wavelengths_nm[il]:.0f} nm, μ={mu[2]:.3f}",
               xlabel="x [Mm]",ylabel="y [Mm]")
        fig.colorbar(im,ax=ax,label="Intensity / nm")
    fig.savefig(images/"intensity_maps.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout="constrained")
    sca=np.mean([s["scattering_fraction"] for s in per_snapshot],axis=0)
    axes[0].plot(wavelengths_nm[optical_band],sca[optical_band])
    axes[0].set(xlabel="Wavelength [nm]",ylabel="Scattering / total extinction near τ≈1",
                title="Scattering importance")
    axes[1].semilogy(wavelengths_nm[optical_band],top[optical_band])
    axes[1].set(xlabel="Wavelength [nm]",ylabel="Top half-cell optical depth",
                title="Upper-boundary optical depth")
    for ax in axes:ax.grid(alpha=.25)
    fig.savefig(images/"transfer_limits.png",dpi=160);plt.close(fig)
    manifest={
        "method":"3D LTE piecewise-isothermal formal solution, periodic horizontal rays; "
                 "total extinction and thermalized scattering source B_lambda",
        "campaign":str(campaign),"campaign_status":state["status"],
        "snapshot_times_code":[s["time"] for s in per_snapshot],
        "source_snapshots":[{"prim":s["source_prim"],"uov":s["source_uov"]} for s in per_snapshot],
        "grey_table":str(grey),"grey_table_sha256":sha256(grey),
        "optab_mono_dir":str(mono_dir),"optab_files":source_files,
        "solar_reference":str(args.solar_reference.resolve()),
        "solar_reference_url":"https://ssb.stsci.edu/cdbs/current_calspec/sun_reference_stis_002.fits",
        "solar_reference_sha256":sha256(args.solar_reference),
        "wavelengths":len(wavelengths_nm),
        "wavelengths_in_300_2500_nm_band":int(np.count_nonzero(optical_band)),
        "dwarf_temperature_reference":dwarf_reference,"mu":mu.tolist(),
        "mu_weight":mu_weight.tolist(),"azimuths_rad":azimuths.tolist(),
        "vertical_substeps":args.substeps,
        "spectral_band_nm":[300,2500],
        "model_band_flux_erg_s_cm2":captured_model,
        "model_band_flux_over_fsun":captured_model/FSUN,
        "model_100_10000_nm_quasi_bolometric_flux_over_fsun":quasi_bolometric_model/FSUN,
        "model_100_10000_nm_quasi_bolometric_teff_K":quasi_bolometric_teff,
        "per_snapshot_band_flux_over_fsun":[v/FSUN for v in per_snapshot_band_flux],
        "mean_fld_top_flux_over_fsun":mean_fld_top,
        "fld_flux_effective_temperature_K":fld_teff,
        "solar_reference_band_flux_erg_s_cm2":captured_solar,
        "band_limited_model_teff_lower_bound_K":model_teff_lower,
        "model_400_600_to_800_1200_band_mean_flux_ratio":flux_color_ratio,
        "solar_400_600_to_800_1200_band_mean_flux_ratio":solar_color_ratio,
        "matched_blackbody_400_600_to_800_1200_band_mean_flux_ratio":blackbody_color_ratio,
        "500nm_thermalized_scattering_vs_absorption_only_fraction":scatter_sensitivity,
        "500nm_substep_flux_erg_s_cm2_nm":substep_fluxes,
        "500nm_substep_relative_error_vs_8":substep_error,
        "500nm_mu3_vs_mu5_relative_difference_at_8_substeps":angular_error,
        "band_flux_sparse_grid_relative_difference":spectral_sampling_difference,
        "max_mean_top_half_cell_tau_in_300_2500_nm":float(np.max(top[optical_band])),
        "wavelengths_with_top_half_cell_tau_over_0p1_nm":wavelengths_nm[affected].tolist(),
        "min_mean_total_column_tau":float(np.min(np.mean(
            [s["total_tau"] for s in per_snapshot],axis=0))),
        "max_mean_scattering_fraction_near_tau1_in_300_2500_nm":float(np.max(sca[optical_band])),
    }
    (output/"summary.json").write_text(json.dumps(manifest,indent=2)+"\n")
    note=("This LTE formal solution uses grey-FLD thermodynamic snapshots. "
          "Scattering is treated as thermalized, velocities are omitted from line shifts, "
          "and opacity sampling is too sparse for precision line profiles. "
          "The 100–10000 nm integral is quasi-bolometric, while the 300–2500 nm "
          "integral is a band-limited lower bound. Strong UV/blue lines are "
          "limited by the finite upper atmosphere. "
          "Solar reference values are scaled from 1 AU using nominal AU and solar radius. "
          "The CALSPEC spectrum itself combines measured optical data and an infrared model.")
    image_tags="".join(f"<figure><img src='images/{p}' alt='{p}'><figcaption>{t}</figcaption></figure>" for p,t in (
        ("spectral_flux.png","Absolute emergent flux"),
        ("spectral_shape.png","Normalized spectral shape"),
        ("limb_darkening.png","Angular dependence"),
        ("intensity_maps.png","Surface intensity maps at four wavelengths"),
        ("transfer_limits.png","Scattering and upper-boundary checks")))
    formation_summary_path=output/"formation_summary.json"
    formation_data=(json.loads(formation_summary_path.read_text())
                    if formation_summary_path.is_file() else None)
    formation_ready=(formation_data is not None and
                     abs(formation_data["snapshot_time_code"]-
                         per_snapshot[-1]["time"])<1e-6)
    image_tags+="".join(
        f"<figure><img src='images/{name}' alt='{name}'><figcaption>{title}</figcaption></figure>"
        for name,title in (("formation_height.png","波長別の τ=1 高度"),
                           ("contribution_functions.png","鉛直方向の強度への寄与"),
                           ("vertical_resolution_sensitivity.png",
                            "固定スナップショットでの格子粗視化感度"))
        if formation_ready and (images/name).is_file())
    coarsening_note=""
    if formation_ready and formation_data.get("fixed_snapshot_2x_coarsening"):
        values=formation_data["fixed_snapshot_2x_coarsening"]
        near500=min(values,key=lambda k:abs(float(k.removesuffix("nm"))-500))
        row=values[near500]
        coarsening_note=(
            "<p style='color:#9b2424'><strong>鉛直解像度に敏感:</strong> "
            f"{near500} の同一スナップショットを鉛直方向だけ 2 倍粗くすると、"
            f"合成流束は {100*row['vertical_only_relative_difference']:+.1f}% 変わる。"
            "水平のみの粗視化は "
            f"{100*row['horizontal_only_relative_difference']:+.2f}%。"
            "これは固定データの後処理感度であり、96 km HD 計算そのものの収束誤差ではない。"
            "XD の鉛直高解像度計算で再検証が必要。</p>")
    movie_tags="".join(
        f"<figure><video controls preload='metadata' poster='movies/{p}.png'>"
        f"<source src='movies/{p}.mp4' type='video/mp4'></video>"
        f"<figcaption>{title}</figcaption></figure>"
        for p,title in (("solar_convection_xz","xz 断面の温度・鉛直速度"),
                        ("solar_convection_xy","xy 断面の温度・鉛直速度"),
                        ("solar_convection_xy_fluct","xy 断面の平均を引いた鉛直速度"))
        if (output/"movies"/f"{p}.mp4").is_file())
    campaign_link=os.path.relpath(campaign/"index.html",output)
    movie_manifest_link=("<a href='movies/manifest.json'>動画の source manifest</a> ·"
                         if (output/"movies"/"manifest.json").is_file() else "")
    formation_link=("<a href='formation.csv'>波長別形成高度</a> · "
                    "<a href='contribution_functions.npz'>寄与関数</a> · "
                    "<a href='formation_summary.json'>形成高度の要約</a> ·"
                    if formation_ready else "")
    snapshot_rows="".join(
        f"<tr><td>{s['time']:.0f}</td><td>{per_snapshot_band_flux[i]/FSUN:.4f}</td>"
        f"<td>{completed[i]['top_radiative_flux_over_solar']:.4f}</td></tr>"
        for i,s in enumerate(per_snapshot))
    dwarf_row=(f"<tr><td>経験的な K1V / K2V 温度</td><td>"
               f"{dwarf_reference['teff_K']['K1V']:.0f} / "
               f"{dwarf_reference['teff_K']['K2V']:.0f} K (色・重力の一致を意味しない)</td></tr>"
               if dwarf_reference else "")
    dwarf_citation=("<li><a href='https://github.com/emamajek/SpectralType/blob/master/"
                    "EEM_dwarf_UBVIJHK_colors_Teff.txt'>Pecaut &amp; Mamajek dwarf sequence</a>"
                    " — K1V/K2V の温度目安</li>" if dwarf_reference else "")
    (output/"index.html").write_text(f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<title>WOR-109 太陽対流スナップショットの波長・角度別放射輸送</title>
<style>body{{font:22px/1.55 system-ui,sans-serif;max-width:1350px;margin:2rem auto;padding:0 1.5rem;color:#173044}}
h1{{font-size:2.1rem}}h2{{font-size:1.55rem;margin-top:2rem}}.key{{color:#0b5a6b;font-weight:700}}
.box{{background:#eef6f8;border-left:6px solid #14697e;padding:1rem 1.3rem}}
img,video{{width:100%;height:auto}}figure{{margin:2rem 0}}figcaption{{font-size:1rem;color:#475569}}
table{{border-collapse:collapse}}td,th{{border-bottom:1px solid #ccd5dc;padding:.5rem .8rem;text-align:left}}</style>
<h1>WOR-109: 太陽対流 3D スナップショットの放射輸送後処理</h1>
<p class="box"><span class="key">{len(per_snapshot)} 個の完成スナップショット</span> (t={per_snapshot[0]['time']:.0f}–{per_snapshot[-1]['time']:.0f})、
100–10000 nm の {len(wavelengths_nm)} 波長 × {len(mu)} 方向余弦 × {len(azimuths)} 方位角。<br>
300–2500 nm 積分放射流束は <span class="key">{captured_model/FSUN:.3f} F<sub>sun</sub></span>、
同帯域の観測太陽は {captured_solar/FSUN:.3f} F<sub>sun</sub>。
100–10000 nm の準ボロメトリック流束は <span class="key">{quasi_bolometric_model/FSUN:.3f} F<sub>sun</sub></span>。
その T<sub>eff</sub> 換算は {quasi_bolometric_teff:.0f} K。</p>
<h2>主な診断</h2><table><tr><th>量</th><th>結果</th></tr>
<tr><td>灰色 FLD 上端流束の T<sub>eff</sub> 換算</td><td>{fld_teff:.0f} K</td></tr>
{dwarf_row}
<tr><td>400–600 / 800–1200 nm の平均流束比</td><td>{flux_color_ratio:.3f} (計算) / {blackbody_color_ratio:.3f} (同 T<sub>eff</sub> 黒体) / {solar_color_ratio:.3f} (太陽)</td></tr>
<tr><td>500 nm: 散乱を不透明度に含めた影響</td><td>{100*scatter_sensitivity:+.2f}% (熱化近似対散乱無視)</td></tr>
<tr><td>500 nm: 8 分割に対する光線積分誤差</td><td>{100*substep_error:.2f}%</td></tr>
<tr><td>500 nm: μ 積分 3 点と 5 点の差</td><td>{100*angular_error:.2f}% (ともに 8 分割)</td></tr>
<tr><td>波長点を半分にした場合の帯域積分差</td><td>{100*spectral_sampling_difference:.2f}%</td></tr>
<tr><td>同期間の FLD 上端流束</td><td>{np.mean([r['top_radiative_flux_over_solar'] for r in completed]):.3f} F<sub>sun</sub></td></tr>
<tr><td>τ≈1 の散乱比率の最大値 (300–2500 nm)</td><td>{100*np.max(sca[optical_band]):.1f}%</td></tr>
<tr><td>最上部半セルの光学的厚さの最大値 (300–2500 nm)</td><td>{np.max(top[optical_band]):.3g}</td></tr></table>
<p>帯域平均の色は同じ放射温度の黒体に近い。この意味では、太陽との差の多くは今回の流束・放射温度が低いことと整合する。
黒体は線吸収や周辺減光を検証するモデルではない。</p>
<h2>終盤 3 スナップショット</h2><table><tr><th>t [code]</th><th>後処理 300–2500 nm / F<sub>sun</sub></th>
<th>FLD 上端 / F<sub>sun</sub></th></tr>{snapshot_rows}</table>
<p style="color:#9b2424"><strong>短波長の一部は上端で光学的に厚い:</strong>
{', '.join(f'{w:.1f}' for w in wavelengths_nm[affected])} nm の点で最上部半セル τ&gt;0.1。
これらの線の形成層は計算領域より上にも続くため、個別の線強度は信用できません。</p>
{coarsening_note}
<h2>可視化</h2>{image_tags}
<h2>対流の時間発展</h2>{movie_tags}
<h2>解釈の範囲</h2><p>{html.escape(note)}</p>
<p>再現用データ: <a href="spectrum.csv">spectrum.csv</a> ·
<a href="angular_intensity.npz">angular_intensity.npz</a> ·
<a href="summary.json">source manifest / summary.json</a> ·
{formation_link}
{movie_manifest_link}
<a href="{html.escape(campaign_link)}">元の対流計算のまとめ</a></p>
<h2>文献とデータ</h2><ul>
<li><a href="https://doi.org/10.1051/0004-6361/202141076">Hirose et al. (2022), Optab</a> — 単色吸収・線・散乱係数</li>
<li><a href="https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec.html">STScI CALSPEC solar reference</a> — 太陽の比較スペクトル</li>
<li><a href="https://doi.org/10.1051/0004-6361/201220468">Ibgui et al. (2013), IRIS</a> — 3D LTE formal transfer</li>
{dwarf_citation}
<li><a href="https://arxiv.org/abs/1405.6814">Rempel (2014)</a> — 太陽対流境界と解像度</li></ul>
</html>""")
    print(output/"index.html")


if __name__ == "__main__":
    main()
