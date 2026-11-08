#!/usr/bin/env python3
"""Locate monochromatic tau=1 surfaces and vertical contribution functions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from solar_convection_spectral_postprocess import (
    formal_ray, interpolate_extinction, load_opacity_samples, opacity_index_grid,
    planck_nm, prepare_geometry, read_volume, units_from_input,
)


def tau_surface(tau: np.ndarray, z_mm: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate tau=1 between adjacent cell centers in every column."""
    nz, ny, nx = tau.shape
    below=np.count_nonzero(tau>=1,axis=0)
    valid=(below>0)&(below<nz)
    lower=np.clip(below-1,0,nz-2)
    upper=lower+1
    tau_lo=np.take_along_axis(tau,lower[None],axis=0)[0]
    tau_hi=np.take_along_axis(tau,upper[None],axis=0)[0]
    frac=(1-tau_lo)/np.where(tau_hi!=tau_lo,tau_hi-tau_lo,np.nan)
    height=z_mm[lower]+frac*(z_mm[upper]-z_mm[lower])
    return np.where(valid,height,np.nan),valid


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path,
                        help="Directory made by solar_convection_spectral_postprocess.py")
    args=parser.parse_args()
    out=args.output.resolve()
    summary=json.loads((out/"summary.json").read_text())
    campaign=Path(summary["campaign"])
    units=units_from_input(campaign/"athinput.base.validated")
    source=summary["source_snapshots"][-1]
    prim,(x,y,z),t=read_volume(Path(source["prim"]),"prim")
    uov,_,tu=read_volume(Path(source["uov"]),"user_out_var")
    if abs(t-tu)>1e-8 or abs(t-summary["snapshot_times_code"][-1])>1e-6:
        raise ValueError("Last snapshot time changed")
    rho=prim[0]*units["rho_unit"]
    temp=uov[2]
    grey=Path(summary["grey_table"])
    mono=Path(summary["optab_mono_dir"])
    vertices,weights,used=prepare_geometry(rho,temp,grey)
    with h5py.File(grey) as source_table:
        points=np.column_stack((np.log10(source_table["source_points/density"][:]),
                                np.log10(source_table["source_points/temperature"][:])))
        rosseland=source_table["source_points/log10_rosseland"][:]
    spectrum=np.genfromtxt(out/"spectrum.csv",delimiter=",",skip_header=1)
    wavelengths=spectrum[:,0]
    indices,actual=opacity_index_grid(mono,wavelengths)
    if not np.allclose(actual,wavelengths,rtol=1e-10):
        raise ValueError("Spectral grid differs from source summary")
    abs_sample,sca_sample,_=load_opacity_samples(mono,used,indices,points,rosseland)
    dz_cm=float(z[1]-z[0])*units["leng_unit"]
    dx_cm=float(x[1]-x[0])*units["leng_unit"]
    dy_cm=float(y[1]-y[0])*units["leng_unit"]
    z_mm=z*units["leng_unit"]/1e8
    chosen=[int(np.argmin(abs(wavelengths-w))) for w in (400,500,800,1600)]
    contribution={}
    coarsening={}
    mu,mu_weight=np.polynomial.legendre.leggauss(3)
    mu=(mu+1)/2
    mu_weight=mu_weight/2
    azimuths=np.arange(4)*np.pi/2
    def angular_flux(temperature: np.ndarray,extinction: np.ndarray,
                     wavelength: float,dz: float,dx: float,dy: float) -> float:
        return float(sum(2*np.pi*mu_weight[im]*angle/len(azimuths)
                         *np.mean(formal_ray(temperature,extinction,wavelength,
                                             dz,dx,dy,float(angle),float(az),2))
                         for im,angle in enumerate(mu) for az in azimuths))
    nz,ny,nx=temp.shape
    if any(n%2 for n in (nz,ny,nx)):
        raise ValueError("Coarsening diagnostic needs even grid dimensions")
    coarse_temp=temp.reshape(nz//2,2,ny//2,2,nx//2,2).mean(axis=(1,3,5))
    horizontal_temp=temp.reshape(nz,ny//2,2,nx//2,2).mean(axis=(2,4))
    vertical_temp=temp.reshape(nz//2,2,ny,nx).mean(axis=1)
    rows=[]
    for il,wavelength in enumerate(wavelengths):
        chi=(interpolate_extinction(abs_sample,vertices,weights,il,rho.shape)+
             interpolate_extinction(sca_sample,vertices,weights,il,rho.shape))
        dtau=chi*dz_cm
        tau_above=np.cumsum(dtau[::-1],axis=0)[::-1]-dtau
        tau_center=tau_above+0.5*dtau
        surface,valid=tau_surface(tau_center,z_mm)
        rows.append((wavelength,float(np.mean(valid)),
                     float(np.nanmean(surface)) if np.any(valid) else np.nan,
                     float(np.nanstd(surface)) if np.any(valid) else np.nan,
                     float(np.mean(tau_center[-1]))))
        if il in chosen:
            source_function=planck_nm(wavelength,temp)
            term=source_function*(-np.expm1(-dtau))*np.exp(-tau_above)
            vertical=np.mean(term,axis=(1,2))
            bottom=np.mean(source_function[0]*np.exp(-np.sum(dtau,axis=0)))
            formal_mean=float(np.mean(formal_ray(temp,chi,wavelength,dz_cm,
                                                 dx_cm,dy_cm,1.0,0.0,1)))
            if not np.isfinite(vertical).all() or not np.isclose(
                    np.sum(vertical)+bottom,formal_mean,rtol=1e-10):
                raise ValueError("Invalid contribution function")
            contribution[il]=vertical/np.sum(vertical)
        if il in (chosen[1],chosen[3]):
            coarse_chi=chi.reshape(nz//2,2,ny//2,2,nx//2,2).mean(axis=(1,3,5))
            horizontal_chi=chi.reshape(nz,ny//2,2,nx//2,2).mean(axis=(2,4))
            vertical_chi=chi.reshape(nz//2,2,ny,nx).mean(axis=1)
            full_flux=angular_flux(temp,chi,wavelength,dz_cm,dx_cm,dy_cm)
            coarse_flux=angular_flux(coarse_temp,coarse_chi,wavelength,
                                     2*dz_cm,2*dx_cm,2*dy_cm)
            horizontal_flux=angular_flux(horizontal_temp,horizontal_chi,wavelength,
                                         dz_cm,2*dx_cm,2*dy_cm)
            vertical_flux=angular_flux(vertical_temp,vertical_chi,wavelength,
                                       2*dz_cm,dx_cm,dy_cm)
            coarsening[f"{wavelength:.1f}nm"]={
                "full_flux_erg_s_cm2_nm":full_flux,
                "coarsened_flux_erg_s_cm2_nm":coarse_flux,
                "relative_difference":(coarse_flux-full_flux)/full_flux,
                "horizontal_only_relative_difference":
                    (horizontal_flux-full_flux)/full_flux,
                "vertical_only_relative_difference":
                    (vertical_flux-full_flux)/full_flux}
    with (out/"formation.csv").open("w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow(["wavelength_nm","tau1_coverage","tau1_mean_z_Mm",
                         "tau1_std_z_Mm","top_half_cell_tau_mean"])
        writer.writerows(rows)
    np.savez_compressed(out/"contribution_functions.npz",
                        wavelength_nm=wavelengths[chosen],z_Mm=z_mm,
                        normalized_contribution=np.stack([contribution[i]
                                                          for i in chosen]))
    arr=np.asarray(rows)
    lo=arr[np.argmin(abs(arr[:,0]-300)),0]
    hi=arr[np.argmin(abs(arr[:,0]-2500)),0]
    optical=(arr[:,0]>=lo)&(arr[:,0]<=hi)
    plt.rcParams.update({"font.size":16,"axes.titlesize":18})
    fig,ax=plt.subplots(figsize=(12,6),layout="constrained")
    trusted=optical&(arr[:,1]>.95)
    ax.plot(arr[trusted,0],arr[trusted,2],"o-",ms=3,
            label="mean monochromatic τ=1 height")
    bad=optical&(arr[:,1]<=.95)
    if np.any(bad):
        ax.scatter(arr[bad,0],np.full(np.count_nonzero(bad),z_mm[-1]),
                   marker="x",color="crimson",s=80,
                   label="τ=1 above the simulated top")
    ax.set(xlabel="Wavelength [nm]",ylabel="Height [Mm]",
           title="Formation height from vertical optical depth")
    ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/"images"/"formation_height.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6),layout="constrained")
    for il in chosen:
        ax.plot(z_mm,contribution[il],"o-",ms=3,
                label=f"{wavelengths[il]:.0f} nm")
    ax.set(xlabel="Height [Mm]",ylabel="Fractional contribution per z-cell",
           title="Contribution to disk-center emergent intensity")
    ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/"images"/"contribution_functions.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6),layout="constrained")
    labels=list(coarsening)
    xx=np.arange(len(labels))
    width=.23
    for offset,key,title in ((-1,"horizontal_only_relative_difference","horizontal only"),
                             (0,"vertical_only_relative_difference","vertical only"),
                             (1,"relative_difference","all axes")):
        ax.bar(xx+offset*width,
               [100*coarsening[name][key] for name in labels],
               width=width,label=title)
    ax.axhline(0,color="black",lw=.8)
    ax.set(xticks=xx,xticklabels=labels,
           ylabel="Flux change after 2× coarsening [%]",
           title="Fixed-snapshot transfer sensitivity to cell size")
    ax.grid(axis="y",alpha=.25);ax.legend()
    fig.savefig(out/"images"/"vertical_resolution_sensitivity.png",dpi=160)
    plt.close(fig)
    metrics={"snapshot_time_code":t,
             "wavelength_count":len(wavelengths),
             "optical_band_wavelengths_with_tau1_inside":int(np.count_nonzero(trusted)),
             "optical_band_wavelength_count":int(np.count_nonzero(optical)),
             "top_invalid_wavelengths_nm":arr[bad,0].tolist(),
             "fixed_snapshot_2x_coarsening":coarsening,
             "coarsening_scope":"Averaging fixed T and extinction over 2x2x2 cells "
                                "tests synthesis sensitivity, not hydro grid convergence",
             "method":"Vertical tau=1 interpolated between cell centers; "
                      "LTE contribution B_lambda(1-exp(-dtau)) exp(-tau_above)",
             "source_spectral_summary":str(out/"summary.json")}
    (out/"formation_summary.json").write_text(json.dumps(metrics,indent=2)+"\n")
    report=out/"index.html"
    if report.is_file():
        page=report.read_text()
        if any(name not in page for name in (
                "formation_height.png","contribution_functions.png",
                "vertical_resolution_sensitivity.png")):
            figures="".join(
                f"<figure><img src='images/{name}' alt='{name}'>"
                f"<figcaption>{title}</figcaption></figure>"
                for name,title in (("formation_height.png","波長別の τ=1 高度"),
                                   ("contribution_functions.png","鉛直方向の強度への寄与"),
                                   ("vertical_resolution_sensitivity.png",
                                    "固定スナップショットでの格子粗視化感度"))
                if name not in page)
            page=page.replace("<h2>対流の時間発展</h2>",
                              figures+"<h2>対流の時間発展</h2>",1)
        if "鉛直解像度に敏感" not in page:
            near500=min(coarsening,key=lambda k:abs(float(k.removesuffix("nm"))-500))
            row=coarsening[near500]
            caution=("<p style='color:#9b2424'><strong>鉛直解像度に敏感:</strong> "
                     f"{near500} の同一スナップショットを鉛直方向だけ 2 倍粗くすると、"
                     f"合成流束は {100*row['vertical_only_relative_difference']:+.1f}% 変わる。"
                     "水平のみの粗視化は "
                     f"{100*row['horizontal_only_relative_difference']:+.2f}%。"
                     "これは固定データの後処理感度であり、96 km HD 計算そのものの収束誤差ではない。"
                     "XD の鉛直高解像度計算で再検証が必要。</p>")
            page=page.replace("<h2>可視化</h2>",caution+"<h2>可視化</h2>",1)
        if "href='formation.csv'" not in page:
            page=page.replace("<a href=\"summary.json\">source manifest / summary.json</a> ·",
                              "<a href=\"summary.json\">source manifest / summary.json</a> ·"
                              "<a href='formation.csv'>波長別形成高度</a> · "
                              "<a href='contribution_functions.npz'>寄与関数</a> · "
                              "<a href='formation_summary.json'>形成高度の要約</a> ·",1)
        report.write_text(page)
    print(json.dumps({"time":t,"optical_coverage":
                      f"{metrics['optical_band_wavelengths_with_tau1_inside']}/"
                      f"{metrics['optical_band_wavelength_count']}",
                      "invalid_nm":metrics["top_invalid_wavelengths_nm"]}))


if __name__=="__main__":
    main()
