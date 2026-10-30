#!/usr/bin/env python3
"""Build an RSG HSE with a convective interior and spherical gray atmosphere.

Gas hydrostatic balance includes the outward Rosseland radiation acceleration.
The interior temperature gradient is the smaller of the FLD diffusion gradient
and nabla_ad plus a configurable superadiabatic excess.  At the requested
radius tau=2/3 and T=Teff; outside, the spherical Eddington gray relation is
used down to tau=0.  The emitted luminosity follows the same smoothed core
heating kernel as nrfld_star_in_box.cpp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import brentq

from make_hse_model import AthenaEos, mapped_mass, potential, gravity

G = 6.67430e-8
MSUN = 1.98847e33
RSUN = 6.957e10
LSUN = 3.828e33
CLIGHT = 2.99792458e10
AR = 7.5657e-15


class Opacity:
    def __init__(self, path: Path):
        with h5py.File(path, "r") as handle:
            self.logt = np.asarray(handle["axes/log10_temperature"])
            self.logrho = np.asarray(handle["axes/log10_density"])
            self.logp = np.asarray(handle["kappa/log10_planck"])
            self.logr = np.asarray(handle["kappa/log10_rosseland"])
        self.planck_interp = RegularGridInterpolator(
            (self.logrho, self.logt), self.logp, bounds_error=True)
        self.rosseland_interp = RegularGridInterpolator(
            (self.logrho, self.logt), self.logr, bounds_error=True)

    def rosseland(self, density, temperature):
        return 10.0 ** float(self.rosseland_interp(
            (np.log10(density), np.log10(temperature))))

    def planck(self, density, temperature):
        return 10.0 ** float(self.planck_interp(
            (np.log10(density), np.log10(temperature))))


def density_from_pt(eos, pressure, temperature):
    def residual(logrho):
        try:
            p, _ = eos.state_from_rho_temp(10.0**logrho, temperature)
        except ValueError as exc:
            raise ValueError(
                f"EOS rho-T lookup failed at logRho={logrho:.6g}, "
                f"T={temperature:.6e}, Ptarget={pressure:.6e}: {exc}") from exc
        return np.log(p / pressure)
    lo, hi = eos.rho_lim
    flo, fhi = residual(lo), residual(hi)
    if flo * fhi > 0.0:
        raise ValueError(
            f"P,T inversion outside EOS: P={pressure:.6e}, T={temperature:.6e}")
    logrho = brentq(residual, lo, hi, xtol=2e-10, rtol=2e-10)
    return 10.0**logrho


def nabla_ad_from_rho_u(eos, density, specific_u):
    if eos.nvar < 7:
        raise ValueError("gray RSG builder requires the seven-field EOS schema")
    logrho = np.log10(density)
    logu = np.log10(specific_u)
    logx2 = logu + np.log10(eos.ratios[6])
    return 10.0**eos.interp(eos.fields[6], logx2, logrho)


def entropy_from_rho_t(eos, density, temperature):
    _, energy=eos.state_from_rho_temp(density,temperature)
    logu=np.log10(energy/density);logrho=np.log10(density)
    return 10.0**eos.interp(eos.fields[4],
                            logu+np.log10(eos.ratios[4]),logrho)


def build_isentrope(eos, entropy, points=1024):
    rows=[]
    for lr in np.linspace(eos.rho_lim[0]+1e-8,eos.rho_lim[1]-1e-8,points):
        try:
            p,e,t,g=eos.state_from_rho_entropy(10.0**lr,entropy)
            if min(p,e,t,g)>0.0 and np.isfinite((p,e,t,g)).all():
                rows.append((np.log(p),lr,np.log(e),np.log(t),np.log(g)))
        except ValueError:
            pass
    if len(rows)<16: raise ValueError("entropy has insufficient EOS support")
    rows=np.asarray(sorted(rows,key=lambda x:x[0]))
    keep=np.r_[True,np.diff(rows[:,0])>1e-12]
    return rows[keep]


def state_from_isentrope(isentrope, pressure):
    lp=np.log(pressure)
    if lp<isentrope[0,0] or lp>isentrope[-1,0]:
        raise ValueError("P,s state leaves EOS domain")
    values=[np.interp(lp,isentrope[:,0],isentrope[:,i]) for i in range(1,5)]
    return 10.0**values[0],np.exp(values[1]),np.exp(values[2]),np.exp(values[3])


def state_derivatives(eos, density, entropy, step=2.0e-4):
    """Return a table-consistent state and logarithmic P/T derivatives.

    The exported entropy field and the independently exported ``nabla_ad``
    agree only to interpolation accuracy.  Integrating the latter over the
    enormous pressure range of an RSG amplifies a percent-level local error
    into an order-unity entropy drift.  Derivatives of the same discrete
    entropy inversion used for the profile avoid that accumulated mismatch.
    """
    pressure, energy, temperature, _ = eos.state_from_rho_entropy(
        density, entropy)
    rm, rp = density*np.exp(-step), density*np.exp(step)
    sm, sp = entropy*np.exp(-step), entropy*np.exp(step)
    pm, _, tm, _ = eos.state_from_rho_entropy(rm, entropy)
    pp, _, tp, _ = eos.state_from_rho_entropy(rp, entropy)
    psm, _, tsm, _ = eos.state_from_rho_entropy(density, sm)
    psp, _, tsp, _ = eos.state_from_rho_entropy(density, sp)
    return (pressure, energy, temperature,
            (pp-pm)/(2.0*step), (psp-psm)/(2.0*step),
            np.log(tp/tm)/(2.0*step), np.log(tsp/tsm)/(2.0*step))


def heating_fraction(radius, heating_radius):
    if radius <= 0.0:
        return 0.0
    x = min(radius, heating_radius)
    integral = (x**3/3.0 - 2.0*x**5/(5.0*heating_radius**2)
                + x**7/(7.0*heating_radius**4))
    norm = heating_radius**3 * (1.0/3.0 - 2.0/5.0 + 1.0/7.0)
    return integral / norm


def photosphere_state(args, eos, opacity):
    """Solve the tau=2/3 gas-pressure closure at T=Teff."""
    radius = args.radius_rsun * RSUN
    gm = G * args.potential_mass_msun * MSUN
    r0, r1 = args.r0_fraction * radius, args.r1_fraction * radius
    luminosity = args.luminosity_lsun * LSUN
    temperature = args.effective_temperature
    grav = gravity(radius, gm, r0, r1)
    flux = luminosity/(4.0*np.pi*radius*radius)

    def residual(logrho):
        density = 10.0**logrho
        pressure, _ = eos.state_from_rho_temp(density, temperature)
        kappa = opacity.rosseland(density, temperature)
        geff = grav-kappa*flux/CLIGHT
        if geff <= 0.0:
            return np.inf
        return np.log(pressure*kappa/(args.photosphere_tau*geff))

    lo = max(eos.rho_lim[0]+1.0e-6, opacity.logrho[0]+1.0e-6)
    hi = min(eos.rho_lim[1]-1.0e-6, opacity.logrho[-1]-1.0e-6)
    scan = np.linspace(lo, hi, 256)
    values = [(x, residual(x)) for x in scan]
    bracket = next(((x0, x1) for (x0, f0), (x1, f1)
                    in zip(values, values[1:])
                    if np.isfinite(f0) and np.isfinite(f1) and f0*f1 <= 0.0), None)
    if bracket is None:
        raise RuntimeError(f"cannot bracket tau photosphere: {values}")
    logrho = brentq(residual, *bracket, xtol=2e-10, rtol=2e-10)
    density = 10.0**logrho
    pressure, energy = eos.state_from_rho_temp(density, temperature)
    entropy = entropy_from_rho_t(eos, density, temperature)
    return density, pressure, energy, temperature, entropy


def interior_solution(args, eos, opacity, superadiabatic_peak, photosphere):
    """Integrate radiative/convective structure inward from the photosphere.

    The radiative gradient is used where stable.  In unstable layers the
    gradient is the table-consistent adiabatic value plus a Gaussian surface
    excess.  Its finite width represents the unresolved superadiabatic layer;
    the peak is shot to the requested envelope mass rather than hidden in an
    arbitrary surface-density choice.
    """
    radius = args.radius_rsun * RSUN
    gm = G * args.potential_mass_msun * MSUN
    r0, r1 = args.r0_fraction * radius, args.r1_fraction * radius
    rheat = args.heating_radius_fraction*radius
    luminosity = args.luminosity_lsun*LSUN
    rmin = radius * 1.0e-8

    def rhs(r, y):
        pressure, temperature = np.exp(y)
        density = density_from_pt(eos, pressure, temperature)
        entropy = entropy_from_rho_t(eos, density, temperature)
        pcheck, _, _, pr, _, tr, _ = state_derivatives(eos, density, entropy)
        nabla_ad = pcheck*tr/pr
        depth = (radius-r)/(args.superadiabatic_width*radius)
        cap = nabla_ad+superadiabatic_peak*np.exp(-depth*depth)
        kappa = opacity.rosseland(density, temperature)
        grav = gravity(r, gm, r0, r1)
        total_flux = luminosity*heating_fraction(r, rheat)/(
            4.0*np.pi*max(r*r, rmin*rmin))
        gradp_rad = -density*(grav-kappa*total_flux/CLIGHT)
        dlnp_rad = gradp_rad/pressure
        dlnt_rad = -3.0*kappa*density*total_flux/(
            4.0*CLIGHT*AR*temperature**4)
        nabla_rad = dlnt_rad/dlnp_rad if dlnp_rad < 0.0 else np.inf
        if np.isfinite(nabla_rad) and nabla_rad <= cap:
            rad_flux, nabla = total_flux, nabla_rad
        else:
            bcoef = 4.0*CLIGHT*AR*temperature**4*cap/(3.0*kappa*pressure)
            rad_flux = np.clip(
                bcoef*grav/(1.0+bcoef*kappa/CLIGHT), 0.0, total_flux)
            nabla = cap
        dlnp = -density*(grav-kappa*rad_flux/CLIGHT)/pressure
        return dlnp, nabla*dlnp

    # The potential is regularized in the core, whereas the superadiabatic
    # layer changes rapidly near Rstar.  Uniform-r output resolves that layer;
    # a logarithmic grid would spend almost all samples in the smooth core and
    # leave only a handful across the photospheric transition.
    descending = np.linspace(radius, rmin, args.points)
    sol = solve_ivp(rhs, (radius, rmin),
                    (np.log(photosphere[1]), np.log(photosphere[3])),
                    t_eval=descending, rtol=args.rtol, atol=(1e-9,1e-9),
                    method="DOP853")
    if not sol.success:
        raise RuntimeError(sol.message)
    radii = sol.t[::-1]
    pressure = np.exp(sol.y[0,::-1])
    temperature = np.exp(sol.y[1,::-1])
    density = np.array([density_from_pt(eos,p,t)
                        for p,t in zip(pressure,temperature)])
    entropy = np.array([entropy_from_rho_t(eos,d,t)
                        for d,t in zip(density,temperature)])
    mass = cumulative_trapezoid(4.0*np.pi*radii*radii*density,radii,initial=0.0)
    mass += 4.0*np.pi*radii[0]**3*density[0]/3.0
    return (radii, density, pressure, temperature, mass), entropy


def shoot_interior(args, eos, opacity, mass_target):
    photosphere = photosphere_state(args, eos, opacity)
    def residual(peak):
        try:
            result, _ = interior_solution(args, eos, opacity, peak, photosphere)
            return np.log(result[4][-1]/mass_target)
        except (ValueError, RuntimeError, FloatingPointError):
            return np.nan

    trials = [0.0]
    trial = min(0.025, args.superadiabatic_peak_max)
    while trial < args.superadiabatic_peak_max:
        trials.append(trial)
        trial *= 2.0
    if trials[-1] != args.superadiabatic_peak_max:
        trials.append(args.superadiabatic_peak_max)
    samples = [(trial, residual(trial)) for trial in trials]
    bracket = next(((a, c) for (a, b), (c, d) in zip(samples, samples[1:])
                    if np.isfinite(b) and np.isfinite(d) and b*d <= 0.0), None)
    if bracket is None:
        raise RuntimeError(f"superadiabatic shooting failed: {samples}")
    solved = brentq(residual, *bracket, xtol=2e-7, rtol=2e-7)
    result, entropy = interior_solution(args, eos, opacity, solved, photosphere)
    central = np.array([np.log10(result[1][0]), np.log10(result[3][0])])
    return result, central, photosphere, solved, entropy


def gray_temperature(teff, radius, r, tau):
    factor = (radius/r)**2 * 0.75 * (max(tau, 0.0) + 2.0/3.0)
    return teff * factor**0.25


def extend_atmosphere(args, eos, opacity, interior):
    ri, rhoi, pi, ti, mi = interior
    radius = args.radius_rsun * RSUN
    corner = np.sqrt(3.0) * args.box_half_rsun * RSUN
    gm = G * args.potential_mass_msun * MSUN
    r0, r1 = args.r0_fraction * radius, args.r1_fraction * radius
    luminosity = args.luminosity_lsun * LSUN
    rout = np.geomspace(radius*(1.0+1e-10), corner, args.atmosphere_points)
    state = np.array([np.log(pi[-1]), 2.0/3.0, mi[-1]])
    records = []

    def rhs(r, y):
        temperature = gray_temperature(args.effective_temperature, radius, r, y[1])
        pressure = np.exp(y[0])
        pressure_floor,_=eos.state_from_rho_temp(args.density_floor,temperature)
        pressure=max(pressure,pressure_floor)
        density=(args.density_floor if pressure==pressure_floor
                 else density_from_pt(eos, pressure, temperature))
        kappa = opacity.rosseland(density, temperature)
        flux = luminosity/(4.0*np.pi*r*r)
        return np.array([
            -density*(gravity(r,gm,r0,r1)-kappa*flux/CLIGHT)/pressure,
            -kappa*density,
            4.0*np.pi*r*r*density])

    previous = radius
    for r in rout:
        dr = r-previous
        t0=gray_temperature(args.effective_temperature,radius,previous,state[1])
        p0,_=eos.state_from_rho_temp(args.density_floor,t0)
        state[0]=max(state[0],np.log(p0))
        k1 = rhs(previous,state)
        midpoint = state + 0.5*dr*k1
        midpoint[1] = max(midpoint[1],0.0)
        tm=gray_temperature(args.effective_temperature,radius,previous+0.5*dr,midpoint[1])
        pm,_=eos.state_from_rho_temp(args.density_floor,tm)
        midpoint[0]=max(midpoint[0],np.log(pm))
        k2 = rhs(previous+0.5*dr,midpoint)
        state += dr*k2
        state[1] = max(state[1],0.0)
        temperature = gray_temperature(args.effective_temperature,radius,r,state[1])
        pressure = np.exp(state[0])
        pressure_floor,_=eos.state_from_rho_temp(args.density_floor,temperature)
        if pressure<=pressure_floor:
            density=args.density_floor;pressure=pressure_floor;state[0]=np.log(pressure)
        else:
            density=density_from_pt(eos,pressure,temperature)
        records.append((r,density,pressure,temperature,state[2],state[1]))
        previous = r
    records = np.asarray(records)
    return (np.r_[ri,records[:,0]], np.r_[rhoi,records[:,1]],
            np.r_[pi,records[:,2]], np.r_[ti,records[:,3]],
            np.r_[mi,records[:,4]], np.r_[np.full(ri.size,np.nan),records[:,5]])


def diagnostics(args, eos, opacity, profile):
    r,rho,p,t,m,tau = profile
    egas = np.array([eos.state_from_rho_temp(d,temp)[1]
                     for d,temp in zip(rho,t)])
    erad = AR*t**4
    g = np.array([gravity(x,G*args.potential_mass_msun*MSUN,
                          args.r0_fraction*args.radius_rsun*RSUN,
                          args.r1_fraction*args.radius_rsun*RSUN) for x in r])
    kappa = np.array([opacity.rosseland(d,temp) for d,temp in zip(rho,t)])
    rheat = args.heating_radius_fraction*args.radius_rsun*RSUN
    dln_t_dr=np.gradient(np.log(t),r,edge_order=2)
    diffusion_flux=-4.0*CLIGHT*AR*t**4*dln_t_dr/(3.0*kappa*rho)
    radius = args.radius_rsun*RSUN
    luminosity_flux=args.luminosity_lsun*LSUN/(4.0*np.pi*r*r)
    # The exterior is an optically thin spherical-gray construction, not a
    # diffusion solution.  Its force and HSE use conserved luminosity.
    flux=np.where(r<=radius,diffusion_flux,luminosity_flux)
    residual = np.gradient(p,r,edge_order=2) + rho*g - kappa*rho*flux/CLIGHT
    scale = np.maximum(rho*g + np.abs(kappa*rho*flux/CLIGHT),1e-300)
    return egas,erad,kappa,flux,residual/scale


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eos",type=Path,required=True)
    p.add_argument("--opacity",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--metadata",type=Path)
    p.add_argument("--potential-mass-msun",type=float,default=12.0)
    p.add_argument("--mass-msun",type=float,default=3.0)
    p.add_argument("--radius-rsun",type=float,default=830.0)
    p.add_argument("--luminosity-lsun",type=float,default=91932.0)
    p.add_argument("--effective-temperature",type=float,default=3487.0)
    p.add_argument("--r0-fraction",type=float,default=0.2)
    p.add_argument("--r1-fraction",type=float,default=1.2)
    p.add_argument("--heating-radius-fraction",type=float,default=0.2)
    p.add_argument("--box-half-rsun",type=float,default=1010.5)
    p.add_argument("--grid",type=int,default=0)
    p.add_argument("--grid-mass-tolerance",type=float,default=5e-3)
    p.add_argument("--grid-iterations",type=int,default=4)
    p.add_argument("--points",type=int,default=768)
    p.add_argument("--atmosphere-points",type=int,default=384)
    p.add_argument("--initial-logrho",type=float,default=-6.0)
    p.add_argument("--initial-central-temperature",type=float,default=1.0e5)
    p.add_argument("--density-floor",type=float,default=1.0e-21)
    p.add_argument("--photosphere-tau",type=float,default=2.0/3.0)
    p.add_argument("--superadiabatic-width",type=float,default=0.1,
                   help="Gaussian surface-layer width in Rstar")
    p.add_argument("--superadiabatic-peak-max",type=float,default=8.0,
                   help="largest delta-nabla considered by mass shooting")
    p.add_argument("--rtol",type=float,default=2e-6)
    return p.parse_args()


def main():
    args=parse_args();eos=AthenaEos(args.eos);opacity=Opacity(args.opacity)
    desired=args.mass_msun*MSUN; target=desired; mapped=None
    for _ in range(args.grid_iterations if args.grid else 1):
        interior,central,photosphere,superadiabatic_peak,entropy=shoot_interior(
            args,eos,opacity,target)
        profile=extend_atmosphere(args,eos,opacity,interior)
        if not args.grid: break
        mapped=mapped_mass(profile[0],profile[1],args.grid,args.box_half_rsun*RSUN)
        if abs(mapped/desired-1.0)<=args.grid_mass_tolerance: break
        target*=desired/mapped
    if args.grid and abs(mapped/desired-1.0)>args.grid_mass_tolerance:
        raise RuntimeError("Cartesian mapped mass did not converge")
    r,rho,p,t,m,tau=profile
    egas,erad,kappa,flux,hse=diagnostics(args,eos,opacity,profile)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savetxt(args.output,np.column_stack((r,rho,egas,erad,t,p,m,tau,kappa,flux,hse)),
      fmt="%.16e",header=("r_cm rho_g_cm3 egas_erg_cm3 erad_erg_cm3 T_K "
      "Pgas_dyn_cm2 M_spherical_g tau kappa_R flux_rad hse_relative\n"
      "material-only EOS; radiation-force HSE; spherical Eddington gray atmosphere"))
    meta={
      "model":"convective-interior+spherical-gray-atmosphere",
      "eos":str(args.eos.resolve()),"eos_sha256":hashlib.sha256(args.eos.read_bytes()).hexdigest(),
      "opacity":str(args.opacity.resolve()),"opacity_sha256":hashlib.sha256(args.opacity.read_bytes()).hexdigest(),
      "profile_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest(),
      "radiation_force_in_hse":True,"photosphere_tau":args.photosphere_tau,
      "photosphere_density_g_cm3":float(photosphere[0]),
      "photosphere_pressure_dyn_cm2":float(photosphere[1]),
      "photosphere_entropy_cgs":float(photosphere[4]),
      "central_entropy_cgs":float(entropy[0]),
      "superadiabatic_width_Rstar":args.superadiabatic_width,
      "superadiabatic_peak_delta_nabla":float(superadiabatic_peak),
      "potential_mass_msun":args.potential_mass_msun,"requested_mass_msun":args.mass_msun,
      "radius_rsun":args.radius_rsun,"luminosity_lsun":args.luminosity_lsun,
      "effective_temperature_K":args.effective_temperature,
      "central_logrho_logT":list(map(float,central)),
      "mass_inside_radius_msun":float(m[np.searchsorted(r,args.radius_rsun*RSUN)]/MSUN),
      "mapped_grid":args.grid or None,"mapped_mass_msun":None if mapped is None else float(mapped/MSUN),
      "rho_range_g_cm3":[float(rho.min()),float(rho.max())],
      "temperature_range_K":[float(t.min()),float(t.max())],
      "opacity_range_cm2_g":[float(kappa.min()),float(kappa.max())],
      "hse_relative":{
        "max_abs_resolved_interior":float(np.max(np.abs(hse[
          (r>0.01*args.radius_rsun*RSUN)&(r<0.99*args.radius_rsun*RSUN)]))),
        "rms_resolved_interior":float(np.sqrt(np.mean(hse[
          (r>0.01*args.radius_rsun*RSUN)&(r<0.99*args.radius_rsun*RSUN)]**2))),
        "max_abs_atmosphere":float(np.max(np.abs(hse[
          r>args.radius_rsun*RSUN])))}}
    meta_path=args.metadata or args.output.with_suffix(args.output.suffix+".json")
    meta_path.write_text(json.dumps(meta,indent=2)+"\n");print(json.dumps(meta,indent=2))


if __name__=="__main__":main()
