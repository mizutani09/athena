#!/usr/bin/env python3
"""Estimate radial FLD luminosity on the saved Cartesian RSG mesh.

This cell-centred shell estimate is a diagnostic, not an exact finite-volume
surface flux.  In particular a photosphere thinner than one cell is unresolved.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

C_CGS = 2.99792458e10
RSUN = 6.957e10
LSUN = 3.828e33


def grid(path, field, component):
    with h5py.File(path) as handle:
        locations = np.asarray(handle['LogicalLocations'], dtype=int)
        blocks = np.asarray(handle[field][component], dtype=float)
        nz, ny, nx = blocks.shape[1:]
        values = np.empty(((locations[:, 2].max()+1)*nz,
                           (locations[:, 1].max()+1)*ny,
                           (locations[:, 0].max()+1)*nx))
        for block, (bx, by, bz) in enumerate(locations):
            values[bz*nz:(bz+1)*nz, by*ny:(by+1)*ny,
                   bx*nx:(bx+1)*nx] = blocks[block]
        x = np.unique(np.asarray(handle['x1v']).ravel())
    return values, x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opacity', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--luminosity-lsun', type=float, default=91932.)
    parser.add_argument('--case', action='append', required=True,
                        help='label:primitive-HDF5:user-output-HDF5')
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.opacity) as handle:
        logrho = np.asarray(handle['axes/log10_density'])
        logt = np.asarray(handle['axes/log10_temperature'])
        logr = np.asarray(handle['kappa/log10_rosseland'])
    opacity = RegularGridInterpolator((logrho, logt), logr,
                                      bounds_error=False, fill_value=None)
    result = {'method': 'Cell-centred FLD flux from saved Er, T, rho and centred '
                        'Er gradients; radial dot product volume-integrated '
                        'over one-cell-thick spherical shells and divided by '
                        'shell width.  This is approximate, especially near '
                        'the unresolved photosphere.', 'cases': {}}
    if args.profile:
        profile = np.loadtxt(args.profile)
        r, density, er, kappa, stored_flux = (profile[:, col] for col in
                                               (0, 1, 3, 8, 9))
        grad = np.gradient(er, r, edge_order=2)
        sigma = kappa*density
        ratio = np.abs(grad)/np.maximum(sigma*er, 1e-300)
        limiter = (2.+ratio)/(6.+3.*ratio+ratio*ratio)
        lstar_cgs = args.luminosity_lsun*LSUN
        fld_lum = -4.*np.pi*r*r*C_CGS*limiter*grad/sigma/lstar_cgs
        prescribed = 4.*np.pi*r*r*stored_flux/lstar_cgs
        result['profile'] = {}
        for target in (500, 750, 800, 810, 815, 820, 830, 850, 900, 1000):
            index = int(np.argmin(np.abs(r/RSUN-target)))
            result['profile'][str(target)] = {
                'radius_Rsun': float(r[index]/RSUN),
                'prescribed_L_over_Lstar': float(prescribed[index]),
                'FLD_closure_L_over_Lstar': float(fld_lum[index]),
                'limiter_R': float(ratio[index])}
    for item in args.case:
        label, prim_path, uov_path = item.split(':', 2)
        prim, x = grid(prim_path, 'prim', 0)
        erad, _ = grid(uov_path, 'user_out_var', 2)
        temp, _ = grid(uov_path, 'user_out_var', 0)
        dx = float(x[1]-x[0])
        z, y, xx = np.meshgrid(x, x, x, indexing='ij')
        radius = np.sqrt(xx*xx+y*y+z*z)
        rho_cgs = prim*1e-8
        q = np.column_stack((np.log10(rho_cgs.ravel()),
                             np.log10(temp.ravel())))
        outside = np.any((q < [logrho[0], logt[0]])
                         | (q > [logrho[-1], logt[-1]]), axis=1)
        q[:, 0] = np.clip(q[:, 0], logrho[0], logrho[-1])
        q[:, 1] = np.clip(q[:, 1], logt[0], logt[-1])
        kappa = np.power(10., opacity(q)).reshape(prim.shape)
        sigma = kappa*rho_cgs*RSUN
        grad = np.gradient(erad, dx, edge_order=2)
        gradnorm = np.sqrt(sum(g*g for g in grad))
        ratio = gradnorm/np.maximum(sigma*erad, 1e-300)
        limiter = (2.+ratio)/(6.+3.*ratio+ratio*ratio)
        coeff = (C_CGS/np.sqrt(1e13))*limiter/np.maximum(sigma, 1e-300)
        radial_flux = -coeff*(grad[0]*z+grad[1]*y+grad[2]*xx)/np.maximum(radius, 1e-300)
        time_unit = RSUN/np.sqrt(1e13)
        lstar = args.luminosity_lsun*LSUN*time_unit/(1e5*RSUN**3)
        shells = {}
        for r in (500, 700, 750, 800, 830, 850, 900):
            chosen = np.abs(radius-r)<0.5*dx
            # Integral over a thin shell: integral(F_r dV)/dr ~= L(r).
            lum = np.sum(radial_flux[chosen])*dx*dx
            shells[str(r)] = {'Lrad_over_Lstar': float(lum/lstar),
                              'cells': int(chosen.sum()),
                              'median_radial_flux': float(np.median(radial_flux[chosen]))}
        result['cases'][label] = {'dx_Rsun': dx, 'opacity_table_clamped_cells':
                                  int(outside.sum()), 'shells': shells}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2)+'\n')
    print(args.json)


if __name__ == '__main__':
    main()
