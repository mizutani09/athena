#!/usr/bin/env python3
"""Trial: join a gray RSG radiation profile to the FLD streaming field.

Gas state and the profile's prescribed force flux are retained.  The resulting
optically thin gas/radiation disequilibrium is intentional and must be tested.
"""

import argparse
import json
from pathlib import Path

import numpy as np

RSUN = 6.957e10
LSUN = 3.828e33
C = 2.99792458e10


def fld_luminosity(profile, luminosity):
    r, rho, er, kappa = (profile[:, col] for col in (0, 1, 3, 8))
    grad = np.gradient(er, r, edge_order=2)
    sigma = rho*kappa
    ratio = np.abs(grad)/np.maximum(sigma*er, 1e-300)
    limiter = (2.+ratio)/(6.+3.*ratio+ratio*ratio)
    return -4.*np.pi*r*r*C*limiter*grad/sigma/luminosity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--luminosity-lsun', type=float, default=91932.)
    parser.add_argument('--start-rsun', type=float, default=820.)
    parser.add_argument('--end-rsun', type=float, default=850.)
    args = parser.parse_args()
    if not 0 < args.start_rsun < args.end_rsun:
        parser.error('require 0 < start-rsun < end-rsun')
    profile = np.loadtxt(args.input)
    luminosity = args.luminosity_lsun*LSUN
    r = profile[:, 0]
    er_streaming = luminosity/(4.*np.pi*r*r*C)
    t = np.clip((r/RSUN-args.start_rsun)
                /(args.end_rsun-args.start_rsun), 0., 1.)
    blend = t*t*(3.-2.*t)
    corrected = profile.copy()
    corrected[:, 3] = (1.-blend)*profile[:, 3]+blend*er_streaming
    original_lum = fld_luminosity(profile, luminosity)
    corrected_lum = fld_luminosity(corrected, luminosity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, corrected, fmt='%.16e',
               header='r_cm rho_g_cm3 egas_erg_cm3 erad_erg_cm3 T_K '
                      'Pgas_dyn_cm2 M_spherical_g tau kappa_R flux_rad '
                      'hse_relative\nTrial FLD streaming exterior; gas state and '
                      'prescribed force flux retained')
    sample = {}
    for radius in (800, 820, 830, 840, 850, 900, 1000):
        index = int(np.argmin(np.abs(r/RSUN-radius)))
        sample[str(radius)] = {
            'radius_Rsun': float(r[index]/RSUN),
            'old_FLD_L_over_Lstar': float(original_lum[index]),
            'trial_FLD_L_over_Lstar': float(corrected_lum[index]),
            'radiation_to_gas_temperature': float(
                (corrected[index, 3]/7.5657e-15)**.25/corrected[index, 4])}
    mask = (r/RSUN>=args.start_rsun)&(r/RSUN<=1000)
    metadata = {
        'input': str(args.input), 'output': str(args.output),
        'start_Rsun': args.start_rsun, 'end_Rsun': args.end_rsun,
        'unchanged_columns': [0, 1, 2, 4, 5, 6, 7, 8, 9, 10],
        'outer_820_1000_max_L_over_Lstar': float(np.max(corrected_lum[mask])),
        'outer_820_1000_min_L_over_Lstar': float(np.min(corrected_lum[mask])),
        'samples': sample}
    path = args.output.with_suffix(args.output.suffix+'.json')
    path.write_text(json.dumps(metadata, indent=2)+'\n')
    print(path)


if __name__ == '__main__':
    main()
