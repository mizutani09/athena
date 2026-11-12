#!/usr/bin/env python3
"""Audit first-step RSG box-boundary radiation loss against the energy budget."""

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

C_CGS = 2.99792458e10


def history(path):
    lines = path.read_text().splitlines()
    names = re.findall(r'\[\d+\]=([^ ]+)', lines[1])
    rows = [[float(value) for value in line.split()]
            for line in lines[2:] if len(line.split()) == len(names)]
    if len(rows) < 2:
        raise ValueError(f'Need initial and final history rows in {path}')
    return names, np.asarray(rows)


def boundary_energy(path, alpha, velocity_unit):
    with h5py.File(path) as f:
        locations = np.asarray(f['LogicalLocations'], dtype=int)
        blocks = np.asarray(f['user_out_var'][2], dtype=float)
        nz, ny, nx = blocks.shape[1:]
        energy = np.empty(((locations[:, 2].max()+1)*nz,
                           (locations[:, 1].max()+1)*ny,
                           (locations[:, 0].max()+1)*nx))
        for block, (bx, by, bz) in enumerate(locations):
            energy[bz*nz:(bz+1)*nz, by*ny:(by+1)*ny,
                   bx*nx:(bx+1)*nx] = blocks[block]
        width = float(f['x1f'][0, 1]-f['x1f'][0, 0])
        box_face = abs(float(np.min(f['x1f'][:, 0])))
        centers = np.linspace(-box_face+0.5*width, box_face-0.5*width,
                              energy.shape[0])
    faces = (energy[0], energy[-1], energy[:, 0], energy[:, -1],
             energy[:, :, 0], energy[:, :, -1])
    y, z = np.meshgrid(centers, centers, indexing='ij')
    radial_cosine = box_face/np.sqrt(box_face**2+y*y+z*z)
    c_code = C_CGS/velocity_unit
    return (float(alpha*c_code*width**2*sum(face.sum() for face in faces)),
            float(alpha*c_code*width**2
                  *sum((face*radial_cosine).sum() for face in faces)))


def surface_temperature(directory, first=False):
    files = sorted(directory.glob('*.prim.*.athdf'))
    primitive = files[0] if first else files[-1]
    output = Path(str(primitive).replace('.prim.', '.uov.'))
    with h5py.File(primitive) as f, h5py.File(output) as u:
        rho = np.asarray(f['prim'][0], dtype=float)
        temp = np.asarray(u['user_out_var'][0], dtype=float)
        x = np.asarray(f['x1v'])[:, None, None, :]
        y = np.asarray(f['x2v'])[:, None, :, None]
        z = np.asarray(f['x3v'])[:, :, None, None]
        radius = np.sqrt(x*x+y*y+z*z)
        shell = (radius >= 800) & (radius < 830)
        return float(np.average(temp[shell], weights=rho[shell]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True,
                        help='label:run-directory:Marshak-alpha (use 0 for cut_diff)')
    parser.add_argument('--velocity-unit', type=float, default=np.sqrt(1e13))
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    result = {'note': 'Boundary luminosity inferred from gas+Er energy change '
                       'and prescribed heating; gravity-work and printed history '
                       'precision are small limits for this first step.',
              'cases': {}}
    for item in args.case:
        label, dirname, alpha_string = item.rsplit(':', 2)
        directory = Path(dirname)
        alpha = float(alpha_string)
        names, h = history(next(directory.glob('*.hst')))
        data = {name: h[:, index] for index, name in enumerate(names)}
        dt = float(data['time'][-1]-data['time'][0])
        lstar = float(data['Lheat'][-1])
        delta_total = float((data['tot-E'][-1]+data['Erad'][-1])
                            -(data['tot-E'][0]+data['Erad'][0]))
        output = sorted(directory.glob('*.uov.*.athdf'))
        face_initial, projected_initial = boundary_energy(
            output[0], alpha, args.velocity_unit)
        face_final, projected_final = boundary_energy(
            output[-1], alpha, args.velocity_unit)
        initial_temperature = surface_temperature(directory, first=True)
        final_temperature = surface_temperature(directory)
        result['cases'][label] = {
            'dt_code': dt, 'alpha': alpha,
            'delta_gas_plus_radiation_code': delta_total,
            'inferred_boundary_luminosity_over_Lstar':
                (lstar*dt-delta_total)/(dt*lstar),
            'alpha_c_E_face_initial_over_Lstar': face_initial/lstar,
            'alpha_c_E_face_final_over_Lstar': face_final/lstar,
            'radial_projected_face_initial_over_Lstar': projected_initial/lstar,
            'radial_projected_face_final_over_Lstar': projected_final/lstar,
            'surface_800_830_temperature_K': final_temperature,
            'surface_800_830_initial_temperature_K': initial_temperature,
            'surface_800_830_fractional_cooling':
                1.0-final_temperature/initial_temperature}
        face_names = ('Lrad_xm', 'Lrad_xp', 'Lrad_ym', 'Lrad_yp',
                      'Lrad_zm', 'Lrad_zp')
        if all(name in data for name in face_names):
            face_luminosities = {name: float(data[name][-1]/lstar)
                                 for name in face_names}
            result['cases'][label]['measured_boundary_faces_over_Lstar'] = (
                face_luminosities)
            result['cases'][label]['measured_boundary_total_over_Lstar'] = (
                sum(face_luminosities.values()))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2)+'\n')

    labels = list(result['cases'])
    colors = ['#a84553', '#dc9849', '#618f9d', '#779266']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, ylabel, scale in (
            (axes[0], 'inferred_boundary_luminosity_over_Lstar',
             'boundary-associated luminosity / Lstar', 1.0),
            (axes[1], 'surface_800_830_fractional_cooling',
             '800–830 Rsun gas cooling [%]', 100.0)):
        ax.bar(labels, [scale*result['cases'][label][key] for label in labels],
               color=colors[:len(labels)])
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='x', rotation=20)
    axes[0].axhline(1, color='black', linestyle='--', linewidth=0.8)
    fig.suptitle('RSG first-step radiation boundary audit (0.84 day)')
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)
    print(args.figure)
    print(args.json)


if __name__ == '__main__':
    main()
