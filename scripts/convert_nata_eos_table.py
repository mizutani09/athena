#!/usr/bin/env python3
"""Convert a NATA rho-u table to the Athena++ general-EOS table format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


COMMENTS = (
    "Log p/e_int(e_spec,rho)",
    "Log e_int/p(p/rho,rho)",
    "Log asq*rho/p(p/rho,rho) = Log gamma1",
    "Log T(e_spec,rho)",
    "Log S_cgs(e_spec,rho)",
    "Log S_cgs(p/rho,rho)",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--logrho-min", type=float, default=-10.5)
    parser.add_argument("--logrho-max", type=float, default=-4.5)
    parser.add_argument("--logx2-min", type=float, default=11.2)
    parser.add_argument("--logx2-max", type=float, default=14.5)
    parser.add_argument("--dlogx2", type=float, default=0.02)
    return parser.parse_args()


def load_nata(path):
    with path.open() as stream:
        header = [next(stream) for _ in range(4)]
    nrho = int(header[1].split()[0])
    nu = int(header[2].split()[0])
    raw = np.loadtxt(path, skiprows=4)
    if raw.shape[0] != nrho * nu:
        raise RuntimeError(f"Unexpected row count: {raw.shape[0]} != {nrho}*{nu}")
    table = raw.reshape(nrho, nu, raw.shape[1])
    if not np.allclose(table[:, :, 0], table[:, 0, 0, None]):
        raise RuntimeError("Unexpected logRho ordering")
    if not np.allclose(table[:, :, 1], table[0, :, 1, None].T):
        raise RuntimeError("Unexpected logU ordering")
    return table


def convert(args):
    table = load_nata(args.input)
    logrho_all = table[:, 0, 0]
    tolerance = 5.0e-4
    selected = np.where(
        (logrho_all >= args.logrho_min - tolerance)
        & (logrho_all <= args.logrho_max + tolerance)
    )[0]
    if selected.size < 2:
        raise RuntimeError("Fewer than two density columns selected")
    table = table[selected]
    logrho = np.linspace(args.logrho_min, args.logrho_max, selected.size)

    intervals = round((args.logx2_max - args.logx2_min) / args.dlogx2)
    if intervals < 1:
        raise ValueError("The requested logX2 grid needs at least two points")
    logx2 = np.linspace(args.logx2_min, args.logx2_max, intervals + 1)
    if not np.isclose(logx2[1] - logx2[0], args.dlogx2):
        raise ValueError("logX2 range is not divisible by dlogx2")

    ierr_column = 6
    entropy_column = 5
    gamma1_column = 12 if table.shape[2] >= 13 else 7
    output = np.empty((6, logx2.size, logrho.size))

    for density_index, density in enumerate(logrho):
        column = table[density_index]
        valid = (
            (column[:, ierr_column] == 0)
            & np.isfinite(column[:, 1])
            & np.isfinite(column[:, 2])
            & np.isfinite(column[:, 4])
            & np.isfinite(column[:, entropy_column])
            & np.isfinite(column[:, gamma1_column])
            & (column[:, 2] > -900.0)
            & (column[:, 4] > -900.0)
            & (column[:, entropy_column] > 0.0)
            & (column[:, gamma1_column] > 0.0)
        )
        if np.count_nonzero(valid) < 2:
            raise RuntimeError(f"Insufficient valid data at logRho={density}")

        logu = column[valid, 1]
        logt = column[valid, 2]
        logp = column[valid, 4]
        entropy = column[valid, entropy_column]
        gamma1 = column[valid, gamma1_column]
        if logx2[0] < logu[0] or logx2[-1] > logu[-1]:
            raise RuntimeError(f"logU does not cover logRho={density}")

        logp_at_u = np.interp(logx2, logu, logp)
        output[0, :, density_index] = logp_at_u - density - logx2
        output[3, :, density_index] = np.interp(logx2, logu, logt)
        output[4, :, density_index] = np.interp(
            logx2, logu, np.log10(entropy)
        )

        logq = logp - density
        order = np.argsort(logq)
        logq = logq[order]
        if np.any(np.diff(logq) <= 0.0):
            raise RuntimeError(f"P/rho is not monotonic at logRho={density}")
        if logx2[0] < logq[0] or logx2[-1] > logq[-1]:
            raise RuntimeError(f"P/rho does not cover logRho={density}")
        output[1, :, density_index] = np.interp(
            logx2, logq, (logu - (logp - density))[order]
        )
        gamma_at_q = np.interp(logx2, logq, gamma1[order])
        output[2, :, density_index] = np.log10(gamma_at_q)
        output[5, :, density_index] = np.interp(
            logx2, logq, np.log10(entropy[order])
        )

    if not np.all(np.isfinite(output)):
        raise RuntimeError("Converted table contains NaN or Inf")
    return logrho, logx2, output


def write_table(path, logrho, logx2, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        stream.write("# Entries must be space separated.\n")
        stream.write("# n_var, n_x2, n_x1\n")
        stream.write(f"6 {logx2.size} {logrho.size}\n")
        stream.write("# Log x2 limits\n")
        stream.write(f"{logx2[0]:.16e} {logx2[-1]:.16e}\n")
        stream.write("# Log rho limits\n")
        stream.write(f"{logrho[0]:.16e} {logrho[-1]:.16e}\n")
        stream.write("# Ratios\n")
        stream.write("1.0 1.0 1.0 1.0 1.0 1.0\n")
        for comment, field in zip(COMMENTS, data):
            stream.write(f"# {comment}\n")
            for row in field:
                stream.write(" ".join(f"{value:.10e}" for value in row) + "\n")


def main():
    args = parse_args()
    logrho, logx2, data = convert(args)
    write_table(args.output, logrho, logx2, data)
    print(f"Wrote {args.output}")
    print(f"shape: 6 x {logx2.size} x {logrho.size}")
    print(f"logRho: {logrho[0]} .. {logrho[-1]}")
    print(f"logX2: {logx2[0]} .. {logx2[-1]}")


if __name__ == "__main__":
    main()
