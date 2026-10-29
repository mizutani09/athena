#!/usr/bin/env python3
"""Convert a complete NATA rho-u table to Athena++'s six-field EOS format."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--logrho-min", type=float, default=-18.0)
    p.add_argument("--logrho-max", type=float, default=-3.0)
    p.add_argument("--logx2-min", type=float, default=8.0)
    p.add_argument("--logx2-max", type=float, default=14.5)
    p.add_argument("--nx2", type=int, default=326)
    return p.parse_args()


def main():
    args = parse_args()
    with args.input.open() as stream:
        header = [next(stream) for _ in range(4)]
    nrho_all = int(header[1].split()[0])
    nu_all = int(header[2].split()[0])
    raw = np.loadtxt(args.input, skiprows=4)
    if raw.ndim != 2 or raw.shape[0] != nrho_all*nu_all or raw.shape[1] < 8:
        raise ValueError("NATA table is incomplete or has an unsupported schema")
    table = raw.reshape(nrho_all, nu_all, raw.shape[1])
    logrho_all = table[:, 0, 0]
    if not np.allclose(table[:, :, 0], logrho_all[:, None], atol=5e-4):
        raise ValueError("unexpected NATA density ordering")
    if not np.allclose(table[:, :, 1], table[0, :, 1][None, :], atol=5e-4):
        raise ValueError("unexpected NATA energy ordering")

    eps = 5e-4
    selected = np.flatnonzero((logrho_all >= args.logrho_min-eps)
                              & (logrho_all <= args.logrho_max+eps))
    if selected.size < 2:
        raise ValueError("requested density interval contains fewer than two rows")
    table = table[selected]
    logrho = logrho_all[selected]
    if np.max(np.abs(np.diff(logrho)-np.diff(logrho).mean())) > 2e-4:
        raise ValueError("selected density axis is not uniform")
    # Restore exact endpoints lost to the NATA writer's four-decimal output.
    logrho = np.linspace(logrho[0], logrho[-1], logrho.size)
    logx2 = np.linspace(args.logx2_min, args.logx2_max, args.nx2)
    out = np.empty((6, args.nx2, logrho.size))
    gamma_col = 12 if raw.shape[1] >= 13 else 7

    for jr, lrho in enumerate(logrho):
        col = table[jr]
        valid = ((col[:, 6] == 0) & np.all(np.isfinite(
            col[:, [1, 2, 4, 5, gamma_col]]), axis=1)
                 & (col[:, 5] > 0.0) & (col[:, gamma_col] > 0.0)
                 & (col[:, 2] > -900.0) & (col[:, 4] > -900.0))
        if valid.sum() < 2:
            raise ValueError(f"insufficient valid states at log10(rho)={lrho}")
        lu, lt, lp, entropy, gamma1 = (
            col[valid, 1], col[valid, 2], col[valid, 4],
            col[valid, 5], col[valid, gamma_col])
        if logx2[0] < lu[0]-eps or logx2[-1] > lu[-1]+eps:
            raise ValueError(f"specific-energy domain is too narrow at {lrho}")
        out[0, :, jr] = np.interp(logx2, lu, lp)-lrho-logx2
        out[3, :, jr] = np.interp(logx2, lu, lt)
        out[4, :, jr] = np.interp(logx2, lu, np.log10(entropy))

        logq = lp-lrho
        order = np.argsort(logq)
        logq, luq = logq[order], lu[order]
        sq, gq = entropy[order], gamma1[order]
        keep = np.r_[True, np.diff(logq) > 1e-12]
        logq, luq, sq, gq = logq[keep], luq[keep], sq[keep], gq[keep]
        if logx2[0] < logq[0]-eps or logx2[-1] > logq[-1]+eps:
            raise ValueError(f"P/rho domain is too narrow at {lrho}")
        out[1, :, jr] = np.interp(logx2, logq, luq-logq)
        out[2, :, jr] = np.log10(np.interp(logx2, logq, gq))
        out[5, :, jr] = np.interp(logx2, logq, np.log10(sq))

    if not np.isfinite(out).all():
        raise ValueError("converted EOS contains a non-finite value")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comments = [
        "Log p/e_int(e_spec,rho)", "Log e_int/p(p/rho,rho)",
        "Log asq*rho/p(p/rho,rho) = Log gamma1", "Log T(e_spec,rho)",
        "Log S_cgs(e_spec,rho)", "Log S_cgs(p/rho,rho)"]
    with args.output.open("w") as stream:
        stream.write("# Entries must be space separated.\n# n_var, n_x2, n_x1\n")
        stream.write(f"6 {args.nx2} {logrho.size}\n# Log x2 limits\n")
        stream.write(f"{logx2[0]:.16e} {logx2[-1]:.16e}\n# Log rho limits\n")
        stream.write(f"{logrho[0]:.16e} {logrho[-1]:.16e}\n# Ratios\n")
        stream.write("1.0 1.0 1.0 1.0 1.0 1.0\n")
        for comment, values in zip(comments, out):
            stream.write(f"# {comment}\n")
            np.savetxt(stream, values, fmt="%.10e")
    metadata = {
        "source": str(args.input.resolve()),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "nvar": 6, "nx2": args.nx2, "nrho": int(logrho.size),
        "logx2": [float(logx2[0]), float(logx2[-1])],
        "logrho": [float(logrho[0]), float(logrho[-1])],
    }
    meta = args.metadata or args.output.with_suffix(args.output.suffix+".json")
    meta.write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
