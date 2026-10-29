#!/usr/bin/env python3
"""Resolve one tabulated-EOS matter/radiation exchange equation in double precision."""

import argparse
import math

import numpy as np


def read_ascii_table(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                rows.append(stripped)
    nvar, nx2, nx1 = map(int, rows[0].split())
    x2min, x2max = map(float, rows[1].split())
    x1min, x1max = map(float, rows[2].split())
    ratios = np.fromstring(rows[3], sep=" ")
    values = np.fromstring(" ".join(rows[4:]), sep=" ")
    expected = nvar*nx2*nx1
    if ratios.size != nvar or values.size != expected:
        raise ValueError("invalid EOS table shape")
    return (values.reshape(nvar, nx2, nx1), ratios,
            x2min, x2max, x1min, x1max)


class TableTemperature:
    def __init__(self, path, rho_unit, energy_unit, temperature_unit):
        (self.data, self.ratios, self.x2min, self.x2max,
         self.x1min, self.x1max) = read_ascii_table(path)
        self.rho_unit = rho_unit
        self.energy_unit = energy_unit
        self.temperature_unit = temperature_unit
        self.nx2 = self.data.shape[1]
        self.nx1 = self.data.shape[2]
        self.x2norm = (self.nx2 - 1)/(self.x2max - self.x2min)
        self.x1norm = (self.nx1 - 1)/(self.x1max - self.x1min)

    def coordinates(self, rho, egas):
        x1 = math.log10(rho*self.rho_unit)
        x2 = math.log10(egas*self.ratios[3]*self.energy_unit) - x1
        return x2, x1

    def log_temperature_and_slope(self, rho, egas):
        x2, x1 = self.coordinates(rho, egas)
        x = (x2 - self.x2min)*self.x2norm
        y = (x1 - self.x1min)*self.x1norm
        ix = min(max(int(x), 0), self.nx2 - 2)
        iy = min(max(int(y), 0), self.nx1 - 2)
        wx0 = 1.0 + ix - x
        wy0 = 1.0 + iy - y
        field = self.data[3]
        value = (wx0*wy0*field[ix, iy]
                 + wx0*(1.0-wy0)*field[ix, iy+1]
                 + (1.0-wx0)*wy0*field[ix+1, iy]
                 + (1.0-wx0)*(1.0-wy0)*field[ix+1, iy+1])
        lower = field[ix+1, iy] - field[ix, iy]
        upper = field[ix+1, iy+1] - field[ix, iy+1]
        slope = self.x2norm*(wy0*lower + (1.0-wy0)*upper)
        return value, slope, ix

    def temperature_and_slope(self, rho, egas):
        log_temperature, logarithmic_slope, ix = self.log_temperature_and_slope(
            rho, egas)
        temperature = 10.0**log_temperature/self.temperature_unit
        return temperature, logarithmic_slope, ix


def solve(table, rho, eold, erad, coupling, radiation_constant):
    def evaluate(egas):
        temperature, slope, cell = table.temperature_and_slope(rho, egas)
        equilibrium = radiation_constant*temperature**4
        residual = egas - eold + coupling*(equilibrium-erad)
        derivative = 1.0 + coupling*4.0*equilibrium*slope/egas
        return residual, derivative, temperature, equilibrium, cell

    lower = max(np.finfo(float).tiny, eold*0.5)
    upper = eold*2.0
    fl = evaluate(lower)[0]
    fu = evaluate(upper)[0]
    for _ in range(64):
        if fl <= 0.0 <= fu:
            break
        if fl > 0.0:
            lower *= 0.5
            fl = evaluate(lower)[0]
        if fu < 0.0:
            upper *= 2.0
            fu = evaluate(upper)[0]
    else:
        raise RuntimeError("could not bracket the exchange root")

    for _ in range(256):
        middle = lower + 0.5*(upper-lower)
        if middle == lower or middle == upper:
            break
        fm = evaluate(middle)[0]
        if fm <= 0.0:
            lower = middle
        else:
            upper = middle

    candidates = {lower, upper}
    for seed in (lower, upper):
        down = up = seed
        for _ in range(256):
            down = np.nextafter(down, -math.inf)
            up = np.nextafter(up, math.inf)
            candidates.add(float(down))
            candidates.add(float(up))
    best = min(candidates, key=lambda value: abs(evaluate(value)[0]))
    result = evaluate(best)
    below = float(np.nextafter(best, -math.inf))
    above = float(np.nextafter(best, math.inf))
    scale = max(abs(best), abs(eold), abs(coupling*(result[3]-erad)))
    return (best, result, abs(result[0])/scale, upper-lower,
            evaluate(below)[0], evaluate(above)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--eold", type=float, required=True)
    parser.add_argument("--erad", type=float, required=True)
    parser.add_argument("--coupling", type=float, required=True,
                        help="dt*c*sigma_P in code units")
    parser.add_argument("--radiation-constant", type=float, required=True)
    parser.add_argument("--rho-unit", type=float, default=1.0e-8)
    parser.add_argument("--energy-unit", type=float, default=1.0e5)
    parser.add_argument("--temperature-unit", type=float, default=1.0)
    args = parser.parse_args()
    table = TableTemperature(args.table, args.rho_unit, args.energy_unit,
                             args.temperature_unit)
    root, values, normalized, width, residual_below, residual_above = solve(
        table, args.rho, args.eold, args.erad, args.coupling,
        args.radiation_constant)
    residual, derivative, temperature, equilibrium, cell = values
    print("egas={:.17e}".format(root))
    print("residual={:.17e}".format(residual))
    print("normalized_residual={:.17e}".format(normalized))
    print("bracket_width={:.17e}".format(width))
    print("residual_nextafter_below={:.17e}".format(residual_below))
    print("residual_nextafter_above={:.17e}".format(residual_above))
    print("temperature={:.17e}".format(temperature))
    print("equilibrium_erad={:.17e}".format(equilibrium))
    print("derivative={:.17e}".format(derivative))
    print("x2_cell={}".format(cell))


if __name__ == "__main__":
    main()
