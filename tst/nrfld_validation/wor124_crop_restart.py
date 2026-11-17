#!/usr/bin/env python3
"""Crop an Athena++ restart to an aligned, fully refined central cube.

This preserves each selected MeshBlock's raw restart body, including CT face
fields and FLD state.  It changes only the parameter header, mesh metadata,
and logical locations.  It is an experimental boundary-condition diagnostic:
the periodic boundaries inherited from the parent are not physical for a
long-lived central zoom.
"""

import argparse
import ctypes
import json
from pathlib import Path
import re
import struct


class RegionSize(ctypes.Structure):
    _fields_ = [(name, ctypes.c_double) for name in
                ("x1min", "x2min", "x3min", "x1max", "x2max", "x3max",
                 "x1rat", "x2rat", "x3rat")]
    _fields_ += [(name, ctypes.c_int) for name in ("nx1", "nx2", "nx3")]


class LogicalLocation(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("lx1", "lx2", "lx3")]
    _fields_ += [("level", ctypes.c_int)]


def replace_mesh_parameter(prefix, key, value):
    text = prefix.decode("ascii")
    match = re.search(r"(?ms)^<mesh>\s*\n(.*?)(?=^<|\Z)", text)
    if match is None:
        raise ValueError("restart has no <mesh> block")
    block = match.group(1)
    pattern = rf"(?m)^({re.escape(key)}\s*=\s*).*$"
    block, count = re.subn(pattern, lambda m: m.group(1) + str(value), block)
    if count != 1:
        raise ValueError(f"expected one mesh/{key}, found {count}")
    text = text[:match.start(1)] + block + text[match.end(1):]
    return text.encode("ascii")


def require_complete_tree(leaves):
    # A complete octree proves every cell of the new root grid has one source.
    nodes = {(level, x, y, z) for level, x, y, z in leaves}
    visited = set()

    def visit(node):
        if node in nodes:
            visited.add(node)
            return
        level, x, y, z = node
        if level > 63:
            raise ValueError("incomplete or invalid crop tree")
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    visit((level + 1, 2*x + dx, 2*y + dy, 2*z + dz))

    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                visit((1, x, y, z))
    if visited != nodes:
        raise ValueError("crop contains overlapping or detached blocks")


def crop(source, target, half_width, dry_run=False):
    with source.open("rb") as stream:
        prefix_sample = stream.read(65536)
        marker = prefix_sample.find(b"<par_end>")
        if marker < 0:
            raise ValueError("restart parameter terminator not found")
        newline = prefix_sample.find(b"\n", marker)
        if newline < 0:
            raise ValueError("unterminated restart parameters")
        param_end = newline + 1
        stream.seek(param_end)
        old_count, old_root = struct.unpack("=ii", stream.read(8))
        region = RegionSize.from_buffer_copy(stream.read(ctypes.sizeof(RegionSize)))
        time, dt, cycle = struct.unpack("=ddi", stream.read(20))
        data_size = struct.unpack("=Q", stream.read(8))[0]
        header_end = stream.tell()
        loc_size = ctypes.sizeof(LogicalLocation)
        id_size = loc_size + 8
        user_bytes = source.stat().st_size - header_end - old_count*(id_size + data_size)
        if user_bytes < 0:
            raise ValueError("restart size does not match block layout")
        user_data = stream.read(user_bytes)
        entries = []
        for index in range(old_count):
            loc = LogicalLocation.from_buffer_copy(stream.read(loc_size))
            cost = stream.read(8)
            entries.append((index, loc, cost))
        data_start = stream.tell()
        if data_start + old_count*data_size != source.stat().st_size:
            raise ValueError("restart body offset mismatch")

        if any(getattr(region, f"x{axis}rat") != 1.0 for axis in (1, 2, 3)):
            raise ValueError("only uniform parent root grids are supported")
        if not (region.nx1 == region.nx2 == region.nx3):
            raise ValueError("only cubic parent root grids are supported")
        mesh_match = re.search(rb"(?ms)^<meshblock>\s*\n(.*?)(?=^<|\Z)",
                               prefix_sample[:param_end])
        if mesh_match is None:
            raise ValueError("missing meshblock parameters")
        block = mesh_match.group(1)
        dims = []
        for axis in (1, 2, 3):
            m = re.search(rb"(?m)^nx" + str(axis).encode() + rb"\s*=\s*(\d+)", block)
            if m is None:
                raise ValueError("missing meshblock dimension")
            dims.append(int(m.group(1)))
        if len(set(dims)) != 1:
            raise ValueError("only cubic MeshBlocks are supported")
        nroot = region.nx1 // dims[0]
        if nroot != 1 << old_root:
            raise ValueError("expected power-of-two root grid")
        full_width = region.x1max - region.x1min
        if not all(abs(getattr(region, f"x{axis}min") + full_width/2) < 1e-9
                   and abs(getattr(region, f"x{axis}max") - full_width/2) < 1e-9
                   for axis in (1, 2, 3)):
            raise ValueError("only centered cubic parent grids are supported")
        ratio = full_width / (2*half_width)
        if ratio < 2 or abs(ratio - round(ratio)) > 1e-10:
            raise ValueError("crop must align with parent block hierarchy")
        ratio = round(ratio)
        if ratio & (ratio - 1):
            raise ValueError("crop ratio must be a power of two")
        # The new domain has two root blocks per dimension, and must coincide
        # with two parent blocks at absolute level crop_abs.
        crop_abs = old_root + ratio.bit_length() - 2
        parent_blocks_at_crop = nroot * (1 << (crop_abs - old_root))
        start = parent_blocks_at_crop//2 - 1
        level_shift = crop_abs - 1
        kept = []
        new_locs = []
        for index, loc, cost in entries:
            if loc.level < crop_abs:
                continue
            factor = 1 << (loc.level - crop_abs)
            xyz = (loc.lx1, loc.lx2, loc.lx3)
            if not all(start*factor <= q < (start+2)*factor for q in xyz):
                continue
            new = (loc.level-level_shift,
                   *(q - start*factor for q in xyz))
            kept.append((index, cost))
            new_locs.append(new)
        require_complete_tree(new_locs)

        prefix = prefix_sample[:param_end]
        for axis in (1, 2, 3):
            prefix = replace_mesh_parameter(prefix, f"nx{axis}", 2*dims[0])
            prefix = replace_mesh_parameter(prefix, f"x{axis}min", -half_width)
            prefix = replace_mesh_parameter(prefix, f"x{axis}max", half_width)
        numlevel_match = re.search(rb"(?m)^numlevel\s*=\s*(\d+)", prefix)
        if numlevel_match:
            old_numlevel = int(numlevel_match.group(1))
            prefix = replace_mesh_parameter(prefix, "numlevel",
                                            old_numlevel - (crop_abs-old_root))
        report = dict(source=str(source), target=str(target), source_cycle=cycle,
                      source_time=time, source_dt=dt, source_blocks=old_count,
                      crop_blocks=len(kept), half_width_au=half_width,
                      source_root_level=old_root, crop_root_level=1,
                      source_data_bytes=data_size, user_mesh_data_bytes=user_bytes,
                      source_bytes=source.stat().st_size,
                      boundary_warning="Periodic crop is for short A/B diagnostics only",
                      target_bytes=len(prefix)+8+ctypes.sizeof(RegionSize)+20+8
                          +user_bytes+len(kept)*(id_size+data_size))
        if dry_run:
            return report
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        region.x1min = region.x2min = region.x3min = -half_width
        region.x1max = region.x2max = region.x3max = half_width
        region.nx1 = region.nx2 = region.nx3 = 2*dims[0]
        with target.open("xb") as out:
            out.write(prefix)
            out.write(struct.pack("=ii", len(kept), 1))
            out.write(bytes(region))
            out.write(struct.pack("=ddiQ", time, dt, cycle, data_size))
            out.write(user_data)
            for (index, cost), (level, x, y, z) in zip(kept, new_locs):
                out.write(bytes(LogicalLocation(x, y, z, level)))
                out.write(cost)
            for index, _ in kept:
                stream.seek(data_start + index*data_size)
                chunk = stream.read(data_size)
                if len(chunk) != data_size:
                    raise ValueError("truncated MeshBlock body")
                out.write(chunk)
        if target.stat().st_size != report["target_bytes"]:
            raise ValueError("crop size mismatch")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--half-width-au", type=float, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(crop(args.source, args.target, args.half_width_au,
                          args.dry_run), indent=2))


if __name__ == "__main__":
    main()
