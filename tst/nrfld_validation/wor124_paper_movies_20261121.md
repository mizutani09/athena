# WOR-124 collapse movies and magnetic-field visualization (2026-11-21)

## Source and sampling

The visualization joins the existing lower-AMR 32.25 microG trajectory to
the paper AMR+1 continuation.  It does not interpolate states in time and did
not require a new simulation.

- lower-AMR source: job 341015, cycle 60--1780, sampled every 40 cycles;
- AMR+1 source: job 344378, cycle 1800--2400, sampled every 20 cycles;
- repeated segment-boundary outputs at the same cycle are deduplicated by
  cycle, retaining the later filename;
- 75 unique HDF snapshots, 6 frames/s, 12.5 s per movie;
- final HDF SHA-256:
  `943f4714a639b95f30bbd597a09dd95dca1cb0c9d9f7fcb73ebbf269847af59a`.

Every panel uses the fixed initial core origin.  This avoids artificial camera
jumps while the initially almost uniform sphere has many equally dense cells.
Once `rho_max >= 1e-14 g cm^-3`, the discrete density maximum stays within
1.269 au of the origin; above `1e-12 g cm^-3` it stays within 0.317 au, and the
final offset is 0.159 au.

## Products

`paper_collapse_movie.py` samples the AMR leaf cells piecewise-constantly onto
a 512 by 512 display raster.  The density/temperature movie contains the full
7000-au view, central 100-au density, and central 100-au gas temperature.  The
magnetic movie contains xy and xz density slices with streamlines colored by
magnetic-field magnitude.

- `collapse_density_temperature.mp4`: H.264/yuv420p, 1848x596, 75 frames,
  12.5 s, SHA-256
  `06244d1f74f9b854643148aabf6a55e37b22c6585682f445140d8161a16b8870`;
- `collapse_core_magnetic.mp4`: H.264/yuv420p, 1860x810, 75 frames, 12.5 s,
  SHA-256
  `ffed2e8a2fa14375d77686e900b8f165a1e73658c9278e9f461b4274183736c6`;
- final magnetic views at +/-50 au and +/-300 au are available as both PNG
  and SVG;
- `movie_manifest.json` records each source path, cycle, time, AMR index,
  MeshBlock count, peak density and temperature, and density-peak coordinate.

The browsable report and all final products are under
`/home/kosuke/simulation/test/collapse_rotmhd_20261121/`
`paper_amr12_v2_job344378/movies/`.  The selected local HDF cache occupies
about 15 GB and is intentionally separate from the Git worktree.

## Interpretation and limits

The overview movie shows the global sphere developing a compact central
condensation and the central gas heating into the approximately 300 K regime.
The magnetic movie shows increasing winding in the xy plane and an
hourglass-like pinch plus small-scale distortion in the xz plane.  The
envelope-scale final view connects that central pinch to the predominantly
vertical outer field.

These streamlines are qualitative.  They are integrated through a uniformly
sampled display field made from cell-centred B, not through the native
face-centred constrained-transport field.  They therefore must not be used to
measure magnetic flux, divB, reconnection, or exact field-line topology.  The
quantitative magnetic-energy, plasma-beta, conservation, and divB statements
remain those from the original HDF/history audits.  The visible step/ring
features at some radii can include the piecewise-constant AMR display sampling
and should not be interpreted from the movie alone.

The final products passed a 75-frame monotonic-cycle manifest check, HDF final
hash check, H.264 frame-count/duration check, and visual inspection of the
first, middle, and final density and magnetic frames.
