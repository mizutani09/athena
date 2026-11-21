# WOR-124 3-D collapse time evolution (2026-11-21)

## Source and construction

The 75 real HDF snapshots previously selected for the 2-D paper movies were
rendered with a fixed three-dimensional camera.  The sequence joins the
lower-AMR trajectory through cycle 1780 to the AMR+1 continuation from cycle
1800 through 2400; duplicated segment-boundary cycles were already removed by
the source manifest.  It spans 13.7168--29.4721 kyr and
`rho_max=8.4989e-18--1.00418e-10 g cm^-3`.

Each snapshot is sampled onto two 96 cubed display lattices:

- global: +/-2500 au, showing the initial sphere, global collapse, and outer
  magnetic geometry;
- core: +/-50 au, showing the late flattened dense object and field winding.

Both cameras, density levels, density color ranges, magnetic color range, and
seed geometry are fixed in time.  Every frame reports its actual simulation
time, cycle, and native peak density.  Snapshots are shown at equal playback
intervals; the movie is not interpolated and is not uniform in physical time.
This choice retains the dense late-time sampling needed to see first-core
development.

## Products and validation

The final playback rate is 3 frames/s, giving 25.0 s for 75 frames.  Three
H.264/yuv420p products were generated:

- combined global+core, 1920x720, SHA-256
  `1d704a36dc31cc91e7622dc77332051e0d6dd660ddc436358653cd345c3351b9`;
- global only, 960x720, SHA-256
  `44f6d94553689b4e6f468caf92a4b96524bbba97fb81cfe510acac3a88049202`;
- core only, 960x720, SHA-256
  `a332cee6ff3a26f4605c6bff04f296653407023f58f54a4b50769cce2fda79e7`.

The original final-state orbit was also re-encoded from its retained source
frames at a slower 24.0 s duration.  The core and envelope SHA-256 values are
`96bdc57a8d9a47f6fb91065840e8a7c631f786a8eb350be7d4645653aa376124`
and `0a68d4ba65b81c9f5d593dabb5e37349d5020f69addee9cea517892eb383f46e`.

The full final render took 2 min 15.05 s and 610,352 KiB peak RSS.  Its output
occupies about 79 MB including all 150 source PNG panels.  All five videos were
decoded end to end with ffmpeg.  Frame count, duration, dimensions, codec and
pixel format were checked with ffprobe.  The first, middle, and final rendered
states and a decoded middle frame of the combined movie were visually
inspected.  All 75 manifest cycles are strictly increasing.

The browsable output is
`/home/kosuke/simulation/test/collapse_rotmhd_20261121/`
`paper_amr12_v2_job344378/three_d_evolution_v2/index.html`.

## Interpretation limits

The time evolution makes the transition from the initially almost vertical
field and nearly uniform sphere to a centrally pinched field, flattened dense
structure, and strongly wound inner field visually explicit.  It is a
morphology product, not a new quantitative magnetic diagnostic.

AMR leaf-cell data are sampled piecewise-constantly, so facets and changes in
display texture can reflect AMR resolution.  The magnetic trajectories are
Runge--Kutta integrations through the normalized sampled cell-centred B field,
not native face-centred constrained-transport fluxes.  Magnetic flux, divB,
reconnection, and exact topology must continue to use the native HDF/history
audits.  Equal playback spacing also means apparent speed in the movie is not
physical collapse speed; the printed time is authoritative.
