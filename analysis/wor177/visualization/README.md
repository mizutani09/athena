# WOR-177 relaxation visualization

The report summarizes the rejected L6 and accepted L7 isolated-star runs.
Selected XD equatorial slices are intentionally not committed; the generated
PNG figures, HTML, and provenance manifest are committed instead.

Regenerate after copying the source files listed in `report_manifest.json` and
the four selected `out3` snapshots into a local source root:

```sh
python3 analysis/wor177/visualization/make_relaxation_report.py \
  --source-root /path/to/local/wor177-source \
  --output analysis/wor177/visualization/report
```

Open `report/index.html` in a browser.  Each embedded figure links to its
full-resolution PNG.
