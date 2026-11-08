#!/usr/bin/env python3
from pathlib import Path
import tarfile

base = Path('/work/mizutaniks/output/fld/collapse/wor92_xd_20261107')
jobs = [337947, 337948, 337949, 337950, 337956, 337957, 337958, 337959,
        337960, 337961, 337962, 337963, 337965, 337968, 337972, 337973, 337974, 337976]
target = base / 'scripts/wor92_firstcore_pdebug_evidence_20261109.tar.gz'
with tarfile.open(target, 'w:gz') as archive:
    for job in jobs:
        case = base / 'debug' / ('job_%d' % job)
        for name in ('provenance.txt', 'summary.txt', 'launcher.log', 'rank-0.log',
                     'rank-10.log'):
            path = case / name
            if path.is_file():
                archive.add(str(path), arcname=str(path.relative_to(base)))
        for path in (case / 'output').glob('*.hst'):
            archive.add(str(path), arcname=str(path.relative_to(base)))
        path = base / 'debug' / ('slurm-%d.out' % job)
        if path.is_file():
            archive.add(str(path), arcname=str(path.relative_to(base)))
print(target)
