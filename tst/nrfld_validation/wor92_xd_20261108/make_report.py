#!/usr/bin/env python3
"""Build a standalone WOR-92 XD status report from archived job evidence."""
from pathlib import Path
from html import escape
import re
import tarfile

ROOT = Path(__file__).resolve().parent
LONG = ROOT / 'wor92_xd_long337351_evidence.tar.gz'
RETRY = ROOT / 'wor92_xd_fgmres600_debug_evidence.tar.gz'


def member_text(archive, member):
    with tarfile.open(archive, 'r:gz') as source:
        return source.extractfile(member).read().decode('utf-8', errors='replace')


def fgmres(log):
    rows = []
    for line in log.splitlines():
        if line.startswith('[LINEAR_FGMRES]'):
            rows.append((int(re.search(r'iterations=(\d+)', line).group(1)),
                         float(re.search(r'relative_residual=([\d.eE+-]+)', line).group(1))))
    return rows


def history(text):
    by_time = {}
    for line in text.splitlines():
        if line and not line.startswith('#'):
            row = [float(value) for value in line.split()]
            by_time[row[0]] = row
    return [by_time[key] for key in sorted(by_time)]


def svg_chart(series, xvalues, title, ymin, ymax, y_ticks, x_label, y_label,
              width=820, height=300):
    left, right, top, bottom = 76, 24, 28, 52
    plot_width = width - left - right
    plot_height = height - top - bottom
    xmin, xmax = min(xvalues), max(xvalues)
    def point(x, y):
        px = left + (x - xmin) / (xmax - xmin) * plot_width
        py = top + (ymax - y) / (ymax - ymin) * plot_height
        return px, py
    pieces = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
              f'<title>{escape(title)}</title>',
              f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']
    for value in y_ticks:
        px, py = point(xmin, value)
        pieces.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#e0e8ec"/>')
        pieces.append(f'<text x="{left-10}" y="{py+5:.1f}" text-anchor="end">{value:g}</text>')
    for frac in (0, .25, .5, .75, 1):
        value = xmin + frac * (xmax - xmin)
        px, _ = point(value, ymin)
        pieces.append(f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" stroke="#eef2f4"/>')
        pieces.append(f'<text x="{px:.1f}" y="{height-bottom+22}" text-anchor="middle">{value:g}</text>')
    pieces.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle">{escape(x_label)}</text>')
    pieces.append(f'<text x="{left}" y="18">{escape(y_label)}</text>')
    for name, color, xs, ys in series:
        coords = [point(x, y) for x, y in zip(xs, ys)]
        shape = ' '.join(f'{x:.1f},{y:.1f}' for x, y in coords)
        pieces.append(f'<polyline points="{shape}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')
        for x, y in coords:
            pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}"/>')
    pieces.append('</svg>')
    return ''.join(pieces)


segment0 = member_text(LONG, 'job_337351/segment_000.log')
segment1 = member_text(LONG, 'job_337351/segment_001.log')
retry10 = member_text(RETRY, 'job_337766/launcher.log')
retry14 = member_text(RETRY, 'job_337771/launcher.log')
rows = history(member_text(LONG, 'job_337351/output/cfl030_flux_precision16_local.hst'))
old = fgmres(segment1)
new = fgmres(retry10)
next_rows = fgmres(retry14)
assert len(rows) == 14 and len(old) == 8 and len(new) == 9 and len(next_rows) == 14
assert old[-1][0] == 300 and old[-1][1] > 2e-8
assert max(value for _, value in new + next_rows) < 2e-8
assert max(count for count, _ in new) == 480

rho0, jeans0 = rows[0][15], rows[0][34]
times = [row[0] for row in rows]
trajectory = svg_chart([
    ('rho', '#0e7490', times, [(row[15]/rho0-1)*100 for row in rows]),
    ('jeans', '#bd6a25', times, [(row[34]/jeans0-1)*100 for row in rows]),
], times, 'Density and Jeans resolution through the failed run', -1.2, 2.2,
    (-1, 0, 1, 2), 'simulation time (code units)', 'change from initial (%)')
max_len = max(len(old), len(new))
iterations = svg_chart([
    ('original', '#bd493f', list(range(1, len(old)+1)), [count for count, _ in old]),
    ('retry', '#147d69', list(range(1, len(new)+1)), [count for count, _ in new]),
], list(range(1, max_len+1)), 'FGMRES iterations after cycle 10 restart', 0, 600,
    (0, 150, 300, 450, 600), 'linear solve since cycle 10 restart', 'FGMRES iterations')

html = f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>WOR-92 XD 計算結果 | 2026-11-08</title>
<style>
:root{{font-family:system-ui,-apple-system,"Noto Sans JP",sans-serif;color:#17313a;background:#f3f7f8}}*{{box-sizing:border-box}}
body{{margin:0;line-height:1.65}}header{{background:#123b4a;color:#fff;padding:2rem max(1.2rem,calc((100vw - 1080px)/2))}}
header h1{{margin:.25rem 0;font-size:clamp(1.7rem,3vw,2.4rem)}}header p{{max-width:850px;margin:.3rem 0;color:#dcebef}}
main{{max-width:1080px;margin:auto;padding:1.2rem 1.2rem 3rem}}h2{{margin:2rem 0 .7rem;border-bottom:2px solid #b8d3dc;padding-bottom:.3rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.75rem}}.card,figure,.panel{{background:#fff;border:1px solid #d7e3e7;border-radius:10px;padding:1rem}}
.card strong{{font-size:1.3rem;display:block}}.bad{{border-left:5px solid #bd493f}}.good{{border-left:5px solid #147d69}}.wait{{border-left:5px solid #bd6a25}}
figure{{margin:1rem 0}}figure svg{{width:100%;height:auto}}figcaption{{font-size:.88rem;color:#536b74}}
.legend span{{display:inline-block;margin-right:1rem}}.dot{{display:inline-block;width:.8rem;height:.8rem;border-radius:50%;margin-right:.35rem}}
table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{text-align:left;padding:.55rem .65rem;border-bottom:1px solid #dfe9ec;vertical-align:top}}th{{background:#e2edf1}}
.scroll{{overflow-x:auto}}a{{color:#086b83}}code{{font-size:.88em;overflow-wrap:anywhere}}.note{{background:#fff6e9;border-left:4px solid #bd6a25;padding:.8rem 1rem}}
</style></head><body>
<header><div>ATHENA++ / WOR-92</div><h1>XD計算の確認と再開試験</h1><p>2026-11-08 JST時点の検証記録。第一コア形成の判定にはまだ到達していません。</p></header>
<main>
<section><h2>現在の状態</h2><div class="cards">
<div class="card bad"><strong>337351 · FAILED</strong>P-large-a。cycle 10まで保存後、cycle 13の線形解で300反復上限に達した。</div>
<div class="card good"><strong>337766 / 337771 · PASS</strong>反復上限600で同じcheckpointからcycle 20まで厳格残差目標を維持。</div>
<div class="card wait"><strong>337772 · SUBMITTED</strong>cycle 20 checkpointからのP-large-a継続。投入時は利用枠待ち。</div></div></section>
<section><h2>失敗の切り分け</h2><p>337351の初回区間はcycle 0–10を完了し、Newton 10段階とFGMRES 33回が収束しました。再開区間はcycle 10–12のNewtonが収束した後、cycle 13のFGMRESが300反復で相対残差 <code>1.2641885398e-6</code>にとどまりました。要求値は<code>2e-8</code>です。計算環境やcheckpoint読み取りの失敗ではありません。</p>
<div class="scroll"><table><thead><tr><th>ジョブ</th><th>条件</th><th>結果</th><th>最大FGMRES反復</th><th>最大真相対残差</th></tr></thead><tbody>
<tr><td>337351</td><td>64 rank、上限300、0→10→13</td><td>失敗。最終保存checkpointはcycle 10</td><td>300（未収束）</td><td>1.2641885398e-6</td></tr>
<tr><td>337766</td><td>同じcycle 10から上限600、cycle 14まで</td><td>成功。Newton 4/4、FGMRES 9/9</td><td>480</td><td>{max(value for _,value in new):.10e}</td></tr>
<tr><td>337771</td><td>cycle 14から上限600、cycle 20まで</td><td>成功。Newton 6/6、FGMRES 14/14</td><td>{max(count for count,_ in next_rows)}</td><td>{max(value for _,value in next_rows):.10e}</td></tr>
</tbody></table></div><p>上限600は収束基準を緩めません。cycle 20のcheckpointは960 MeshBlock、時刻<code>268.26820770391913</code>で正常に保存されました。337766/337771ともopacity表の範囲外・clamp・非有限値の診断件数は0です。</p>
<figure>{iterations}<figcaption>赤：元の300反復上限、緑：600反復での再試験。同じcycle 10 checkpointから開始。線の連結は解の順序を示します。</figcaption></figure>
<div class="legend"><span><i class="dot" style="background:#bd493f"></i>元の上限300</span><span><i class="dot" style="background:#147d69"></i>再試験の上限600</span></div></section>
<section><h2>計算履歴と物理量</h2><figure>{trajectory}<figcaption>水色：最大密度、橙色：最小Jeans解像度。337351のhistory出力から重複したrestart行を除いたもの。</figcaption></figure>
<div class="legend"><span><i class="dot" style="background:#0e7490"></i>ρmax</span><span><i class="dot" style="background:#bd6a25"></i>Jeans minimum</span></div>
<p>失敗直前のhistory時刻は約180.712、最大密度<code>5.34277e-18 g/cm³</code>、最大気体温度<code>13.9862 K</code>、Jeans最小<code>60.1467</code>。第一コア比較の密度<code>1e-10 g/cm³</code>には達していません。</p>
<p class="note">historyの質量・エネルギーは限られた桁数で出力されています。ここにある値だけから高精度の保存則を判定しません。</p></section>
<section><h2>証拠と再現</h2><ul>
<li><a href="README.md">検証記録・Git hash・入力・制限</a></li>
<li><a href="wor92_xd_long337351_evidence.tar.gz">337351のログ、history、provenance</a></li>
<li><a href="wor92_xd_fgmres600_debug_evidence.tar.gz">337766/337771のログ、history、provenance</a></li>
<li><a href="wor92_pdebug_cycle10_fgmres600.sbatch">cycle 10からのP-debug試験</a> / <a href="wor92_pdebug_cycle14_fgmres600.sbatch">cycle 14からのP-debug試験</a></li>
<li><a href="wor92_plarge_cap6_fgmres600_continuation.sbatch">337772に投入したP-large-a継続スクリプト</a></li>
</ul><p>コード変更は分離ブランチ<code>codex/wor92-xd-restart-sync-20261107</code>、実行binary SHA-256は<code>41f16d20537cef3615a41d1b2150338878d64bf9f9888dc80925edd6a299d72c</code>です。WOR-92は長時間継続と物理・保存検証が残るためIn Progressです。</p></section>
</main></body></html>'''
(ROOT / 'report.html').write_text(html, encoding='utf-8')
print(ROOT / 'report.html')
