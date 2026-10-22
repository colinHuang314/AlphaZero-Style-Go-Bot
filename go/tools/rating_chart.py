"""Redraws docs/rating_history.svg (rating over time; data from docs/RESULTS.md). python tools/rating_chart.py docs/rating_history.svg"""
import datetime as dt
import sys

out = sys.argv[1]
D = lambda m, d, h=12: dt.datetime(2026, m, d, h)
est = [(D(9, 25, 7), 2200), (D(9, 26, 8), 2344), (D(9, 27, 8), 2496), (D(9, 28, 8), 2548), (D(9, 29, 8), 2567),
       (D(10, 1, 8), 2630), (D(10, 5, 9), 2660)]
yard = [(D(10, 7, 9), 2671, 80), (D(10, 8, 9), 2707, 55), (D(10, 9, 14), 2680, 80), (D(10, 10, 14), 2771, 50),
        (D(10, 11, 14), 2775, 50), (D(10, 12, 14), 2752, 50), (D(10, 13, 9), 2811, 48), (D(10, 16, 14), 2767, 50),
        (D(10, 17, 14), 2790, 50), (D(10, 18, 14), 2811, 48), (D(10, 19, 14), 2804, 48)]
cgos = [(D(9, 30, 12), 2630, "c439, 10 blocks", "71 CGOS games: 2630"),
        (D(10, 21, 12), 2854, "E_c2600, 20 blocks", "161 CGOS games: ≈2854")]

W, H, L, R, T, B = 900, 470, 64, 24, 52, 70
x0, x1, y0, y1 = D(9, 24, 0), D(10, 23, 0), 2100, 3100
X = lambda t: L + (t - x0).total_seconds() / (x1 - x0).total_seconds() * (W - L - R)
Y = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
blue, accent, ink, ink2, muted, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
     f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif" role="img" '
     f'aria-label="go-zero CGOS 9x9 rating over time: about 2200 on Sep 25, 2630 on CGOS on Oct 2, 2854 on CGOS on Oct 21">',
     f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
     f'<text x="{L}" y="26" font-size="16" fill="{ink}" font-weight="600">go-zero strength over four weeks of laptop training</text>']
s.append(f'<rect x="{X(D(10, 5, 6)):.1f}" y="{T}" width="{X(D(10, 15, 23)) - X(D(10, 5, 6)):.1f}" height="{H - T - B}" fill="#0b0b0b" fill-opacity="0.04"/>')
for v in range(2100, 3101, 100):
    s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{grid}"/>')
    s.append(f'<text x="{L - 8}" y="{Y(v) + 4:.1f}" font-size="11" fill="{muted}" text-anchor="end">{v}</text>')
for t, lab in [(D(9, 24, 10), "10 blocks"), (D(10, 5, 14), "15 blocks"), (D(10, 16, 6), "20 blocks")]:
    s.append(f'<text x="{X(t):.1f}" y="{T + 16}" font-size="11" fill="{muted}">{lab}</text>')
d = D(9, 25, 0)
while d <= D(10, 22, 0):
    s.append(f'<text x="{X(d):.1f}" y="{H - B + 20}" font-size="11" fill="{muted}" text-anchor="middle">{d:%b} {d.day}</text>')
    d += dt.timedelta(days=4)
s.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="#c3c2b7"/>')
s.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(3000):.1f}" y2="{Y(3000):.1f}" stroke="{muted}" stroke-width="1.5" stroke-dasharray="6 5"/>')
s.append(f'<text x="{W - R - 4}" y="{Y(3000) - 7:.1f}" font-size="11" fill="{muted}" text-anchor="end">goal 3000</text>')
pts = sorted([(t, v) for t, v in est] + [(t, v) for t, v, _ in yard] + [(t, v) for t, v, *_ in cgos])
s.append('<polyline fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round" points="%s"/>'
         % (blue, " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts)))
for t, v, e in yard:
    x, a, b = X(t), Y(v + e), Y(v - e)
    s.append(f'<path d="M{x:.1f},{a:.1f}V{b:.1f}M{x - 3:.1f},{a:.1f}h6M{x - 3:.1f},{b:.1f}h6" stroke="{blue}" stroke-opacity="0.5" stroke-width="1.3"/>')
    s.append(f'<circle cx="{x:.1f}" cy="{Y(v):.1f}" r="4.5" fill="{blue}" stroke="#fff" stroke-width="1.5"/>')
for t, v in est:
    s.append(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="4.5" fill="#fff" stroke="{blue}" stroke-width="2"/>')
for i, (t, v, l1, l2) in enumerate(cgos):
    x, y = X(t), Y(v)
    s.append(f'<rect x="{x - 7:.1f}" y="{y - 7:.1f}" width="14" height="14" transform="rotate(45 {x:.1f} {y:.1f})" fill="{accent}" stroke="#fff" stroke-width="2"/>')
    tx, ty, anchor = (x + 12, y + 30, "start") if i == 0 else (x - 16, y - 24, "end")
    s.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="12" fill="{ink}" text-anchor="{anchor}" font-weight="600">{l1}</text>')
    s.append(f'<text x="{tx:.1f}" y="{ty + 15:.1f}" font-size="12" fill="{ink2}" text-anchor="{anchor}">{l2}</text>')
ly = H - 22
items = [("est", "Estimate from self-play matches"), ("yard", "KataGo yardstick (95% CI)"), ("cgos", "Measured on CGOS")]
lx = L
for kind, lab in items:
    if kind == "est":
        s.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="4.5" fill="#fff" stroke="{blue}" stroke-width="2"/>')
    elif kind == "yard":
        s.append(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="4.5" fill="{blue}"/>')
    else:
        s.append(f'<rect x="{lx}" y="{ly - 9}" width="10" height="10" transform="rotate(45 {lx + 5} {ly - 4})" fill="{accent}"/>')
    s.append(f'<text x="{lx + 16}" y="{ly}" font-size="12" fill="{ink2}">{lab}</text>')
    lx += 16 + 7.0 * len(lab) + 28
s.append("</svg>")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", out)
