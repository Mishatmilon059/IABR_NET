"""Exact read-out of the vector figures of LoRD-Net (Khobahi, Shlezinger, Soltanalian, Eldar, arXiv 2102.02993v1 = the only
arXiv version; IEEE TSP 69:5651-5664, 2021). The figures in the arXiv PDF are vector graphics (matplotlib), so every plotted
data point is a polyline vertex whose page coordinates can be read exactly with PyMuPDF and converted to data units
through the major tick marks and their labels. The only error left is the PDF's coordinate rounding (~0.01 pt), i.e.
far below one marker width. Nothing here is a re-simulation: these are the numbers the authors plotted.
Usage: python -P lordnet_digitize.py <path to lordnet_arxiv_2102.02993v1.pdf>
"""
import sys, json, math, os, re
import fitz  # PyMuPDF

PDF = sys.argv[1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lordnet_digitized.json')
doc = fitz.open(PDF)
BLACK = (0.0, 0.0, 0.0)


def is_gray(c):
    return c is not None and len(c) == 3 and abs(c[0] - c[1]) < 1e-3 and abs(c[1] - c[2]) < 1e-3 and 0.5 < c[0] < 0.95


def seglist(path):
    return [(it[1].x, it[1].y, it[2].x, it[2].y) for it in path['items'] if it[0] == 'l']


def spans(page):
    out = []
    for b in page.get_text('dict')['blocks']:
        for l in b.get('lines', []):
            for s in l['spans']:
                if s['text'].strip():
                    out.append((s['text'], fitz.Rect(s['bbox'])))
    return out


def linfit(px, val):
    n = len(px); mx = sum(px) / n; mv = sum(val) / n
    sxx = sum((a - mx) ** 2 for a in px); sxv = sum((a - mx) * (b - mv) for a, b in zip(px, val))
    k = sxv / sxx; c = mv - k * mx
    res = max(abs(k * a + c - b) for a, b in zip(px, val))
    return k, c, res


results = {}
for pno in range(doc.page_count):
    page = doc[pno]
    dr = page.get_drawings()
    sp = spans(page)
    # ---- 1. axes frames from long black spines
    hs, vs = [], []
    for p in dr:
        if p['color'] != BLACK or len(p['items']) != 1 or p['items'][0][0] != 'l':
            continue
        x0, y0, x1, y1 = seglist(p)[0]
        if abs(y0 - y1) < 0.05 and abs(x1 - x0) > 60:
            hs.append((min(x0, x1), max(x0, x1), y0, p['width']))
        if abs(x0 - x1) < 0.05 and abs(y1 - y0) > 60:
            vs.append((min(y0, y1), max(y0, y1), x0, p['width']))
    frames = []
    for (xa, xb, yh, w) in hs:
        for (xa2, xb2, yh2, w2) in hs:
            if yh2 > yh + 20 and abs(xa - xa2) < 0.2 and abs(xb - xb2) < 0.2:
                lv = [v for v in vs if abs(v[2] - xa) < 0.2 and abs(v[0] - yh) < 0.2 and abs(v[1] - yh2) < 0.2]
                rv = [v for v in vs if abs(v[2] - xb) < 0.2 and abs(v[0] - yh) < 0.2 and abs(v[1] - yh2) < 0.2]
                if lv and rv:
                    frames.append((xa, yh, xb, yh2, w))
    frames = sorted(set(frames), key=lambda f: (round(f[1] / 50), f[0]))
    for fi, (X0, Y0, X1, Y1, wsp) in enumerate(frames):
        # ---- 2. major ticks (short black segments on the left / bottom spine, the longest ones)
        xt, yt = [], []
        for p in dr:
            if p['color'] != BLACK or len(p['items']) != 1 or p['items'][0][0] != 'l':
                continue
            a, b, c, d = seglist(p)[0]
            L = math.hypot(c - a, d - b)
            if L > 4:
                continue
            if abs(a - c) < 0.05 and X0 - 0.1 <= a <= X1 + 0.1 and abs(min(b, d) - Y1) < 0.2:
                xt.append((a, L))
            if abs(b - d) < 0.05 and Y0 - 0.1 <= b <= Y1 + 0.1 and abs(max(a, c) - X0) < 0.2:
                yt.append((b, L))
        if not xt or not yt:
            continue
        Lx = max(l for _, l in xt); Ly = max(l for _, l in yt)
        xmaj = sorted(set(round(a, 3) for a, l in xt if l > 0.8 * Lx))
        ymaj = sorted(set(round(b, 3) for b, l in yt if l > 0.8 * Ly))
        # ---- 3. tick labels
        xlab = []
        for a in xmaj:
            c = [(t, r) for t, r in sp if Y1 < r.y0 < Y1 + 12 and abs((r.x0 + r.x1) / 2 - a) < 5 and re.fullmatch(r'-?[0-9.]+', t.strip())]
            if c:
                xlab.append((a, float(c[0][0])))
        ylab, ylog = [], None
        for b in ymaj:
            c = [(t, r) for t, r in sp if X0 - 25 < r.x1 < X0 and abs((r.y0 + r.y1) / 2 - b) < 4.5 and re.fullmatch(r'[0-9.]+', t.strip())]
            if not c:
                continue
            t, r = sorted(c, key=lambda q: -q[1].x1)[0] if len(c) == 1 else sorted(c, key=lambda q: q[1].x0)[0]
            if t.strip() == '10':
                sup = sorted([(t2, r2) for t2, r2 in sp if abs(r2.x0 - r.x1) < 6 and r2.x0 >= r.x1 - 0.5 and r2.y0 < r.y0 + 2 and r2.y1 < r.y1 + 1 and r2.y0 > r.y0 - 4 and r2 != r],
                             key=lambda q: q[1].x0)
                s = ''.join(t2 for t2, _ in sup[:3]).replace('−', '-').strip()
                m = re.match(r'(-?[0-9]+)', s)
                if m:
                    ylab.append((b, float(m.group(1)))); ylog = True
                    continue
            ylab.append((b, float(t))); ylog = False
        if len(xlab) < 2 or len(ylab) < 2:
            continue
        # x scale: linear or log2/log10 (choose the better fit)
        kx, cx, rx = linfit([a for a, _ in xlab], [v for _, v in xlab])
        xlogscale = False
        if all(v > 0 for _, v in xlab):
            k2, c2, r2 = linfit([a for a, _ in xlab], [math.log10(v) for _, v in xlab])
            if r2 * 1e3 < rx:
                kx, cx, rx, xlogscale = k2, c2, r2, True
        ky, cy, ry = linfit([b for b, _ in ylab], [v for _, v in ylab])
        X = (lambda a: 10 ** (kx * a + cx)) if xlogscale else (lambda a: kx * a + cx)
        Yv = (lambda b: 10 ** (ky * b + cy)) if ylog else (lambda b: ky * b + cy)
        # ---- 4. caption of this frame: nearest text line starting 'Fig. N.' below the frame with x-overlap
        capline, best = '', None
        for blk in page.get_text('dict')['blocks']:
            for ln in blk.get('lines', []):
                t = ''.join(s_['text'] for s_ in ln['spans']).strip()
                lb = fitz.Rect(ln['bbox'])
                if re.match(r'Fig\. \d+\.', t) and lb.y0 > Y1 and lb.x0 < X1 and lb.x1 > X0 - 60:
                    if best is None or lb.y0 < best:
                        best, capline = lb.y0, t[:75]
        sub = [(t, r) for t, r in sp if Y1 + 12 < r.y0 < Y1 + 45 and X0 - 20 < r.x0 < X1 and re.fullmatch(r'\([ab]\)', t.strip())]
        subl = sub[0][0].strip() if sub else ''
        # ---- 5. curves, legend samples
        curves = []
        for p in dr:
            c = p['color']
            if c is None or is_gray(c) or p['width'] < 1.5 * wsp:
                continue
            sg = seglist(p)
            if not sg or len(sg) != len(p['items']):
                continue
            r = p['rect']
            if not (r.x0 >= X0 - 1 and r.x1 <= X1 + 1 and r.y0 >= Y0 - 1 and r.y1 <= Y1 + 1):
                continue
            dash = p.get('dashes') or ''
            if len(sg) == 1 and abs(sg[0][1] - sg[0][3]) < 0.05 and 6 < abs(sg[0][2] - sg[0][0]) < 16:
                # legend sample: find label text right of it on the same line
                yy = sg[0][1]; xe = max(sg[0][0], sg[0][2])
                lab = sorted([(t, rr) for t, rr in sp if abs((rr.y0 + rr.y1) / 2 - yy) < 2.5 and xe < rr.x0 < xe + 160], key=lambda q: q[1].x0)
                txt, last = [], None
                for t, rr in lab:
                    if last is not None and rr.x0 - last > 8:
                        break
                    txt.append(t); last = rr.x1
                curves.append(dict(kind='legend', color=c, dash=dash, label=''.join(txt).strip(), y=yy))
                continue
            pts = [(sg[0][0], sg[0][1])] + [(s[2], s[3]) for s in sg]
            if pts[-1][0] - pts[0][0] < 20:
                continue
            curves.append(dict(kind='curve', color=c, dash=dash, pts=pts))
        leg = [q for q in curves if q['kind'] == 'legend']
        data = []
        for q in [q for q in curves if q['kind'] == 'curve']:
            m = [l for l in leg if l['color'] == q['color'] and l['dash'] == q['dash']]
            label = m[0]['label'] if len(m) == 1 else ('AMBIGUOUS:' + '|'.join(l['label'] for l in m))
            xy = [(round(X(a), 4), float('%.6g' % Yv(b)), bool(X0 - 0.2 <= a <= X1 + 0.2 and Y0 - 0.2 <= b <= Y1 + 0.2)) for a, b in q['pts']]
            data.append(dict(label=label, color=[round(v, 3) for v in q['color']], dashed=q['dash'] not in ('', '[] 0'), points=xy))
        key = f'p{pno + 1}_f{fi}'
        results[key] = dict(page=pno + 1, frame=[round(v, 2) for v in (X0, Y0, X1, Y1)], caption=capline, sub=subl,
                            x_ticks=xlab, y_ticks=ylab, x_log=xlogscale, y_log=ylog, fit_residual_x=rx, fit_residual_y=ry, curves=data)

json.dump(results, open(OUT, 'w'), indent=0)
for k, v in results.items():
    print(f"\n### {k}: page {v['page']} {v['sub']} | caption: {v['caption']!r}")
    print(f"    x ticks {[t[1] for t in v['x_ticks']]} (log={v['x_log']}, fit resid {v['fit_residual_x']:.1e}); "
          f"y ticks {[t[1] for t in v['y_ticks']]} (log={v['y_log']}, fit resid {v['fit_residual_y']:.1e})")
    for c in v['curves']:
        pts = c['points']
        print(f"    [{c['label']}] {'dashed' if c['dashed'] else 'solid'} n={len(pts)}"
              + ('' if all(p[2] for p in pts) else '  (some vertices outside frame)'))
        print('      ' + ', '.join(f'({p[0]:g}, {p[1]:.3g})' for p in pts))
print('\nwritten', OUT)

# ---------------- derived checks against statements in the text ----------------
NAMES = {'p9_f0': 'Fig4a_128x16_Rayleigh', 'p9_f1': 'Fig4b_128x16_COST2100', 'p9_f2': 'Fig5a_64x10_Rayleigh',
         'p9_f3': 'Fig5b_64x10_COST2100', 'p10_f0': 'Fig6_ThetaH_vs_ThetaHC', 'p10_f1': 'Fig8_BER_vs_B',
         'p10_f2': 'Fig7_unfolded_benchmarks', 'p11_f0': 'Fig9_training_curves_8dB', 'p12_f0': 'Fig10a_per_layer_Rayleigh_8dB',
         'p12_f1': 'Fig10b_per_layer_COST2100_8dB'}
print('\n=== frame -> figure mapping used below (by caption line + subfigure label + legend content) ===')
for k, v in results.items():
    print(f"  {k} -> {NAMES.get(k, '?')}   caption={v['caption'][:40]!r} sub={v['sub']!r}")


def curve(key, lab):
    c = [q for q in results[key]['curves'] if q['label'] == lab]
    assert len(c) == 1, (key, lab, [q['label'] for q in results[key]['curves']])
    return [(p[0], p[1]) for p in c[0]['points']]


def snr_at(pts, ber):  # log-linear interpolation
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if (y0 - ber) * (y1 - ber) <= 0:
            t = (math.log10(ber) - math.log10(y0)) / (math.log10(y1) - math.log10(y0))
            return x0 + t * (x1 - x0)
    return float('nan')


print('\n=== text claim (Sec. IV-B): Fig. 4(a), BER 1e-2: LoRD-Net B=2048 at ~3 dB, DeepSIC needs ~5 dB ===')
for lab in ['LoRD-Net-TrainingSizeB=2048', 'DeepSIC-TrainingSizeB=2048', 'LoRD-Net-TrainingSizeB=1024', 'nML-PerfectCSI']:
    print(f"  {lab:32s} SNR at BER 1e-2 = {snr_at(curve('p9_f0', lab), 1e-2):.2f} dB")
print('=== text claim: Fig. 4(b), BER 3e-2 (similar behaviour) ===')
for lab in ['LoRD-Net-TrainingSizeB=2048', 'DeepSIC-TrainingSizeB=2048', 'LoRD-Net-TrainingSizeB=1024', 'nML-PerfectCSI']:
    print(f"  {lab:32s} SNR at BER 3e-2 = {snr_at(curve('p9_f1', lab), 3e-2):.2f} dB")

print('\n=== BER tables (Figs. 4-5) and ratio nML(perfect CSI) / LoRD-Net(B=2048) ===')
for key in ['p9_f0', 'p9_f1', 'p9_f2', 'p9_f3']:
    print(' ', NAMES[key])
    labs = ['nML-PerfectCSI', 'DeepSIC-TrainingSizeB=2048', 'LoRD-Net-TrainingSizeB=1024', 'LoRD-Net-TrainingSizeB=2048']
    cs = {l: curve(key, l) for l in labs}
    print('    SNR   ' + ' '.join(f'{l[:22]:>23s}' for l in labs) + '   nML/LoRD2048  DeepSIC/LoRD2048')
    for i in range(6):
        row = [cs[l][i][1] for l in labs]
        print(f"    {cs[labs[0]][i][0]:4.0f}  " + ' '.join(f'{v:23.3e}' for v in row) + f"   {row[0] / row[3]:10.1f}   {row[1] / row[3]:10.1f}")

print('\n=== are plotted BERs integer error counts on a 2048 x n-bit test set? (n = 16 for 128x16, 10 for 64x10) ===')
for key, n in [('p9_f0', 16), ('p9_f1', 16), ('p9_f2', 10), ('p9_f3', 10), ('p10_f0', 16), ('p10_f1', 16), ('p10_f2', 16),
               ('p11_f0', 16), ('p12_f0', 16), ('p12_f1', 16)]:
    for c in results[key]['curves']:
        ys = [p[1] for p in c['points'] if p[1] < 0.01]
        if not ys:
            continue
        k = [y * 2048 * n for y in ys]
        dev = max(abs(v - round(v)) for v in k)
        print(f"  {NAMES[key]:32s} {c['label'][:42]:42s} points<1e-2: {len(ys):3d}  errors=BER*2048*{n}: "
              f"{[round(v, 2) for v in k[:6]]}  max|frac| {dev:.2f}")

print('\n=== Fig. 10 / Fig. 9 read-outs at SNR 8 dB, 128x16 ===')
for key in ['p12_f0', 'p12_f1']:
    for c in results[key]['curves']:
        p = c['points']
        print(f"  {NAMES[key]:32s} {c['label']:28s} layer index 0,5,10,15,20,29 -> " + ', '.join(f'{p[i][1]:.3g}' for i in [0, 5, 10, 15, 20, 29]))
for c in results['p11_f0']['curves']:
    p = c['points']
    print(f"  Fig9 {c['label']:45s} first {p[0][1]:.3g}, at x=80 {min(p, key=lambda q: abs(q[0] - 80))[1]:.3g}, final {p[-1][1]:.3g}")
