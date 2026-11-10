# Digitize SignalNet result curves from the vector figures of arXiv:2106.05490v2 (exact plotted points).
# Input: JSON produced by vecdig.py (per form XObject: gridlines + polylines in PDF user units).
# Axis tick values were identified from the label glyph structure (number of digits / minus signs)
# and cross-checked against the published (IEEE TSP 2022) raster figures Fig. 4-7 and 9.
# Usage: python -I signalnet_digitize.py signalnet_vec.json out.json
import json, sys
sys.stdout.reconfigure(encoding='utf-8')
V = json.load(open(sys.argv[1]))
# form id -> (arXiv fig, published fig, y tick values bottom->top, y in dB?)
CAL = {
    '142': ('arXiv Fig.6 / TSP Fig.4 detection loss (11)', [0, 10, 20, 30, 40]),
    '179': ('arXiv Fig.7 / TSP suppl. Fig.12 detection MSE', [0, 2, 4, 6, 8, 10]),
    '215': ('arXiv Fig.8 architecture comparison 3-bit m=5 (MSE dB)', [-35, -30, -25, -20, -15, -10, -5, 0, 5]),
    '251': ('arXiv Fig.9a / TSP Fig.5a frequency MSE dB, 1-bit', [-70, -60, -50, -40, -30, -20, -10, 0]),
    '285': ('arXiv Fig.9b / TSP Fig.5b frequency MSE dB, 3-bit', [-70, -60, -50, -40, -30, -20, -10, 0]),
    '319': ('arXiv Fig.10a / TSP Fig.6a amplitude MSE dB, 1-bit', [-25, -20, -15, -10, -5, 0, 5, 10]),
    '351': ('arXiv Fig.10b / TSP Fig.6b amplitude MSE dB, 3-bit', [-25, -20, -15, -10, -5, 0, 5, 10]),
    '387': ('arXiv Fig.11a / TSP Fig.7a phase MSE (linear, rad^2), 1-bit', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    '421': ('arXiv Fig.11b / TSP Fig.7b phase MSE (linear, rad^2), 3-bit', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    '461': ('arXiv Fig.13 / TSP Fig.9 out-of-distribution m=2 (MSE dB)', [-50, -40, -30, -20, -10, 0, 10]),
}
XT = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
NAMES = {(0.122, 0.467, 0.705): 'tab:blue', (1.0, 0.498, 0.055): 'tab:orange', (0.173, 0.627, 0.173): 'tab:green',
         (0.84, 0.153, 0.157): 'tab:red', (0.58, 0.404, 0.74): 'tab:purple', (0.549, 0.337, 0.294): 'tab:brown',
         (0.891, 0.467, 0.762): 'tab:pink', (0.498, 0.498, 0.498): 'tab:gray', (1.0, 0.0, 0.0): 'red', (0.0, 0.502, 0.0): 'green',
         (0.0, 0.0, 1.0): 'blue', (0.0, 0.0, 0.0): 'black', (0.0, 0.749, 0.749): 'cyan', (0.0, 0.75, 0.75): 'cyan'}
SNRS = [-20, -10, -5, -3, 0, 5, 10, 15, 19]
out = {}
for fid, (title, yt) in CAL.items():
    f = V[fid]; gv, gh = f['grid_v'], f['grid_h']
    if len(gv) != len(XT) or len(gh) != len(yt):
        print('SKIP', fid, title, 'grid mismatch', len(gv), len(gh)); continue
    fx = lambda x: XT[0] + (x - gv[0]) / (gv[-1] - gv[0]) * (XT[-1] - XT[0])
    fy = lambda y: yt[0] + (y - gh[0]) / (gh[-1] - gh[0]) * (yt[-1] - yt[0])
    print('\n==', title, '(form %s)' % fid)
    rows = []
    for cv in f['curves']:
        if cv['n'] < 30 and not (cv['n'] == 2 and cv['dash'].strip('[] ') and cv['lw'] >= 15):
            continue
        col = NAMES.get(tuple(cv['color']), str(cv['color']))
        dash = cv['dash'].strip('[] ') or 'solid'
        pts = [(round(fx(x), 2), round(fy(y), 2)) for x, y in cv['pts']]
        if cv['n'] == 2:
            print('  horizontal %s %s line at y = %.2f' % (col, dash, pts[0][1])); rows.append({'color': col, 'dash': dash, 'const': pts[0][1]}); continue
        d = {round(x): y for x, y in pts}
        vals = ['%6.2f' % d[s] if s in d else '   n/a' for s in SNRS]
        print('  %-11s %-14s n=%d  ' % (col, dash, cv['n']) + ' '.join('%s:%s' % (s, v) for s, v in zip(SNRS, vals)))
        rows.append({'color': col, 'dash': dash, 'points': pts})
    out[fid] = {'title': title, 'curves': rows}
json.dump(out, open(sys.argv[2], 'w'), indent=0)
