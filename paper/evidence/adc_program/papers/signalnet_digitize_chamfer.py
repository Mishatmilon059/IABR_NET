# Pixel-digitize the Chamfer-loss figure of SignalNet (arXiv v2 Fig. 12 = IEEE TSP 2022 Fig. 8),
# which is embedded as a raster image (917x729 RGB) in arXiv:2106.05490v2.
# Axis: x = SNR [dB] gridlines at -20..20 step 5; y = "cumulative normalized chamfer loss [dB]" gridlines 5..30 step 5
# (tick values read from the image itself and from the published figure).
# Usage: python -I signalnet_digitize_chamfer.py img_425.png img_427.png
import sys, zlib, struct
sys.stdout.reconfigure(encoding='utf-8')


def readpng(p):
    b = open(p, 'rb').read(); i = 8; idat = b''; w = h = 0
    while i < len(b):
        L = struct.unpack('>I', b[i:i + 4])[0]; t = b[i + 4:i + 8]; d = b[i + 8:i + 8 + L]
        if t == b'IHDR': w, h = struct.unpack('>II', d[:8])
        elif t == b'IDAT': idat += d
        i += 12 + L
    raw = zlib.decompress(idat); st = w * 3 + 1
    return w, h, [[tuple(raw[y * st + 1 + 3 * x: y * st + 4 + 3 * x]) for x in range(w)] for y in range(h)]


def runs(idx):
    out = []; s = None
    for k in range(len(idx)):
        if s is None: s = idx[k]
        if k + 1 == len(idx) or idx[k + 1] != idx[k] + 1:
            out.append((s + idx[k]) / 2); s = None
    return out


COL = {'SignalNet (red)': lambda c: c[0] > 200 and c[1] < 60 and c[2] < 60,
       'AIC+Est (green)': lambda c: c[0] < 60 and 90 < c[1] < 170 and c[2] < 60,
       'Detector+FFT (blue)': lambda c: c[0] < 60 and c[1] < 60 and c[2] > 200,
       'AIC+FFT (black)': lambda c: c[0] < 50 and c[1] < 50 and c[2] < 50}
for p, name in zip(sys.argv[1:], ['(a) one-bit', '(b) three-bit']):
    w, h, px = readpng(p)
    gray = lambda c: abs(c[0] - c[1]) < 8 and abs(c[1] - c[2]) < 8 and 150 < c[0] < 200
    vcols = [x for x in range(w) if sum(gray(px[y][x]) for y in range(h)) > 0.35 * h]
    hrows = [y for y in range(h) if sum(gray(px[y][x]) for x in range(w)) > 0.35 * w]
    xs, ys = runs(vcols), runs(hrows)
    print('==', name, p, 'x-grid', [round(v, 1) for v in xs], 'y-grid', [round(v, 1) for v in ys])
    if len(xs) != 9 or len(ys) != 4:
        print('   calibration failed'); continue
    # detected y gridlines (top->bottom) are 25,20,15,10 dB; 30 and 5 dB coincide with the black axes frame
    yvals = [25, 20, 15, 10]
    fx = lambda snr: xs[0] + (snr + 20) / 40 * (xs[-1] - xs[0])
    fy = lambda row: yvals[0] + (row - ys[0]) / (ys[-1] - ys[0]) * (yvals[-1] - yvals[0])
    x_lo, x_hi = int(min(xs)) - 30, int(max(xs)) + 30
    for cname, f in COL.items():
        vals = []
        for snr in [-20, -15, -10, -5, -3, 0, 3, 5, 10, 15, 19]:
            x = int(round(fx(snr))); hits = []
            for dx in (-1, 0, 1):
                hits += [y for y in range(int(ys[0]) - 40, int(ys[-1]) + 40) if 0 <= y < h and f(px[y][x + dx])]
            hits.sort()
            vals.append('%d:%s' % (snr, '%.1f' % fy(hits[len(hits) // 2]) if hits else 'n/a'))
        print('   %-20s ' % cname + ' '.join(vals))
