"""Where do SignalNet's plotted curves cross (a) the paper's own learning thresholds and (b) the thresholds re-computed in
signalnet_thresholds.log? Uses the exact plotted points digitised from the vector figures of arXiv:2106.05490v2
(signalnet_digitize.py -> signalnet_digitized_full.json). Curves = per-m network estimators (solid) and Periodogram/FFT (dotted).
Usage: python signalnet_crossings.py signalnet_digitized_full.json
"""
import json, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D = json.load(open(sys.argv[1]))
COLM = {'red': 1, '[0.0, 0.5, 0.0]': 2, 'green': 2, 'blue': 3, 'black': 4, 'cyan': 5}
PAPER_F = [-18.06, -16.44, -16.01, -15.81, -15.69]          # eq. (45), as plotted
MC_F = [-22.83, -20.67, -20.21, -20.06, -20.01]             # Monte Carlo of Algorithm 1 (signalnet_thresholds.log)
AMP = {'paper text 0.0675': -11.71, 'plotted line': -11.25}
PHASE = 3.29


def describe(pts, thr, better_is_lower=True):
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    below = y < thr if better_is_lower else y > thr
    if below.all(): return 'better than threshold at every SNR (-20..19 dB)'
    if not below.any(): return 'never better than threshold (min %.2f vs %.2f)' % (y.min(), thr)
    first = x[np.argmax(below)]
    stays = below[np.argmax(below):].all()
    return 'better from SNR %+d dB%s (value at -20 dB: %.2f)' % (first, '' if stays else ' (not monotone: re-crosses)', y[0])


for fid, label in [('251', 'frequency, 1-bit'), ('285', 'frequency, 3-bit')]:
    print('==', label, '(TSP Fig. 5)')
    for cv in D[fid]['curves']:
        if 'points' not in cv or cv['dash'] != 'solid': continue
        m = COLM.get(cv['color'])
        if m is None: continue
        print('   network m=%d: vs paper thr %.2f dB -> %s | vs re-computed thr %.2f dB -> %s'
              % (m, PAPER_F[m - 1], describe(cv['points'], PAPER_F[m - 1]), MC_F[m - 1], describe(cv['points'], MC_F[m - 1])))
    for cv in D[fid]['curves']:
        if 'points' not in cv or cv['dash'] == 'solid': continue
        m = COLM.get(cv['color'])
        if m is None or len(cv['points']) < 30: continue
        print('   periodogram m=%d: vs re-computed thr -> %s' % (m, describe(cv['points'], MC_F[m - 1])))
for fid, label in [('319', 'amplitude, 1-bit'), ('351', 'amplitude, 3-bit')]:
    print('==', label, '(TSP Fig. 6)')
    for cv in D[fid]['curves']:
        if 'points' not in cv or cv['dash'] != 'solid': continue
        m = COLM.get(cv['color']); ys = [p[1] for p in cv['points']]
        print('   network m=%d: range %.2f..%.2f dB, spread over SNR %.2f dB; vs -11.71 dB -> %s' % (m, min(ys), max(ys), max(ys) - min(ys), describe(cv['points'], -11.71)))
for fid, label in [('387', 'phase, 1-bit'), ('421', 'phase, 3-bit')]:
    print('==', label, '(TSP Fig. 7), threshold pi^2/3 = 3.29')
    for cv in D[fid]['curves']:
        if 'points' not in cv or cv['dash'] != 'solid': continue
        m = COLM.get(cv['color'])
        print('   network m=%d: %s; value at 19 dB %.2f' % (m, describe(cv['points'], PHASE), cv['points'][-1][1]))
print('== detection (TSP Fig. 4 = arXiv Fig. 6), loss (11); thresholds: paper 1.67 (continuous), integer-output 1.744; legend labels as printed in arXiv Fig. 6')
names = {'tab:blue': '"1 bit Detection Mod"', 'tab:orange': '"2 bit Detection Mod" (text: three-bit)', 'tab:green': '"1 bit AIC"', 'tab:red': '"3 bit AIC"',
         'tab:purple': '"2 bit MDL"', 'tab:brown': '"6 bit MDL"', 'tab:pink': '"1 bit EM-VAMP"', 'tab:gray': '"3 bit EM-VAMP"'}
for cv in D['142']['curves']:
    if 'points' not in cv: continue
    print('   %-42s vs 1.67 -> %s | vs 1.744 -> %s' % (names.get(cv['color'], cv['color']), describe(cv['points'], 1.67), describe(cv['points'], 1.744)))
print('== detection MSE (arXiv Fig. 7 / TSP suppl. Fig. 12); blind MSE threshold Var(m) = 2.0 (not drawn in the paper)')
names7 = {'tab:blue': '"1 bit Detection Mod"', 'tab:orange': '"3 bit Detection Mod"', 'tab:green': '"2 bit AIC"', 'tab:red': '"3 bit AIC"',
          'tab:purple': '"2 bit MDL"', 'tab:brown': '"3 bit MDL"', 'tab:pink': '"1 bit EM-VAMP"', 'tab:gray': '"3 bit EM-VAMP"'}  # arXiv Fig. 7 legend
for cv in D['179']['curves']:
    if 'points' not in cv: continue
    print('   %-42s vs 2.0 -> %s; value at 19 dB %.2f' % (names7.get(cv['color'], cv['color']), describe(cv['points'], 2.0), cv['points'][-1][1]))
