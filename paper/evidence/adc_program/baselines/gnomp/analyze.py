"""Tables and paired bootstrap for GNOMP vs NOMP, consistent NOMP and quantiser ML on the common test banks.
Paired bootstrap: 2000 resamples within SNR groups, seed 0 (common.boot_metrics). '*' = 95% CI excludes 0.
Reads the stats written by run_gnomp.py and run_ml.py from the C: cache. Writes analysis.log (stdout) and analysis.json."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import numpy as np
import common as C

CACHE = 'C:/Users/Milon/iabr_deep_research_cache/adc_program/baselines/gnomp/cache'
SNRS = [-10, 0, 10, 15, 25]; BITS = [1, 2, 3, 4]; NS = 300
METHODS = {'NOMP': 'NOMP', 'consistent NOMP': 'consistentNOMP', 'ML (all starts)': 'MLallstarts',
           'GNOMP-L': 'GNOMP-L', 'GNOMP-CFAR': 'GNOMP-CFAR'}
PAIRS = [('GNOMP-L', 'ML (all starts)'), ('GNOMP-CFAR', 'ML (all starts)'), ('GNOMP-L', 'consistent NOMP'),
         ('GNOMP-L', 'NOMP'), ('GNOMP-CFAR', 'GNOMP-L'), ('ML (all starts)', 'NOMP')]
out = dict(per_bits={}, per_snr={})
lines = []


def P(s=''):
    print(s, flush=True); lines.append(s)


def load(tag, bits, snr):
    return np.load(f'{CACHE}/stats_{tag}_b{bits}_s{snr}.npy')


for bits in BITS:
    S = {}; snr_arr = np.concatenate([np.full(NS, s) for s in SNRS])
    for name, tag in METHODS.items():
        try:
            S[name] = np.concatenate([load(tag, bits, s) for s in SNRS])
        except FileNotFoundError:
            pass
    if 'ML (all starts)' not in S or len(S) < len(METHODS):
        P(f'bits {bits}: missing stats ({sorted(set(METHODS) - set(S))}); skipped')
        continue
    mt = {m: C.metric_by_snr(S[m], snr_arr) for m in S}
    Bt = C.boot_metrics(S, snr_arr, B=2000, seed=0)
    rec = dict(pd_paper_mean={m: mt[m]['mean']['pd_paper'] for m in S}, pd_strict_mean={m: mt[m]['mean']['pd_strict'] for m in S},
               pd_paper_by_snr={m: {str(s): mt[m][s]['pd_paper'] for s in SNRS} for m in S}, pairs={})
    P(f'== {bits}-bit: mean over SNR, paper-style Pd (strict Pd)')
    for m in S:
        P(f'   {m:18s} {rec["pd_paper_mean"][m]:.4f} ({rec["pd_strict_mean"][m]:.4f})   by SNR ' +
          ' '.join(f'{s}:{mt[m][s]["pd_paper"]:.3f}' for s in SNRS))
    for a, b in PAIRS:
        d = Bt[a][:, 0] - Bt[b][:, 0]
        pt = rec['pd_paper_mean'][a] - rec['pd_paper_mean'][b]
        lo, hi = float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))
        star = '*' if lo > 0 or hi < 0 else ''
        rec['pairs'][f'{a} - {b}'] = [pt, lo, hi]
        P(f'   {a} - {b}: {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}')
    out['per_bits'][str(bits)] = rec
    # per-SNR GNOMP-L vs ML (paired bootstrap of the single-SNR Pd difference)
json.dump(out, open(f'{HERE}/analysis.json', 'w'), indent=1, default=float)
open(f'{HERE}/analysis.log', 'w').write('\n'.join(lines) + '\n')
print('done', flush=True)
