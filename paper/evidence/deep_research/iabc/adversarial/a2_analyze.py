"""Tables from cache/a1__*.npz and cache/a1_official.npz (written by a1_replicate.py). No simulation.
Bootstrap: 2000 resamples within SNR groups, seed 0, same indices for every estimator / method of a cell (paired).
'% removed' = 1 - sum(MSE_after) / sum(MSE_before) over the scenes (equal-size SNR groups -> pooled over SNR).
usage: python a2_analyze.py > a2_analyze.log"""
import os, sys, json
import numpy as np
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import common as C

AC = os.path.dirname(os.path.abspath(__file__)) + '/cache'
B = 2000; DB2 = None
EST = ['iabc', 'sc_tun', 'sc_snrgate', 'iabc_sct', 'oracle']
fmt = lambda v: f'{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]' + ('*' if v[1] > 0 or v[2] < 0 else '')
pf = lambda v: f'{100 * v[0]:.1f}% [{100 * v[1]:.1f},{100 * v[2]:.1f}]'


def boot_idx(snr, mask):
    rng = np.random.default_rng(0); ix = np.where(mask)[0]; s = snr[ix].astype(int)
    return np.concatenate([g[rng.integers(0, len(g), (B, len(g)))] for g in [ix[s == v] for v in sorted(set(s.tolist()))]], 1)


def removed(z, q, e, mask):
    bef = z[f'{q}_before']; aft = z[f'{q}_{e}']; I = boot_idx(z['snr'], mask)
    bs = 1 - aft[I].sum(1) / bef[I].sum(1)
    return [1 - aft[mask].sum() / bef[mask].sum(), *np.percentile(bs, [2.5, 97.5])]


def diffmse(z, q, e1, e2, mask):
    """paired difference of mean residual MSE (e1 - e2), in deg^2 / dB^2."""
    I = boot_idx(z['snr'], mask); a, b = z[f'{q}_{e1}'], z[f'{q}_{e2}']
    return [a[mask].mean() - b[mask].mean(), *np.percentile(a[I].mean(1) - b[I].mean(1), [2.5, 97.5])]


out = {}
cells = sorted(f[4:-4] for f in os.listdir(AC) if f.startswith('a1__'))
print('## A. % of identifiable per-antenna MSE removed (fresh seeds 8800+k, L=3, 200 scenes/SNR; g3_L1/g3_L6: 300 scenes at SNR 0 and 15)')
for c in cells:
    z = dict(np.load(f'{AC}/a1__{c}.npz')); snr = z['snr']; out[c] = {}
    for q, u in [('ph', 'deg'), ('g', 'dB')]:
        if z[f'{q}_before'].mean() < 1e-8:
            continue
        for lab, m in [('all', np.ones(len(snr), bool)), ('SNR>=10', snr >= 10), ('SNR=0', snr == 0), ('SNR=15', snr == 15)]:
            if not m.any():
                continue
            bm = np.sqrt(z[f'{q}_before'][m].mean())
            row = {e: removed(z, q, e, m) for e in EST}
            out[c][f'{q}_{lab}'] = dict(before_rms=float(bm), **{e: [float(x) for x in v] for e, v in row.items()},
                                        rms_after_iabc=float(np.sqrt(z[f'{q}_iabc'][m].mean())),
                                        meanrms_before=float(np.sqrt(z[f'{q}_before'][m]).mean()), meanrms_iabc=float(np.sqrt(z[f'{q}_iabc'][m]).mean()))
            print(f'{c:7s} {q:2s} {lab:8s} before RMS {bm:6.2f} {u} -> IABC {np.sqrt(z[f"{q}_iabc"][m].mean()):6.2f} (mean per-scene RMS {np.sqrt(z[f"{q}_before"][m]).mean():.2f} -> {np.sqrt(z[f"{q}_iabc"][m]).mean():.2f}) | '
                  + ' | '.join(f'{e} {pf(v)}' for e, v in row.items()), flush=True)
        for lab, m in [('SNR=0', snr == 0), ('SNR=15', snr == 15), ('SNR>=10', snr >= 10)]:
            if m.any():
                d = diffmse(z, q, 'iabc', 'sc_tun', m); d2 = diffmse(z, q, 'iabc', 'sc_snrgate', m)
                out[c][f'{q}_{lab}_iabc_minus_sctun_mse'] = d; out[c][f'{q}_{lab}_iabc_minus_scgate_mse'] = d2
                print(f'{"":7s} {q:2s} {lab:8s} residual MSE IABC - SC tuned {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] | IABC - SC snrgate {d2[0]:+.3f} [{d2[1]:+.3f},{d2[2]:+.3f}] ({u}^2; negative = IABC better)')

print('\n## B. paper Pd, mean over SNR (-10..25), paired 95% CI (* = excludes 0)')
DIFFS = [('IABC+NOMP', 'NOMP'), ('IABR-Net heat', 'NOMP'), ('IABR-v2 dual', 'NOMP'), ('IABC+NOMP', 'NOMP+SC tuned'),
         ('IABC+NOMP+SC tuned', 'NOMP+SC tuned'), ('IABR-Net heat', 'NOMP+SC tuned'), ('NOMP+SC snrgate', 'NOMP+SC tuned'),
         ('IABC+NOMP+SC snrgate', 'NOMP+SC snrgate'), ('IABR-Net heat', 'NOMP+SC snrgate'), ('IABC+NOMP+SC tuned', 'NOMP+SC snrgate')]
for c in cells:
    if c.startswith('g3_L'):
        continue
    z = dict(np.load(f'{AC}/a1__{c}.npz')); snr = z['snr']
    Sd = {k[2:]: z[k] for k in z if k.startswith('S_')}
    mb = {m: C.metric_by_snr(s, snr) for m, s in Sd.items()}; Bt = C.boot_metrics(Sd, snr, B=B, seed=0)
    out[c]['pd'] = {m: mb[m]['mean']['pd_paper'] for m in Sd}
    print(f'{c:7s} ' + ' | '.join(f'{m} {mb[m]["mean"]["pd_paper"]:.4f}' for m in Sd))
    for a, b in DIFFS:
        d = C.ci(Bt[a][:, 0] - Bt[b][:, 0]); d = [mb[a]['mean']['pd_paper'] - mb[b]['mean']['pd_paper'], d[1], d[2]]
        out[c][f'{a} - {b}'] = d
        print(f'{"":7s} {a + " - " + b:42s} {fmt(d)}')
    # per-SNR for the large severities
    if c in ['g6', 'c30_6', 'c20_4', 'ph45', 'clean']:
        for a, b in [('IABC+NOMP', 'NOMP'), ('IABR-Net heat', 'NOMP'), ('IABC+NOMP+SC tuned', 'NOMP+SC tuned'), ('IABR-Net heat', 'NOMP+SC snrgate')]:
            line = []
            for s in sorted(set(snr.astype(int).tolist())):
                m = snr.astype(int) == s; Sm = {a: Sd[a][m], b: Sd[b][m]}; Bm = C.boot_metrics(Sm, snr[m], B=B, seed=0)
                dd = C.metric_by_snr(Sd[a][m], snr[m])['mean']['pd_paper'] - C.metric_by_snr(Sd[b][m], snr[m])['mean']['pd_paper']
                lo, hi = np.percentile(Bm[a][:, 0] - Bm[b][:, 0], [2.5, 97.5]); line.append(f'{s}:{fmt([dd, lo, hi])}')
                out[c].setdefault('per_snr', {}).setdefault(f'{a} - {b}', {})[s] = [dd, lo, hi]
            print(f'{"":7s} per SNR {a + " - " + b:38s} ' + ' '.join(line))

print('\n## C. official ideal-hardware bank: NOMP and IABC+NOMP re-run here from cached Ht (CNN re-run, no cached predictions)')
z = dict(np.load(f'{AC}/a1_official.npz')); snr = z['snr']; Sd = {'NOMP': z['S_NOMP'], 'IABC+NOMP': z['S_IABC_NOMP']}
mb = {m: C.metric_by_snr(s, snr)['mean']['pd_paper'] for m, s in Sd.items()}; Bt = C.boot_metrics(Sd, snr, B=B, seed=0)
d = C.ci(Bt['IABC+NOMP'][:, 0] - Bt['NOMP'][:, 0]); d = [mb['IABC+NOMP'] - mb['NOMP'], d[1], d[2]]
out['official'] = dict(pd=mb, diff=d, fresh_vs_cached_Hc=float(z['dHc']))
print(f'N={len(snr)} NOMP {mb["NOMP"]:.4f} IABC+NOMP {mb["IABC+NOMP"]:.4f} diff {fmt(d)} | max rel |fresh Hc - cached Hc| {float(z["dHc"]):.1e}')
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + '/a2_results.json', 'w'), indent=1, default=float)
