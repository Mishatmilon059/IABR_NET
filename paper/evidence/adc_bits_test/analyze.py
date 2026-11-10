"""analyze.py <bits>: per-SNR tables + pooled unknown-SNR table with paired bootstrap CIs -> RESULT_b<k>.md, res_b<k>.json (numpy only)."""
import os, sys, json
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np
OUT = 'D:/ai_ml_project/paper/evidence/adc_bits_test'
b = int(sys.argv[1]); SNRS = [0, 10, 20]
METHODS = ['NOMP', 'consistent NOMP', 'CNN read-out', 'ML known sigma', 'ML snr+5', 'ML snr-5', 'ML nominal 10dB', 'ML joint sigma',
           'hybrid known sigma', 'hybrid snr+5', 'hybrid snr-5', 'hybrid nominal 10dB', 'hybrid joint sigma']
Z = {s: np.load(f'{OUT}/cells/b{b}_snr{s}.npz', allow_pickle=True) for s in SNRS}
S = {m: [Z[s][m] for s in SNRS] for m in METHODS}   # per SNR (n,7): good, bad, dropped, src_ok, src_tot, sumsq, L
N = len(S['NOMP'][0])


def pd_(x):
    g, bd = x[:, 0].sum(), x[:, 1].sum(); sb = 2 * (x[:, 2] * x[:, 6]).sum()
    return g / (g + bd), g / (g + bd + sb)


rng = np.random.default_rng(0); B = 2000
R = [rng.integers(0, N, (B, N)) for _ in SNRS]


def boot(m):
    """pooled over the three SNR groups; stratified resampling, same indices for every method. returns (B,2): paper, strict"""
    g = sum(S[m][i][:, 0][R[i]].sum(1) for i in range(3)); bd = sum(S[m][i][:, 1][R[i]].sum(1) for i in range(3))
    sb = sum((2 * S[m][i][:, 2] * S[m][i][:, 6])[R[i]].sum(1) for i in range(3))
    return np.stack([g / (g + bd), g / (g + bd + sb)], 1)


BT = {m: boot(m) for m in METHODS}
pool = {m: pd_(np.concatenate(S[m])) for m in METHODS}
ci = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))
PAIRS = [('CNN read-out', 'NOMP'), ('hybrid known sigma', 'ML known sigma'), ('hybrid nominal 10dB', 'ML known sigma'), ('hybrid nominal 10dB', 'ML nominal 10dB'),
         ('hybrid joint sigma', 'ML known sigma'), ('hybrid joint sigma', 'ML joint sigma'), ('hybrid snr+5', 'ML snr+5'), ('hybrid snr-5', 'ML snr-5'),
         ('ML snr+5', 'ML known sigma'), ('ML snr-5', 'ML known sigma'), ('ML nominal 10dB', 'ML known sigma'), ('ML joint sigma', 'ML known sigma'),
         ('CNN read-out', 'ML known sigma'), ('CNN read-out', 'ML nominal 10dB'), ('hybrid nominal 10dB', 'CNN read-out')]
pairs = {}
for a, c in PAIRS:
    pt = [pool[a][k] - pool[c][k] for k in (0, 1)]
    d = BT[a] - BT[c]
    pairs[f'{a} - {c}'] = dict(paper=[pt[0], *ci(d[:, 0])], strict=[pt[1], *ci(d[:, 1])])
fm = lambda t: f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]' + ('*' if t[1] > 0 or t[2] < 0 else '')
lat = json.load(open(f'{OUT}/latency_b{b}.json'))
res = dict(bits=b, n=N, pool={m: dict(paper=pool[m][0], strict=pool[m][1], ci_paper=list(ci(BT[m][:, 0])), ci_strict=list(ci(BT[m][:, 1]))) for m in METHODS},
           per_snr={s: {m: pd_(S[m][i])[0] for m in METHODS} for i, s in enumerate(SNRS)}, pairs=pairs, latency=lat)
json.dump(res, open(f'{OUT}/res_b{b}.json', 'w'), indent=1, default=float)

L = [f'# RESULT {b}-bit ADC (peak AGC, L=3, ideal hardware)', '',
     f'Fresh scenes seeds 14000/14010/14020 (0/10/20 dB), {N} scenes per cell, same scenes for all methods. Pd = paper-style Pd (|err| <= 1 deg). Script: adc_bits.py, analyze.py.', '',
     '## Per-SNR Pd (paper-style)', '', '| method | 0 dB | 10 dB | 20 dB |', '|---|---|---|---|']
for m in METHODS:
    L.append(f'| {m} | ' + ' | '.join(f'{res["per_snr"][s][m]:.3f}' for s in SNRS) + ' |')
L += ['', f'## Pooled unknown-SNR (all {3 * N} scenes; paired bootstrap 95% CI, 2000 resamples, seed 0, stratified by SNR)', '',
      '| method | Pd paper [95% CI] | Pd strict [95% CI] |', '|---|---|---|']
for m in METHODS:
    r = res['pool'][m]
    L.append(f'| {m} | {r["paper"]:.3f} [{r["ci_paper"][0]:.3f},{r["ci_paper"][1]:.3f}] | {r["strict"]:.3f} [{r["ci_strict"][0]:.3f},{r["ci_strict"][1]:.3f}] |')
L += ['', '## Paired differences (pooled; * = CI excludes 0)', '', '| a - b | paper | strict |', '|---|---|---|']
for k, v in pairs.items():
    L.append(f'| {k} | {fm(v["paper"])} | {fm(v["strict"])} |')
L += ['', '## Latency (seconds per scene, 1 thread, 12 scenes at 10 dB, batch of 12)', '', '| item | s/scene |', '|---|---|']
for k, v in lat.items():
    L.append(f'| {k} | {v:.4f} |')
open(f'{OUT}/RESULT_b{b}.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
