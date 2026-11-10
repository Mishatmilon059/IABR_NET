"""Compact summary tables for REPORT.md, read from results.json (written by s3_analyze.py) and s4_banks.json. No new simulation.
'% removed' = 1 - MSE_after / MSE_before of the identifiable per-antenna error (per-scene RMS^2 averaged over the 200 scenes of one SNR).
'gated oracle' = per SNR, the better of {no correction, oracle ALS with true angles}: an upper bound on what an SNR-aware estimator
built on the classical ALS model could reach (not a method; context only).
usage: python s5_summary.py > s5_summary.log"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f'{HERE}/results.json'))
cells = R['cells']; SNRS = [str(s) for s in R['meta']['snrs']]


def pr(c, q, e, s):
    ent = cells[c]['est'][f'{q}_{e}']
    return 100 * ent['per_snr_pct_removed_mse'][s] if 'per_snr_pct_removed_mse' in ent else float('nan')


print('## A. % of identifiable per-antenna MSE removed, by SNR (positive = removed, negative = added error)')
print('estimators: IABC | self-cal tuned (gated, 1,2,4) | self-cal gate open 4 it | oracle ALS (true angles)')
for q, cl in [('ph', ['ph2', 'ph5', 'ph8', 'ph12', 'ph20', 'ph30', 'ph45', 'c8_3', 'c30_6']), ('g', ['g1', 'g2', 'g3', 'g4', 'g6', 'c8_3', 'c30_6'])]:
    print(f'\n### {"phase" if q == "ph" else "gain"}')
    print('| cell | train range | before RMS @25 dB | ' + ' | '.join(f'SNR {s}: IABC / SC-tuned / SC-open / oracle' for s in ['0', '10', '15', '25']) + ' | all-SNR IABC [CI] | all-SNR SC-tuned [CI] |')
    print('|' + '---|' * 9)
    for c in cl:
        e = cells[c]['est']; b = e[f'{q}_before']['per_snr']['25']['mean']
        cols = [' / '.join(f'{pr(c, q, k, s):.0f}' for k in ['iabc', 'sc_tun', 'sc_t0_i4', 'oracle']) for s in ['0', '10', '15', '25']]
        a = e[f'{q}_iabc']['pct_removed_mse']; t = e[f'{q}_sc_tun']['pct_removed_mse']
        print(f'| {c} | {"yes" if cells[c]["in_train_range"] else "no"} | {b:.2f} | ' + ' | '.join(cols) +
              f' | {100 * a[0]:.0f}% [{100 * a[1]:.0f},{100 * a[2]:.0f}] | {100 * t[0]:.0f}% [{100 * t[1]:.0f},{100 * t[2]:.0f}] |')

print('\n## B. % of identifiable MSE removed, pooled over scenes (exact, from per-scene caches cache/sweep__*.npz)')
print('gated oracle = per SNR group, the better of {no correction, oracle ALS with true angles} (a bound for an SNR-aware ALS-type estimator)')
for c in ['ph5', 'ph8', 'ph12', 'ph20', 'ph30', 'ph45', 'g2', 'g3', 'g4', 'g6', 'c8_3', 'c20_4', 'c30_6']:
    z = np.load(f'{HERE}/cache/sweep__{c}.npz'); snr = z['snr']
    for q in ['ph', 'g']:
        bef = z[f'{q}_before'] ** 2
        if bef.mean() < 1e-6:
            continue
        f = lambda k, m=slice(None): 100 * (1 - (z[f'{q}_{k}'][m] ** 2).mean() / bef[m].mean())
        go = sum(min(bef[snr == s].sum(), (z[f'{q}_oracle'][snr == s] ** 2).sum()) for s in np.unique(snr))
        hi = snr >= 10
        print(f'{c:6s} {"phase" if q == "ph" else "gain ":5s} all SNR: IABC {f("iabc"):6.1f}% | SC tuned {f("sc_tun"):7.1f}% | IABC+SC tuned {f("iabc_sct"):7.1f}% | '
              f'gated oracle {100 * (1 - go / bef.sum()):5.1f}%  || SNR>=10: IABC {f("iabc", hi):5.1f}% | SC tuned {f("sc_tun", hi):5.1f}% | '
              f'IABC+SC tuned {f("iabc_sct", hi):5.1f}% | SC open 4it {f("sc_t0_i4", hi):5.1f}% | oracle {f("oracle", hi):5.1f}%')

print('\n## C. Pd (paper, mean over SNR -10..25, L=3, 1600 scenes) and paired differences [95% CI]')
M = ['NOMP', 'IABC+NOMP', 'IABR-Net heat', 'IABR-v2 dual', 'NOMP+SC default', 'NOMP+SC tuned', 'IABC+NOMP+SC tuned']
D = ['IABC+NOMP - NOMP', 'IABR-Net heat - NOMP', 'IABC+NOMP - NOMP+SC tuned', 'IABC+NOMP+SC tuned - NOMP+SC tuned', 'IABR-Net heat - NOMP+SC tuned']
print('| cell | ' + ' | '.join(M) + ' |'); print('|' + '---|' * (len(M) + 1))
for c in cells:
    print(f'| {c} | ' + ' | '.join(f'{cells[c]["pd"][m]["mean"]["pd_paper"]:.3f}' for m in M) + ' |')
st = lambda v: f'{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]' + ('*' if v[1] > 0 or v[2] < 0 else '')
print('\n| cell | ' + ' | '.join(D) + ' |'); print('|' + '---|' * (len(D) + 1))
for c in cells:
    print(f'| {c} | ' + ' | '.join(st(cells[c]['pd_diffs'][d]['pd_paper']) for d in D) + ' |')
print('\n## D. Per-SNR IABC+NOMP - NOMP and IABR-Net heat - NOMP (paper Pd) for large severities')
for c in ['g6', 'c30_6', 'c20_4', 'ph45', 'ph30']:
    for d in ['IABC+NOMP - NOMP', 'IABR-Net heat - NOMP', 'IABC+NOMP+SC tuned - NOMP+SC tuned']:
        p = cells[c]['pd_diffs'][d]['per_snr_pd_paper']
        print(f'{c:6s} {d:36s} ' + ' '.join(f'{s}:{st(p[s])}' for s in SNRS))
cnt = {}
for n, r in R['cached_banks'].items():
    cnt[r['verdict']] = cnt.get(r['verdict'], 0) + 1
print('\n## E. cached banks: IABC+NOMP - NOMP (paper Pd) verdicts:', cnt)
for n in ['official', 'T2_d1', 'T2_d2', 'T2_d5', 'gain_g4_snr15', 'sep_3deg_snr15', 'tail_snr35']:
    r = R['cached_banks'][n]; d = r['diff_pd_paper']
    print(f'{n:16s} N={r["n"]:5d} NOMP {r["pd_nomp"]:.4f} IABC+NOMP {r["pd_iabc_nomp"]:.4f} diff {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] {r["verdict"]}')
