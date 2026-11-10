"""Supplement to analyze.py: pooled official-subset + val (4000 scenes, 500/SNR, no heat-map families so both pools are identical
in construction). Legal selection rules vs NOMP os4rc3 AND vs the best classical setting NOMP os8rc6; task-4 low-SNR
(-10/-5/0 dB) posterior-mean / voting rules with per-SNR paired-bootstrap CIs. stdout -> supp.log, json -> supp.json"""
import json
from analyze import *

pools = {b: load_pool(b) for b in ['official', 'val']}
fam_off = pools['official']['fam']
legal_nh = np.where(~np.isin(fam_off, ['oracle_ref', 'truth_ref', 'heat', 'heatcomb']))[0]
legal_v = np.where(~np.isin(pools['val']['fam'], ['oracle_ref', 'truth_ref']))[0]
assert list(fam_off[legal_nh]) == list(pools['val']['fam'][legal_v])
NCo = evaluate(pools['official'], legal_nh, None); NCv = evaluate(pools['val'], legal_v, None)
NC = {r: np.concatenate([NCo[r], NCv[r]]) for r in NCo if not r.startswith('changed__')}
snr = np.concatenate([pools['official']['snr'], pools['val']['snr']])
NC['nomp_os8rc6'] = np.concatenate([pools['official']['ncorrect'][:, legal_nh[1]], pools['val']['ncorrect'][:, legal_v[1]]])
assert fam_off[legal_nh[1]] == 'nomp_os8rc6'
out = {}
print(f'pooled official subset + val: {len(snr)} scenes, {len(legal_nh)} legal candidates')
print('rule              | mean   | per SNR -10..25                                           | vs NOMP os4rc3          | vs NOMP os8rc6')
for r in ['nomp', 'nomp_os8rc6', 'ml', 'ml_sep', 'mll_sep', 'mapU_sep', 'mass_nosep', 'mass_sep', 'mbr', 'postmean', 'ml_sep_ro',
          'mass_sep_ro', 'mbr_ro', 'oracle', 'oracle_nomp+swap', 'oracle_nomp+fftcomb', 'oracle_nomp+branch', 'oracle_nomp+random',
          'oracle_seeded_ref']:
    nc = NC[r]; per = pd_snr(nc, snr); d0 = boot_diff(nc, NC['nomp'], snr); d1 = boot_diff(nc, NC['nomp_os8rc6'], snr)
    st = lambda d: f'{d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]{"*" if d[1] > 0 or d[2] < 0 else " "}'
    print(f'{r:17s} | {per["mean"]:.4f} | ' + ' '.join(f'{per[s]:.4f}' for s in sorted(k for k in per if k != 'mean')) + f' | {st(d0)} | {st(d1)}')
    out[r] = dict(pd=per, diff_vs_nomp=d0, diff_vs_os8rc6=d1)
print('\nTask 4, low SNR (pooled 500 scenes per SNR): diff vs NOMP os4rc3 [95% CI]')
out['lowsnr'] = {}
for s in [-10, -5, 0]:
    m = snr.astype(int) == s; row = {}
    for r in ['ml', 'mass_sep', 'mass_nosep', 'mbr', 'postmean', 'oracle']:
        d = boot_diff(NC[r][m], NC['nomp'][m], snr[m]); row[r] = d
    out['lowsnr'][s] = row
    print(f'{s:4d} dB | NOMP {NC["nomp"][m].mean()/6:.4f} | ' + ' | '.join(f'{r} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]' for r, d in row.items()))
json.dump(out, open(f'{HERE}/supp.json', 'w'), indent=1)
