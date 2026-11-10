"""Step 4 tables (stdout -> final_tables.log). Reads parts/*.json only (written by evaluate.py); no new metric computation.
Replication protocol (fixed here, before reading the fresh results): a 'candidate win' is a (cell, learned FT variant) whose paired 95% CI
vs the best classical method of that stress bank excludes 0 on the STRESS seeds (variant picked post hoc among 11 learned variants).
It is CLAIMED only if the SAME variant also beats the best classical method of the FRESH bank (picked on the fresh bank itself) with a CI
excluding 0 on the fresh seeds (8400/8415, 1000 scenes). Pre-specified main pipelines: FT dual+SC, FT heat+SC, FT network only."""
import json, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
P = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f'{HERE}/parts/*.json')) if not p.endswith('_smoke.json')}
LEARNED = ['FT heat+SC', 'FT dual+SC', 'FT network only', 'FT nocorr heat+SC', 'FT nocorr dual+SC', 'FT nocorr network only',
           'FT v2b heat+SC', 'FT v2b dual+SC', 'FT v2b network only', 'FT seeds on Ht+SC', 'FT NOMP on Hc+SC']
ORDER = {'M1': ['0.05', '0.1', '0.2', '0.3'], 'M2': ['0.02', '0.05', '0.1', '0.2'], 'M3': ['ph20_g0', 'ph30_g0', 'ph45_g0', 'ph0_g6', 'ph30_g4'],
         'M6': ['5', '10', '20', '30'], 'M8': ['2', '1', '0.5']}


def f(d):
    return f'{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]' + ('*' if d[1] > 0 or d[2] < 0 else '')


def cells(pre):
    for fam, sevs in ORDER.items():
        for snr in [15, 0]:
            for s in sevs:
                n = f'{pre}_{fam}_{s}_snr{snr}'
                if n in P:
                    yield fam, s, snr, n


print('#### A. stress banks (500 scenes, seeds 5015/5000, Y identical to stress run): Pd and paired diff vs best classical (picked on the bank)')
print('| fam | sev | SNR | NOMP | tuned SC | EB-ML-g5 | best classical | orig dual+SC | FT dual+SC | FT heat+SC | FT net only | oracle | FT dual+SC - best | FT heat+SC - best | FT dual+SC - orig dual+SC |')
print('|' + '---|' * 15)
for fam, s, snr, n in cells('stress'):
    r = P[n]; pd = r['pd']; D = r['diffs']; b = r['best_classical']
    print(f'| {fam} | {s} | {snr} | {pd["NOMP"]:.3f} | {pd["NOMP+SC tuned"]:.3f} | {pd["EB-ML-g5"]:.3f} | {pd[b]:.3f} ({b}) | {pd["orig dual+SC"]:.3f} | {pd["FT dual+SC"]:.3f} | '
          f'{pd["FT heat+SC"]:.3f} | {pd["FT network only"]:.3f} | {pd["oracle-seeded"]:.3f} | {f(D["FT dual+SC - BEST"])} | {f(D["FT heat+SC - BEST"])} | {f(D["FT dual+SC - orig dual+SC"])} |')

print('\n#### B. candidate wins on stress seeds (any FT variant, CI vs best classical > 0) and their fresh-seed replication (same variant)')
print('| cell | variant | stress Pd | stress diff vs best | fresh Pd | fresh best classical | fresh diff vs best | fresh diff vs NOMP | fresh diff vs v2b+NOMP+SC tuned | verdict |')
print('|' + '---|' * 10)
claims = []
for fam, s, snr, n in cells('stress'):
    r = P[n]; D = r['diffs']
    for m in LEARNED:
        d = D[f'{m} - BEST']
        if d[1] > 0:
            fn = f'fresh_{fam}_{s}_snr{snr}'
            if fn in P:
                q = P[fn]; dq = q['diffs'][f'{m} - BEST']; dn = q['diffs'][f'{m} - NOMP']; dv = q['diffs'][f'{m} - v2b+NOMP+SC tuned']
                v = 'REPLICATED' if dq[1] > 0 else ('not replicated (CI incl. 0)' if dq[2] >= 0 else 'REVERSED')
                if dq[1] > 0:
                    claims.append((fam, s, snr, m, [round(x, 4) for x in dq]))
                print(f'| {fam} {s} {snr} dB | {m} | {r["pd"][m]:.3f} | {f(d)} | {q["pd"][m]:.3f} | {q["pd"][q["best_classical"]]:.3f} ({q["best_classical"]}) | {f(dq)} | {f(dn)} | {f(dv)} | {v} |')
            else:
                print(f'| {fam} {s} {snr} dB | {m} | {r["pd"][m]:.3f} | {f(d)} | - | - | - | - | - | fresh bank missing |')

print('\n#### C. fresh banks (1000 scenes, seeds 8415/8400): pre-specified pipelines vs best classical (picked on the fresh bank)')
print('| fam | sev | SNR | NOMP | best classical | orig dual+SC | FT dual+SC | FT heat+SC | FT net only | FT v2b dual+SC | FT v2b heat+SC | FT dual+SC - best | FT heat+SC - best | FT v2b heat+SC - best |')
print('|' + '---|' * 14)
for fam, s, snr, n in cells('fresh'):
    r = P[n]; pd = r['pd']; D = r['diffs']; b = r['best_classical']
    print(f'| {fam} | {s} | {snr} | {pd["NOMP"]:.3f} | {pd[b]:.3f} ({b}) | {pd["orig dual+SC"]:.3f} | {pd["FT dual+SC"]:.3f} | {pd["FT heat+SC"]:.3f} | {pd["FT network only"]:.3f} | '
          f'{pd["FT v2b dual+SC"]:.3f} | {pd["FT v2b heat+SC"]:.3f} | {f(D["FT dual+SC - BEST"])} | {f(D["FT heat+SC - BEST"])} | {f(D["FT v2b heat+SC - BEST"])} |')

print('\n#### D. ablation (heat seeds + SC): (a) FT seeds on raw Ht, (b) greedy NOMP on FT-corrected Hc, (c) FT seeds on FT Hc; ref NOMP+SC')
print('| bank | (a) - NOMP+SC | (b) - NOMP+SC | (c) - (a) [value of learned correction] | (c) - (b) [value of learned seeds] |')
print('|---|---|---|---|---|')
for pre in ['stress', 'fresh']:
    for fam, s, snr, n in cells(pre):
        D = P[n]['diffs']
        print(f'| {pre} {fam} {s} {snr} dB | {f(D["FT seeds on Ht+SC - NOMP+SC"])} | {f(D["FT NOMP on Hc+SC - NOMP+SC"])} | {f(D["FT heat+SC - FT seeds on Ht+SC"])} | {f(D["FT heat+SC - FT NOMP on Hc+SC"])} |')
for n in ['clean_snr15', 'clean_snr0', 'fresh_clean_clean_snr15', 'fresh_clean_clean_snr0', 'official']:
    if n in P:
        D = P[n]['diffs']
        print(f'| {n} | {f(D["FT seeds on Ht+SC - NOMP+SC"])} | {f(D["FT NOMP on Hc+SC - NOMP+SC"])} | {f(D["FT heat+SC - FT seeds on Ht+SC"])} | {f(D["FT heat+SC - FT NOMP on Hc+SC"])} |')

print('\n#### E. ablation / correction-variant counts (cells with CI excluding 0: positive / negative)')
for pre in ['stress', 'fresh']:
    for k in ['FT seeds on Ht+SC - NOMP+SC', 'FT NOMP on Hc+SC - NOMP+SC', 'FT heat+SC - FT seeds on Ht+SC', 'FT heat+SC - FT NOMP on Hc+SC',
              'FT v2b heat+SC - FT heat+SC', 'FT nocorr heat+SC - FT heat+SC', 'FT v2b dual+SC - FT dual+SC', 'FT nocorr dual+SC - FT dual+SC',
              'FT dual+SC - orig dual+SC', 'FT heat+SC - orig heat+SC', 'FT network only - orig network only']:
        ds = [(fam, s, snr, P[n]['diffs'][k]) for fam, s, snr, n in cells(pre)]
        pos = [f'{a}/{b}/{c}' for a, b, c, d in ds if d[1] > 0]; neg = [f'{a}/{b}/{c}' for a, b, c, d in ds if d[2] < 0]
        print(f'{pre:6s} {k:36s} n={len(ds):2d}  +{len(pos)} {pos}  -{len(neg)} {neg}')

print('\n#### F. correction variants (FT trunk): own fine-tuned IABC vs IABC-v2b vs none, on clean / official')
for n in ['clean_snr15', 'clean_snr0', 'fresh_clean_clean_snr15', 'fresh_clean_clean_snr0', 'official']:
    if n in P:
        r = P[n]; pd = r['pd']; D = r['diffs']
        print(f'{n:24s} NOMP {pd["NOMP"]:.4f} | orig dual+SC {pd["orig dual+SC"]:.4f} | FT dual+SC {pd["FT dual+SC"]:.4f} {f(D["FT dual+SC - NOMP"])} vs orig {f(D["FT dual+SC - orig dual+SC"])}'
              f' | FT v2b dual+SC {pd["FT v2b dual+SC"]:.4f} {f(D["FT v2b dual+SC - NOMP"])} | FT nocorr dual+SC {pd["FT nocorr dual+SC"]:.4f} {f(D["FT nocorr dual+SC - NOMP"])}'
              f' | FT heat+SC {pd["FT heat+SC"]:.4f} {f(D["FT heat+SC - NOMP"])} vs orig heat {f(D["FT heat+SC - orig heat+SC"])} | FT net only {pd["FT network only"]:.4f} vs orig {f(D["FT network only - orig network only"])}')
if 'official' in P:
    bs = P['official']['by_snr']
    print('official by SNR:', {m: {k: round(v, 3) for k, v in bs[m].items()} for m in ['NOMP', 'orig dual+SC', 'FT dual+SC', 'FT heat+SC', 'FT nocorr dual+SC', 'FT v2b dual+SC']})
print('\nCLAIMED (replicated) wins:', claims)
