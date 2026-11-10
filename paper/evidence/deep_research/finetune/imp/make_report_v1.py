"""Collects parts/*.json (evaluate.py), m4_physics.json, train_hist.json -> results.json, fig_<family>.png and the tables printed
here (stdout -> make_report.log). No new metric computation: every number is read from the per-bank json written by the runs."""
import json, os, glob, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
P = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f'{HERE}/parts/*.json')) if not p.endswith('_smoke.json')}
M4 = json.load(open(f'{HERE}/m4_physics.json')) if os.path.exists(f'{HERE}/m4_physics.json') else None
LEARNED = ['FT heat+SC', 'FT dual+SC', 'FT network only', 'FT nocorr heat+SC', 'FT nocorr dual+SC', 'FT nocorr network only',
           'FT v2b heat+SC', 'FT v2b dual+SC', 'FT v2b network only', 'FT seeds on Ht+SC', 'FT NOMP on Hc+SC']
KEEP_REF = ('BEST', 'NOMP', 'NOMP+SC tuned', 'EB-ML-g5', 'orig dual+SC', 'NOMP+SC', 'FT heat+SC', 'FT seeds on Ht+SC', 'FT NOMP on Hc+SC',
            'FFT parabolic os16', 'orig network only', 'orig heat+SC', 'v2b+NOMP+SC tuned', 'FT dual+SC')


def f3(d):
    return f'{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]' + ('*' if d[1] > 0 or d[2] < 0 else '')


def bank_rows(prefix):
    rows = []
    for name, r in P.items():
        m = re.match(rf'{prefix}_(M\d)_(.+)_snr(\d+)$', name)
        if m:
            rows.append((m.group(1), m.group(2), int(m.group(3)), name, r))
    return rows


out = dict(banks={})
for k, v in P.items():
    e = dict(n=v['n'], pd=v['pd'], pd_strict=v['pd_strict'], best_classical=v['best_classical'],
             diffs={d: x for d, x in v['diffs'].items() if d.split(' - ')[1] in KEEP_REF})
    if 'Y_check_nomp_maxdiff_vs_stress_cache' in v:
        e['Y_check'] = v['Y_check_nomp_maxdiff_vs_stress_cache']
    if 'by_snr' in v:
        e['by_snr'] = v['by_snr']
    out['banks'][k] = e
if M4:
    out['m4_physics'] = M4
th = None
if os.path.exists(f'{HERE}/train_hist.json'):
    th = json.load(open(f'{HERE}/train_hist.json'))
    out['training'] = {k: v for k, v in th.items() if k != 'hist'}
    out['training']['monitor_first_last'] = [th['hist'][0]['mon'], th['hist'][-1]['mon']]

for setname in ['stress', 'fresh']:
    rows = bank_rows(setname)
    if not rows:
        continue
    print(f'\n################ {setname} banks: Pd (paper); diffs paired 95% CI, * = CI excludes 0')
    for fam in sorted({r[0] for r in rows}):
        print(f'\n== {fam} ({setname})')
        for _, sev, snr, name, r in sorted([x for x in rows if x[0] == fam], key=lambda x: (-x[2], x[1])):
            pd = r['pd']; D = r['diffs']; best = r['best_classical']
            bl = max(LEARNED, key=lambda m: pd[m])
            yc = r.get('Y_check_nomp_maxdiff_vs_stress_cache')
            print(f'  {sev:>9s} snr{snr:<2d} NOMP {pd["NOMP"]:.3f} | tunedSC {pd["NOMP+SC tuned"]:.3f} | EB-ML {pd["EB-ML-g5"]:.3f} | best cl {best} {pd[best]:.3f}'
                  f' | v2b+SCt {pd["v2b+NOMP+SC tuned"]:.3f} | orig dual {pd["orig dual+SC"]:.3f} | oracle {pd["oracle-seeded"]:.3f}'
                  + (f' | Ycheck {yc:.0e}' if yc is not None else ''))
            for m in ['FT heat+SC', 'FT dual+SC', 'FT network only', 'FT nocorr dual+SC', 'FT v2b dual+SC']:
                print(f'      {m:20s} {pd[m]:.3f}  vs best {f3(D[m + " - BEST"])}  vs NOMP {f3(D[m + " - NOMP"])}  vs orig dual {f3(D[m + " - orig dual+SC"])}')
            print(f'      best FT variant: {bl} {pd[bl]:.3f} vs best classical {f3(D[bl + " - BEST"])}')
            print(f'      ablation (a) FT seeds on Ht - NOMP+SC {f3(D["FT seeds on Ht+SC - NOMP+SC"])} | (b) FT NOMP on Hc - NOMP+SC {f3(D["FT NOMP on Hc+SC - NOMP+SC"])}'
                  f' | (c)-(a) {f3(D["FT heat+SC - FT seeds on Ht+SC"])} | (c)-(b) {f3(D["FT heat+SC - FT NOMP on Hc+SC"])}')

for name in ['clean_snr15', 'clean_snr0', 'fresh_clean_clean_snr15', 'fresh_clean_clean_snr0', 'official']:
    if name in P:
        r = P[name]; pd = r['pd']; D = r['diffs']
        print(f'\n== {name} (n={r["n"]}) best classical {r["best_classical"]} {pd[r["best_classical"]]:.4f}')
        for m in pd:
            s = f'   {m:24s} {pd[m]:.4f}'
            for ref in ['NOMP', 'orig dual+SC']:
                if f'{m} - {ref}' in D:
                    d = D[f'{m} - {ref}']; s += f'  vs {ref} {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]'
            print(s)
        if 'by_snr' in r:
            print('   by SNR:', {m: {k: round(v, 3) for k, v in r['by_snr'][m].items()} for m in ['NOMP', 'orig dual+SC', 'FT dual+SC', 'FT heat+SC']})

if M4:
    print('\n################ M4 physics control (known quantised codebooks)')
    print('conditioning (rank, cond):', {k: (v['rank'], round(v['cond'], 3) if v['cond'] != float('inf') else 'inf') for k, v in M4['conditioning'].items()})
    for name, r in M4['banks'].items():
        pd = r['pd']; D = r['diffs']; g = r['gap_closed']
        print(f'  {name:22s} NOMP {pd["NOMP"]:.3f} tSC {pd["NOMP+SC tuned"]:.3f} LS {pd["LS-NOMP"]:.3f} MD {pd["MD-NOMP"]:.3f} orac {pd["oracle-seeded"]:.3f}'
              f' oracMD {pd["oracle-seeded MD"]:.3f} clean {pd["clean NOMP"]:.3f} | MD-NOMP {f3(D["MD-NOMP - NOMP"])} | clean-MD {f3(D["clean NOMP - MD-NOMP"])}'
              + (f' | gap closed MD {100 * g["MD-NOMP"]:.0f}% LS {100 * g["LS-NOMP"]:.0f}%' if r['bits'] < 4 else ''))

SEVORDER = {'M3': ['ph20_g0', 'ph30_g0', 'ph45_g0', 'ph0_g6', 'ph30_g4']}   # others: numeric order of increasing severity (M8: decreasing range)


def sev_key(fam, l):
    return -float(l) if fam == 'M8' else float(l)
FIGM = ['NOMP', 'NOMP+SC tuned', 'EB-ML-g5', 'orig dual+SC', 'FT dual+SC', 'FT heat+SC', 'FT network only', 'FT v2b dual+SC', 'oracle-seeded']
STY = {'NOMP': ('k', '-'), 'NOMP+SC tuned': ('0.5', '-'), 'EB-ML-g5': ('tab:brown', '-'), 'orig dual+SC': ('tab:blue', '--'),
       'FT dual+SC': ('tab:red', '-'), 'FT heat+SC': ('tab:orange', '-'), 'FT network only': ('tab:purple', ':'),
       'FT v2b dual+SC': ('tab:green', '-.'), 'oracle-seeded': ('0.7', ':')}
rows = bank_rows('stress')
for fam in sorted({r[0] for r in rows}):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, snr in zip(axs, [15, 0]):
        rr = [x for x in rows if x[0] == fam and x[2] == snr]
        labs = [l for l in SEVORDER.get(fam, sorted({x[1] for x in rr}, key=lambda l: sev_key(fam, l))) if any(x[1] == l for x in rr)]
        cl = f'clean_snr{snr}'; xl = ['clean'] + labs
        for m in FIGM:
            ys = [P[cl]['pd'][m] if cl in P else np.nan] + [next(x[4]['pd'][m] for x in rr if x[1] == l) for l in labs]
            ax.plot(range(len(xl)), ys, marker='o', ms=3, color=STY[m][0], ls=STY[m][1], label=m)
        bc = [P[cl]['pd'][P[cl]['best_classical']] if cl in P else np.nan] + [next(x[4]['pd'][x[4]['best_classical']] for x in rr if x[1] == l) for l in labs]
        ax.plot(range(len(xl)), bc, color='k', lw=5, alpha=0.15, label='best classical (per bank)')
        ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, rotation=20)
        ax.set_title(f'{fam}, SNR {snr} dB (stress seeds, 500 scenes)'); ax.grid(alpha=0.3)
    axs[0].set_ylabel('Pd (paper, |err| <= 1 deg)'); axs[1].legend(fontsize=7, loc='lower left')
    fig.tight_layout(); fig.savefig(f'{HERE}/fig_{fam}.png', dpi=110); plt.close(fig)
if M4:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, snr in zip(axs, [15, 0]):
        for m, c in [('NOMP', 'k'), ('NOMP+SC tuned', '0.5'), ('LS-NOMP', 'tab:blue'), ('MD-NOMP', 'tab:red'), ('oracle-seeded', '0.7'), ('clean NOMP', 'tab:green')]:
            ax.plot([4, 3, 2, 1], [M4['banks'][f'stress_M4_b{b}_snr{snr}']['pd'][m] for b in [4, 3, 2, 1]], marker='o', color=c, label=m)
        ax.set_xticks([4, 3, 2, 1]); ax.invert_xaxis(); ax.set_xlabel('phase-shifter bits'); ax.set_title(f'M4, SNR {snr} dB (stress seeds)'); ax.grid(alpha=0.3)
    axs[0].set_ylabel('Pd (paper)'); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f'{HERE}/fig_M4.png', dpi=110); plt.close(fig)
if th:
    h = [x for x in th['hist'] if 'mon' in x]
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in h[0]['mon']:
        ax.plot([x['step'] for x in h], [x['mon'][k][0] for x in h], label=k)
    ax.set_xlabel('step'); ax.set_ylabel('focal heat loss, fixed monitor batch'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{HERE}/fig_train.png', dpi=110); plt.close(fig)
json.dump(out, open(f'{HERE}/results.json', 'w'), indent=1)
print('\nwrote results.json and figures')
