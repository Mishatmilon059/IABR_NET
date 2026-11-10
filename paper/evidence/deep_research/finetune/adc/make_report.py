"""Collects res_*.json, diag*.json, latency.json, train_hist.json -> results.json, fig_adc.png, fig_train.png and the tables used in
REPORT.md (stdout -> report.log). No new computation except reading the logged results."""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(f'{HERE}/{f}')) if os.path.exists(f'{HERE}/{f}') else None
R = {s: (J(f'res2_{s}.json') or J(f'res_{s}.json')) for s in ['m5', 'fresh', 'clean', 'official']}   # res2 = res + supp.py methods
out = dict(results=R, diag=J('diag.json'), diag2=J('diag2.json'), latency=J('latency.json'), train=J('train_hist.json'))
json.dump(out, open(f'{HERE}/results.json', 'w'), indent=1, default=float)

CLS = ['NOMP', 'NOMP+SC', 'NOMP+SC tuned', 'FFT-peak seeds', 'FFT-peak seeds+SC', 'classical dual', 'classical dual+SC', 'NOMP os8 rc6',
       'multi-start K6', 'consistent NOMP', 'FFT parabolic os4', 'FFT parabolic os16', 'consistent parabolic']
NET = ['orig heat+SC', 'orig dual+SC', 'orig network only', 'FT heat+SC', 'FT dual+SC', 'FT NOMP on Hc', 'FT network only', 'FT heat + consistent']


def table(setname):
    r = R[setname]
    if r is None:
        return
    names = list(r['banks'])
    print(f'\n### {setname}: Pd (paper-style, mean over SNR groups) per method; best classical per cell in brackets\n')
    print('| method | ' + ' | '.join(names) + ' |'); print('|---|' + '---|' * len(names))
    for m in CLS + NET + ['oracle-seeded']:
        row = []
        for n in names:
            b = r['banks'][n]
            if m not in b['pd']:
                row.append('-'); continue
            s = f'{b["pd"][m]:.3f}'
            if m == b['best_classical']:
                s = f'**{s}**'
            row.append(s)
        print(f'| {m} | ' + ' | '.join(row) + ' |')
    print(f'\n{setname}: network methods minus the BEST classical method of the cell (paired bootstrap 95% CI; * = CI excludes 0)\n')
    print('| method | ' + ' | '.join(names) + ' |'); print('|---|' + '---|' * len(names))
    for m in NET + ['oracle-seeded']:
        row = []
        for n in names:
            d = r['banks'][n]['diffs'].get(f'{m} - BEST')
            if d is None:
                row.append('-'); continue
            star = '*' if (d[1] > 0 or d[2] < 0) else ''
            row.append(f'{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]{star}')
        print(f'| {m} | ' + ' | '.join(row) + ' |')
    print('\nbest classical per cell: ' + ', '.join(f'{n}: {r["banks"][n]["best_classical"]}' for n in names))
    for n in names:
        d = r['banks'][n]['diffs']
        print(f'{n}: consistent NOMP - NOMP ' + (str(np.round(d['consistent NOMP - NOMP'], 4).tolist()) if 'consistent NOMP - NOMP' in d else '-') +
              ' | FT dual+SC - orig dual+SC ' + str(np.round(d.get('FT dual+SC - orig dual+SC', [np.nan] * 3), 4).tolist()) +
              ' | FT heat+SC - orig heat+SC ' + str(np.round(d.get('FT heat+SC - orig heat+SC', [np.nan] * 3), 4).tolist()))


for s in ['m5', 'fresh', 'clean', 'official']:
    table(s)
if R['official'] is not None:
    b = R['official']['banks']['official']
    print('\nofficial by SNR (paper Pd):')
    snrs = [k for k in b['by_snr']['NOMP'] if k != 'mean']
    print('| method | ' + ' | '.join(snrs) + ' | mean |'); print('|---|' + '---|' * (len(snrs) + 1))
    for m in b['by_snr']:
        print(f'| {m} | ' + ' | '.join(f'{b["by_snr"][m][k]:.3f}' for k in snrs) + f' | {b["by_snr"][m]["mean"]:.4f} |')
print('\n### headline per cell: best classical (extended set) vs fine-tuned network-only and best FT decoder variant (paired 95% CI)\n')
print('| set | cell | best classical | Pd | FT network only | diff [CI] | orig network only | best FT+decoder | diff [CI] | oracle-seeded |')
print('|---|---|---|---|---|---|---|---|---|---|')
for st in ['m5', 'fresh']:
    if R[st] is None:
        continue
    for n, b in R[st]['banks'].items():
        bc = b['best_classical']; d = b['diffs']['FT network only - BEST']
        fd = max(['FT heat+SC', 'FT dual+SC', 'FT NOMP on Hc', 'FT heat + consistent'], key=lambda m: b['pd'][m]); d2 = b['diffs'][f'{fd} - BEST']
        f = lambda x: f'{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]' + ('*' if x[1] > 0 or x[2] < 0 else '')
        print(f'| {st} | {n} | {bc} | {b["pd"][bc]:.3f} | {b["pd"]["FT network only"]:.3f} | {f(d)} | {b["pd"].get("orig network only", float("nan")):.3f} | '
              f'{fd} {b["pd"][fd]:.3f} | {f(d2)} | {b["pd"]["oracle-seeded"]:.3f} |')
for st in ['official', 'clean']:
    if R[st] is None:
        continue
    for n, b in R[st]['banks'].items():
        d = b['diffs']
        print(f'{n}: FT dual+SC - orig dual+SC {np.round(d["FT dual+SC - orig dual+SC"], 4).tolist()}, FT heat+SC - orig heat+SC '
              f'{np.round(d["FT heat+SC - orig heat+SC"], 4).tolist()}, FT dual+SC - NOMP {np.round(d["FT dual+SC - NOMP"], 4).tolist()}, '
              f'FT network only - orig network only {np.round(d.get("FT network only - orig network only", [np.nan] * 3), 4).tolist()}')
if out['latency']:
    print('\nlatency (ms/scene, 1 thread, batch 1):', {k: round(v, 2) for k, v in out['latency']['ms'].items()})

# ---------------- figure: Pd vs ADC bits
if R['m5'] is not None:
    PLOT = [('NOMP', 'k', '-', 'o'), ('NOMP+SC tuned', '0.45', '--', 'o'), ('classical dual', '#8c564b', '-', 's'), ('NOMP os8 rc6', '#7f7f7f', ':', 's'),
            ('multi-start K6', '#bcbd22', '-', '^'), ('consistent NOMP', '#1f77b4', '-', 'D'), ('FFT parabolic os16', '#17becf', ':', 'd'), ('orig network only', '#c49c94', ':', 'x'), ('orig dual+SC', '#ff7f0e', '--', 'v'),
            ('FT heat+SC', '#d62728', '-', 'v'), ('FT dual+SC', '#9467bd', '-', 'P'), ('FT network only', '#e377c2', ':', 'x'),
            ('FT heat + consistent', '#2ca02c', '-', '*'), ('oracle-seeded', 'k', (0, (1, 2)), '')]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, snr in zip(axs, [15, 0]):
        xs = [1, 2, 3, 4, 5]
        for m, c, ls, mk in PLOT:
            ys = [R['m5']['banks'][f'b{bb}_snr{snr}']['pd'].get(m, np.nan) for bb in [1, 2, 3, 4]]
            cl = R['clean']['banks'][f'clean_snr{snr}']['pd'].get(m, np.nan) if R['clean'] else np.nan
            ax.plot(xs, ys + [cl], color=c, ls=ls, marker=mk, ms=5, lw=1.4, label=m)
        ax.set_xticks(xs); ax.set_xticklabels(['1', '2', '3', '4', 'no ADC']); ax.set_xlabel('ADC bits (peak AGC)')
        ax.set_title(f'SNR {snr} dB (stress M5 banks, seeds 5000/5015, 500 scenes, L=3)'); ax.grid(alpha=0.3)
    axs[0].set_ylabel('Pd (|err| <= 1 deg)')
    axs[1].legend(fontsize=7.5, loc='lower right', ncol=2)
    fig.tight_layout(); fig.savefig(f'{HERE}/fig_adc.png', dpi=150)
if out['train']:
    h = [x for x in out['train']['hist'] if 'loss' in x]
    st = [x['step'] for x in h]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(st, [x['loss'] for x in h], label='train total'); ax[0].plot(st, [x['heat'] for x in h], label='train heat (focal)')
    ax[0].set_xlabel('step'); ax[0].legend(); ax[0].grid(alpha=.3); ax[0].set_title('training loss (50-step means)')
    m0 = out['train']['hist'][0]['mon']
    ax[1].plot([0] + st, [m0['orig'][0]] + [x['mon']['orig'][0] for x in h], label='monitor heat, original distribution')
    ax[1].plot([0] + st, [m0['adc'][0]] + [x['mon']['adc'][0] for x in h], label='monitor heat, ADC half')
    ax[1].set_xlabel('step'); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].set_title('fixed monitoring batch (not used for selection)')
    fig.tight_layout(); fig.savefig(f'{HERE}/fig_train.png', dpi=130)
print('\nwrote results.json, fig_adc.png, fig_train.png')
