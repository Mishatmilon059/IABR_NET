"""Tables, paired bootstraps, figures and results.json from the per-scene sweep results (cache/sweep__*.npz written by s2_sweep.py).
Bootstrap: 2000 resamples within SNR groups, seed 0 (all methods/estimators share the same resample indices -> paired).
  estimation: '% removed (MSE)' = 1 - mean(after_rms^2)/mean(before_rms^2) over the resampled scenes (equal-size SNR groups, so this is
              the mean over SNR); '% removed (RMS)' = 1 - mean(after_rms)/mean(before_rms).
  Pd: common.boot_metrics (mean over SNR of paper Pd / strict Pd / RMSE).
usage: python s3_analyze.py > s3_analyze.log  (writes results.json, tables.md, fig_*.png)"""
from icommon import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['clean', 'ph2', 'ph5', 'ph8', 'ph12', 'ph20', 'ph30', 'ph45', 'g1', 'g2', 'g3', 'g4', 'g6', 'c5_2', 'c8_3', 'c20_4', 'c30_6']
R = {n: dict(np.load(f'{CACHE}/sweep__{n}.npz')) for n in ORDER if os.path.exists(f'{CACHE}/sweep__{n}.npz')}
ORDER = [n for n in ORDER if n in R]
EST = ['before', 'iabc', 'sc_t0_i2', 'sc_t0_i4', 'sc_def', 'sc_tun', 'oracle', 'iabc_sct']
ESTN = {'before': 'before (NOMP faces)', 'iabc': 'after IABC', 'sc_t0_i2': 'self-cal tau0 2it', 'sc_t0_i4': 'self-cal tau0 4it',
        'sc_def': 'self-cal (1,5,2)', 'sc_tun': 'self-cal tuned (1,2,4)', 'oracle': 'oracle ALS (true angles)', 'iabc_sct': 'IABC + tuned self-cal'}
METH = ['NOMP', 'IABC+NOMP', 'IABR-Net heat', 'IABR-v2 dual', 'NOMP+SC default', 'NOMP+SC tuned', 'NOMP+SC tau0 i2', 'NOMP+SC tau0 i4', 'IABC+NOMP+SC tuned']
DIFFS = [('IABC+NOMP', 'NOMP'), ('IABR-Net heat', 'NOMP'), ('IABR-v2 dual', 'NOMP'), ('IABC+NOMP', 'NOMP+SC tuned'),
         ('IABC+NOMP+SC tuned', 'NOMP+SC tuned'), ('IABR-Net heat', 'NOMP+SC tuned'), ('NOMP+SC tuned', 'NOMP')]
SNRS = sorted(set(R[ORDER[0]]['snr'].astype(int).tolist()))
snr = R[ORDER[0]]['snr'].astype(int)
groups = [np.where(snr == s)[0] for s in SNRS]
rng = np.random.default_rng(0); B = 2000
IDX = np.concatenate([g_[rng.integers(0, len(g_), (B, len(g_)))] for g_ in groups], 1)     # (B, N) paired resample indices
HICOLS = np.concatenate([np.arange(k * len(groups[0]), (k + 1) * len(groups[0])) for k, s in enumerate(SNRS) if s >= 10])   # IDX columns of SNR >= 10
inrange = lambda r: float(r['phase_deg']) <= 8 and float(r['gain_db']) <= 3
out = dict(meta=dict(snrs=SNRS, n_per_snr=int(len(groups[0])), L=3, seeds='7700+k (k = SNR index)', bootstrap='2000 within SNR groups, seed 0',
                     training_range='phase <= 8 deg and gain <= 3 dB (uniform per antenna), L 1..9, SNR -15..24, 20% clean'), cells={})
md = []


def pct(a, b):
    return 1 - a / b if b > 0 else np.nan


# ------------------------------------------------------------------ estimation tables
for q, unit in [('ph', 'deg'), ('g', 'dB')]:
    md.append(f'\n### Identifiable {"phase" if q == "ph" else "gain"} error, RMS over 32 antennas ({unit}), mean over 1600 scenes (8 SNRs x 200); '
              f'[median]; "% removed" = 1 - MSE_after/MSE_before with 95% CI\n')
    md.append('| cell | in train range | ' + ' | '.join(ESTN[e] for e in EST) + ' | IABC % removed (MSE) | IABC % removed (RMS) | IABC % removed (MSE, SNR>=10) | '
              'self-cal tuned % removed (MSE) | self-cal tau0 4it % removed (MSE) | self-cal tau0 4it % removed (MSE, SNR>=10) | oracle % removed (MSE, SNR>=10) | unidentifiable part |')
    md.append('|' + '---|' * (len(EST) + 10))
    for n in ORDER:
        r = R[n]; c = out['cells'].setdefault(n, dict(phase_deg=float(r['phase_deg']), gain_db=float(r['gain_db']), in_train_range=inrange(r), est={}, pd={}))
        bef = r[f'{q}_before']; row = []
        for e in EST:
            a = r[f'{q}_{e}']
            ent = dict(mean=float(a.mean()), median=float(np.median(a)),
                       per_snr={s: dict(mean=float(a[g_].mean()), median=float(np.median(a[g_]))) for s, g_ in zip(SNRS, groups)})
            if e != 'before' and bef.mean() > 1e-6:      # float32 targets: a 'zero' impairment leaves ~1e-7 numerical residue
                bm = (a[IDX] ** 2).mean(1) / (bef[IDX] ** 2).mean(1); br = a[IDX].mean(1) / bef[IDX].mean(1)
                ent['pct_removed_mse'] = [pct((a ** 2).mean(), (bef ** 2).mean()), *np.percentile(1 - bm, [2.5, 97.5]).tolist()]
                ent['pct_removed_rms'] = [pct(a.mean(), bef.mean()), *np.percentile(1 - br, [2.5, 97.5]).tolist()]
                ent['per_snr_pct_removed_mse'] = {s: pct((a[g_] ** 2).mean(), (bef[g_] ** 2).mean()) for s, g_ in zip(SNRS, groups)}
                hi_ = np.concatenate([g_ for s, g_ in zip(SNRS, groups) if s >= 10]); IH = IDX[:, HICOLS]
                bh = (a[IH] ** 2).mean(1) / (bef[IH] ** 2).mean(1)
                ent['pct_removed_mse_snr_ge10'] = [pct((a[hi_] ** 2).mean(), (bef[hi_] ** 2).mean()), *np.percentile(1 - bh, [2.5, 97.5]).tolist()]
            c['est'][f'{q}_{e}'] = ent; row.append(f'{a.mean():.2f} [{np.median(a):.2f}]')
        # paired: IABC residual MSE minus self-cal tau0 4it residual MSE
        ia, sc = r[f'{q}_iabc'], r[f'{q}_sc_t0_i4']
        d = (ia[IDX] ** 2).mean(1) - (sc[IDX] ** 2).mean(1)
        c['est'][f'{q}_mse_iabc_minus_sct0i4'] = [float((ia ** 2).mean() - (sc ** 2).mean()), *np.percentile(d, [2.5, 97.5]).tolist()]
        un = r[f'{q}_unident']; c['est'][f'{q}_unident'] = dict(mean=float(un.mean()), median=float(np.median(un)))
        f = lambda e, k: (f'{100 * c["est"][f"{q}_{e}"][k][0]:.0f}% [{100 * c["est"][f"{q}_{e}"][k][1]:.0f},{100 * c["est"][f"{q}_{e}"][k][2]:.0f}]'
                          if k in c['est'][f'{q}_{e}'] else 'n/a')
        md.append(f'| {n} | {"yes" if inrange(r) else "no"} | ' + ' | '.join(row) + f' | {f("iabc", "pct_removed_mse")} | {f("iabc", "pct_removed_rms")} | '
                  f'{f("iabc", "pct_removed_mse_snr_ge10")} | {f("sc_tun", "pct_removed_mse")} | {f("sc_t0_i4", "pct_removed_mse")} | {f("sc_t0_i4", "pct_removed_mse_snr_ge10")} | '
                  f'{f("oracle", "pct_removed_mse_snr_ge10")} | {un.mean():.2f} |')

# per-SNR estimation for key cells
KEY = [n for n in ['ph8', 'g3', 'c8_3', 'ph20', 'g6', 'c30_6', 'ph45'] if n in R]
for q, unit in [('ph', 'deg'), ('g', 'dB')]:
    md.append(f'\n### Per SNR: identifiable {"phase" if q == "ph" else "gain"} RMS error ({unit}), mean over 200 scenes: before / IABC / self-cal tuned (gated) / self-cal tau0 4it / oracle ALS (IABC % removed, MSE)\n')
    md.append('| cell | ' + ' | '.join(f'SNR {s}' for s in SNRS) + ' |'); md.append('|' + '---|' * (len(SNRS) + 1))
    for n in KEY:
        e = out['cells'][n]['est']
        if e[f'{q}_before']['mean'] < 1e-6:
            continue
        md.append(f'| {n} | ' + ' | '.join(f'{e[f"{q}_before"]["per_snr"][s]["mean"]:.2f} / {e[f"{q}_iabc"]["per_snr"][s]["mean"]:.2f} / '
                                          f'{e[f"{q}_sc_tun"]["per_snr"][s]["mean"]:.2f} / {e[f"{q}_sc_t0_i4"]["per_snr"][s]["mean"]:.2f} / {e[f"{q}_oracle"]["per_snr"][s]["mean"]:.2f} '
                                          f'({100 * e[f"{q}_iabc"]["per_snr_pct_removed_mse"][s]:.0f}%)' for s in SNRS) + ' |')

# per-SNR % of identifiable MSE removed, every impaired cell (phase rows for cells with phase, gain rows for cells with gain)
for e in ['iabc', 'sc_tun', 'sc_t0_i4', 'oracle']:
    md.append(f'\n### Per SNR: % of identifiable MSE removed by {ESTN[e]} (1 - MSE_after/MSE_before, 200 scenes per SNR; negative = made worse)\n')
    md.append('| cell | quantity | in train range | ' + ' | '.join(f'SNR {s}' for s in SNRS) + ' | SNR>=10 [95% CI] |'); md.append('|' + '---|' * (len(SNRS) + 4))
    for n in ORDER:
        for q in ['ph', 'g']:
            ent = out['cells'][n]['est'][f'{q}_{e}']
            if 'per_snr_pct_removed_mse' not in ent:
                continue
            h = ent['pct_removed_mse_snr_ge10']
            md.append(f'| {n} | {"phase" if q == "ph" else "gain"} | {"yes" if out["cells"][n]["in_train_range"] else "no"} | ' +
                      ' | '.join(f'{100 * ent["per_snr_pct_removed_mse"][s]:.0f}%' for s in SNRS) + f' | {100 * h[0]:.0f}% [{100 * h[1]:.0f},{100 * h[2]:.0f}] |')

# per-SNR medians (before / IABC / self-cal tau0 4it / oracle)
md.append('\n### Per SNR: MEDIAN identifiable RMS error over 200 scenes, before / IABC / self-cal tau0 4it / oracle ALS (phase deg or gain dB)\n')
md.append('| cell | quantity | ' + ' | '.join(f'SNR {s}' for s in SNRS) + ' |'); md.append('|' + '---|' * (len(SNRS) + 2))
for n in KEY:
    for q in ['ph', 'g']:
        e = out['cells'][n]['est']
        if e[f'{q}_before']['mean'] < 1e-6:
            continue
        md.append(f'| {n} | {"phase" if q == "ph" else "gain"} | ' + ' | '.join(' / '.join(f'{e[f"{q}_{k}"]["per_snr"][s]["median"]:.2f}' for k in ['before', 'iabc', 'sc_t0_i4', 'oracle']) for s in SNRS) + ' |')

# ------------------------------------------------------------------ Pd
md.append('\n### Downstream paper Pd (mean over 8 SNRs, 1600 scenes, L=3) [strict Pd] {RMSE deg}\n')
md.append('| cell | ' + ' | '.join(METH) + ' |'); md.append('|' + '---|' * (len(METH) + 1))
for n in ORDER:
    r = R[n]; Sd = {m: r[f'S_{m}'] for m in METH}
    mb = {m: C.metric_by_snr(Sd[m], r['snr']) for m in METH}
    Bt = C.boot_metrics(Sd, r['snr'], B=2000, seed=0)
    c = out['cells'][n]
    c['pd'] = {m: dict(mean={k: mb[m]['mean'][k] for k in C.KEYS}, per_snr={s: {k: mb[m][s][k] for k in C.KEYS} for s in SNRS},
                       ci_pd_paper=list(C.ci(Bt[m][:, 0])[1:])) for m in METH}
    c['pd_diffs'] = {}
    for a_, b_ in DIFFS:
        dd = {}
        for i, k in enumerate(['pd_paper', 'pd_strict', 'pd_source', 'rmse']):
            dd[k] = [mb[a_]['mean'][k] - mb[b_]['mean'][k], *C.ci(Bt[a_][:, i] - Bt[b_][:, i])[1:]]
        per = {}
        for s, g_ in zip(SNRS, groups):
            Bs = C.boot_metrics({a_: Sd[a_][g_], b_: Sd[b_][g_]}, r['snr'][g_], B=2000, seed=0)
            per[s] = [mb[a_][s]['pd_paper'] - mb[b_][s]['pd_paper'], *C.ci(Bs[a_][:, 0] - Bs[b_][:, 0])[1:]]
        dd['per_snr_pd_paper'] = per
        c['pd_diffs'][f'{a_} - {b_}'] = dd
    c['gate_open'] = dict(default=float(r['gate_def'].mean()), tuned=float(r['gate_tun'].mean()), iabc_tuned=float(r['gate_ct'].mean()),
                          dual_picks_heat=float(r['pick_h'].mean()))
    md.append(f'| {n} | ' + ' | '.join(f'{mb[m]["mean"]["pd_paper"]:.3f} [{mb[m]["mean"]["pd_strict"]:.3f}] {{{mb[m]["mean"]["rmse"]:.2f}}}' for m in METH) + ' |')

md.append('\n### Paired paper-Pd differences (mean over SNR), 95% CI; * = CI excludes 0\n')
md.append('| cell | in train range | ' + ' | '.join(f'{a} - {b}' for a, b in DIFFS) + ' |'); md.append('|' + '---|' * (len(DIFFS) + 2))
star = lambda v: f'{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]' + ('*' if v[1] > 0 or v[2] < 0 else '')
for n in ORDER:
    c = out['cells'][n]
    md.append(f'| {n} | {"yes" if c["in_train_range"] else "no"} | ' + ' | '.join(star(c['pd_diffs'][f'{a} - {b}']['pd_paper']) for a, b in DIFFS) + ' |')

for n in [x for x in ['c30_6', 'ph45', 'g6', 'c20_4', 'ph30', 'c8_3', 'clean'] if x in R]:
    md.append(f'\n### Per-SNR paper-Pd differences, cell {n}\n')
    md.append('| diff | ' + ' | '.join(f'SNR {s}' for s in SNRS) + ' |'); md.append('|' + '---|' * (len(SNRS) + 1))
    for a, b in DIFFS[:5]:
        p = out['cells'][n]['pd_diffs'][f'{a} - {b}']['per_snr_pd_paper']
        md.append(f'| {a} - {b} | ' + ' | '.join(star(p[s]) for s in SNRS) + ' |')

open(f'{HERE}/tables.md', 'w', encoding='utf-8').write('\n'.join(md) + '\n')
print('\n'.join(md))

# ------------------------------------------------------------------ figures
PH = [n for n in ['clean', 'ph2', 'ph5', 'ph8', 'ph12', 'ph20', 'ph30', 'ph45'] if n in R]
GN = [n for n in ['clean', 'g1', 'g2', 'g3', 'g4', 'g6'] if n in R]
sty = {'before': ('k', 'o', '-'), 'iabc': ('C3', 's', '-'), 'sc_t0_i4': ('C0', '^', '--'), 'sc_tun': ('C2', 'v', ':'), 'oracle': ('C7', 'D', '-.'),
       'iabc_sct': ('C1', 'x', '-')}
fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
for row, (q, cells, xk, unit) in enumerate([('ph', PH, 'phase_deg', 'deg'), ('g', GN, 'gain_db', 'dB')]):
    for col, s in enumerate([15, 0]):
        a = ax[row, col]; x = [out['cells'][n][xk] for n in cells]
        for e, (cl, mk, ls) in sty.items():
            a.plot(x, [out['cells'][n]['est'][f'{q}_{e}']['per_snr'][s]['mean'] for n in cells], color=cl, marker=mk, ls=ls, label=ESTN[e])
        a.axvspan(0, 8 if q == 'ph' else 3, color='C3', alpha=0.07, label='IABC training range')
        a.set_xlabel(f'max per-antenna {"phase" if q == "ph" else "gain"} error ({unit}, uniform +-max)')
        a.set_ylabel(f'identifiable {"phase" if q == "ph" else "gain"} error, RMS ({unit})'); a.set_title(f'{"phase-only" if q == "ph" else "gain-only"} sweep, SNR {s} dB')
        a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=8)
fig.suptitle('Per-antenna impairment left after correction (L=3, 200 scenes per point)'); fig.tight_layout(); fig.savefig(f'{HERE}/fig_iabc_reduction.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for a, (n, q, unit) in zip(ax, [('ph8', 'ph', 'deg'), ('g3', 'g', 'dB')]):
    if n not in R:
        continue
    for e, (cl, mk, ls) in sty.items():
        a.plot(SNRS, [out['cells'][n]['est'][f'{q}_{e}']['per_snr'][s]['mean'] for s in SNRS], color=cl, marker=mk, ls=ls, label=ESTN[e])
    a.set_xlabel('SNR (dB)'); a.set_ylabel(f'identifiable {"phase" if q == "ph" else "gain"} error, RMS ({unit})')
    a.set_title('phase-only, max 8 deg' if q == 'ph' else 'gain-only, max 3 dB'); a.grid(alpha=0.3)
ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(f'{HERE}/fig_iabc_vs_snr.png', dpi=130); plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
msty = {'NOMP': 'k', 'IABC+NOMP': 'C3', 'IABR-Net heat': 'C1', 'NOMP+SC tuned': 'C0', 'IABC+NOMP+SC tuned': 'C2'}
for a, (cells, xk, lab) in zip(ax, [(PH, 'phase_deg', 'max phase error (deg)'), (GN, 'gain_db', 'max gain error (dB)'),
                                     ([n for n in ['clean', 'c5_2', 'c8_3', 'c20_4', 'c30_6'] if n in R], None, 'combined (phase deg, gain dB)')]):
    x = np.arange(len(cells)) if xk is None else [out['cells'][n][xk] for n in cells]
    for m, cl in msty.items():
        y = [out['cells'][n]['pd'][m]['mean']['pd_paper'] for n in cells]
        lo = [out['cells'][n]['pd'][m]['ci_pd_paper'][0] for n in cells]; hi = [out['cells'][n]['pd'][m]['ci_pd_paper'][1] for n in cells]
        a.plot(x, y, color=cl, marker='o', label=m); a.fill_between(x, lo, hi, color=cl, alpha=0.12)
    if xk is None:
        a.set_xticks(x); a.set_xticklabels(['(0,0)' if n == 'clean' else f'({int(out["cells"][n]["phase_deg"])},{int(out["cells"][n]["gain_db"])})' for n in cells])
    a.set_xlabel(lab); a.set_ylabel('paper Pd (mean over SNR -10..25)'); a.grid(alpha=0.3)
ax[0].legend(fontsize=8); fig.suptitle('Detection probability vs impairment severity (L=3, 1600 scenes per point, shaded = 95% bootstrap CI)')
fig.tight_layout(); fig.savefig(f'{HERE}/fig_pd_vs_severity.png', dpi=130); plt.close(fig)

if os.path.exists(f'{HERE}/s4_banks.json'):
    out['cached_banks'] = json.load(open(f'{HERE}/s4_banks.json'))
json.dump(out, open(f'{HERE}/results.json', 'w'), indent=1, default=float)
print('wrote results.json, tables.md, fig_iabc_reduction.png, fig_iabc_vs_snr.png, fig_pd_vs_severity.png')
