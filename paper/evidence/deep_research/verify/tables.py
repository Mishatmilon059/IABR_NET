"""Step 5 and 6: side-by-side tables (metric_mine.py metric; paired bootstrap 2000 resamples within SNR groups, seed 0) and the
oracle-seeded headroom on the official bank. Writes tables.md and verify_tables.json.
usage: python tables.py   (stdout -> tables.log)"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import *
import metric_mine as MM
import nomp_mine as NM

METHODS = ['U-Net', 'IABR-v2 dual', 'IABR-Net heat', 'network only', 'NOMP (cached)', 'NOMP (mine)', 'NOMP+selfcal']
DIFFS = [('IABR-Net heat - NOMP', 'IABR-Net heat', 'NOMP (cached)'), ('IABR-v2 dual - NOMP', 'IABR-v2 dual', 'NOMP (cached)'),
         ('NOMP - U-Net', 'NOMP (cached)', 'U-Net'), ('network only - NOMP', 'network only', 'NOMP (cached)'),
         ('NOMP (mine) - NOMP (cached)', 'NOMP (mine)', 'NOMP (cached)')]


def sources(name):
    s = {'U-Net': f'{CACHE}/pred__U-Net_paper__{name}.npz', 'IABR-v2 dual': f'{CACHE}/pred__IABR-v2__{name}.npz',
         'IABR-Net heat': f'{NEWR}/preds_heat/{name}.npz', 'network only': f'{CACHE}/pred__v2_network_only_no_physics__{name}.npz',
         'NOMP (cached)': f'{CACHE}/pred__NOMP_no_network__{name}.npz', 'NOMP (mine)': f'{MYCACHE}/mynomp__{name}.npz',
         'NOMP+selfcal': f'{NEWR}/preds/selfcal_pred__{name}.npz'}
    note = {}
    if not os.path.exists(s['IABR-Net heat']):
        s['IABR-Net heat'] = f'{MYCACHE}/heat__{name}.npz'; note['IABR-Net heat'] = 'recomputed (fill_missing.py)'
    if not os.path.exists(s['network only']):
        s['network only'] = f'{MYCACHE}/netonly__{name}.npz'; note['network only'] = 'recomputed (fill_missing.py)'
    return {k: v for k, v in s.items() if os.path.exists(v)}, note


def f4(x):
    return '—' if x is None or not np.isfinite(x) else f'{x:.4f}'


J = {}; md = []
md.append('## Side-by-side, per bank (mean over SNR groups; 95% paired-bootstrap percentile CI, 2000 resamples within SNR groups, seed 0)\n')
for name in BANKS:
    b = load_truth(name); src, note = sources(name)
    S = {}
    for m in METHODS:
        if m in src:
            ps, ph, n = load_pred(src[m]); S[m], nn = MM.scene_table(ps, ph, n, b['psi'], b['phi'], b['L'])
            assert nn == 0
    Bt = MM.paired_bootstrap(S, b['snr'], 2000, 0)
    snrs = sorted(set(b['snr'].astype(int).tolist()))
    jb = dict(n=len(b['L']), L=sorted(set(b['L'].tolist())), snr=snrs, notes=note, methods={}, diffs={})
    md.append(f'### {name}  (N={len(b["L"])}, L={jb["L"]}, SNR groups={snrs})\n')
    md.append('| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped scenes |\n|---|---|---|---|---|')
    for m in METHODS:
        if m not in S:
            md.append(f'| {m} | not available | | | |'); continue
        per = MM.by_snr(S[m], b['snr'])
        cis = {k: MM.pct_ci(Bt[m][:, j]) for j, k in enumerate(MM.KEYS)}
        jb['methods'][m] = dict(mean=per['mean'], ci=cis, per_snr={str(s): per[s] for s in snrs}, source=src[m])
        mm = per['mean']
        md.append(f'| {m}{" (" + note[m] + ")" if m in note else ""} | {f4(mm["pd_paper"])} [{f4(cis["pd_paper"][0])}, {f4(cis["pd_paper"][1])}] '
                  f'| {f4(mm["pd_strict"])} [{f4(cis["pd_strict"][0])}, {f4(cis["pd_strict"][1])}] '
                  f'| {f4(mm["rmse"])} [{f4(cis["rmse"][0])}, {f4(cis["rmse"][1])}] | {mm["dropped"]} |')
    md.append('\n| paired difference | d Pd paper-style [95% CI] | d Pd strict [95% CI] | d RMSE deg [95% CI] |\n|---|---|---|---|')
    for nm, a, c in DIFFS:
        if a in S and c in S:
            d = Bt[a] - Bt[c]; ma, mc = jb['methods'][a]['mean'], jb['methods'][c]['mean']
            jb['diffs'][nm] = {k: [ma[k] - mc[k], *MM.pct_ci(d[:, j])] for j, k in enumerate(MM.KEYS)}
            x = jb['diffs'][nm]
            md.append(f'| {nm} | {x["pd_paper"][0]:+.4f} [{x["pd_paper"][1]:+.4f}, {x["pd_paper"][2]:+.4f}] | {x["pd_strict"][0]:+.4f} [{x["pd_strict"][1]:+.4f}, {x["pd_strict"][2]:+.4f}] '
                      f'| {x["rmse"][0]:+.4f} [{x["rmse"][1]:+.4f}, {x["rmse"][2]:+.4f}] |')
    md.append('')
    J[name] = jb
    print(f'{name} done', flush=True)

# ---- per-SNR table, official bank, plus oracle-seeded headroom
name = 'official'; b = load_truth(name); src, _ = sources(name)
S = {m: MM.scene_table(*load_pred(src[m]), b['psi'], b['phi'], b['L'])[0] for m in METHODS if m in src}
for R in [3, 10]:
    S[f'oracle-seeded ({R} rounds)'] = MM.scene_table(*load_pred(f'{MYCACHE}/oracle{R}__official.npz'), b['psi'], b['phi'], b['L'])[0]
snr = b['snr'].astype(int); snrs = sorted(set(snr.tolist()))
cols = list(S)
md.append('## Official bank, per SNR (paper-style Pd; strict Pd differs only for U-Net)\n')
md.append('| SNR dB | ' + ' | '.join(cols) + ' |\n|' + '---|' * (len(cols) + 1))
per = {m: MM.by_snr(S[m], b['snr']) for m in cols}
for s in snrs + ['mean']:
    md.append(f'| {s} | ' + ' | '.join(f4(per[m][s]['pd_paper']) for m in cols) + ' |')
md.append('\nU-Net strict Pd per SNR: ' + ', '.join(f'{s}: {per["U-Net"][s]["pd_strict"]:.4f}' for s in snrs) + f'; mean {per["U-Net"]["mean"]["pd_strict"]:.4f}\n')
md.append('RMSE (deg, detected components) per SNR:\n\n| SNR dB | ' + ' | '.join(cols) + ' |\n|' + '---|' * (len(cols) + 1))
for s in snrs + ['mean']:
    md.append(f'| {s} | ' + ' | '.join(f4(per[m][s]['rmse']) for m in cols) + ' |')

# headroom: Pd gap oracle-seeded - NOMP per SNR with paired bootstrap inside each SNR group
md.append('\n## Headroom: oracle-seeded decoder vs NOMP (official bank, paper-style Pd; 95% CI = paired bootstrap 2000 resamples within the SNR group, seed 0)\n')
md.append('| SNR dB | NOMP | oracle-seeded (3 rounds) | gap (oracle - NOMP) [95% CI] | oracle-seeded (10 rounds) | min-residual(NOMP, oracle) | gap (min-res - NOMP) [95% CI] | scenes where oracle residual < NOMP residual | scenes NOMP misses >=1 comp. | of those, oracle all-correct | of those, oracle residual < NOMP residual |\n|---|---|---|---|---|---|---|---|---|---|---|')
H = np.load(f'{CACHE}/net__official.npz')['Ht'].astype(np.complex128)
psN, phN, _ = load_pred(src['NOMP (cached)']); psO, phO, _ = load_pred(f'{MYCACHE}/oracle3__official.npz')


def resid(psi, phi):
    U = np.pi * np.cos(psi); V = np.pi * np.cos(phi); out = np.zeros(len(U))
    for s in range(0, len(U), 1000):
        x, A = NM.ls_fit(H[s:s + 1000], U[s:s + 1000], V[s:s + 1000])
        out[s:s + 1000] = (np.abs(H[s:s + 1000].reshape(-1, 256) - np.einsum('bik,bk->bi', A, x)) ** 2).sum(1)
    return out


rN = resid(psN, phN); rO = resid(psO, phO)
SN = S['NOMP (cached)']; SO = S['oracle-seeded (3 rounds)']
pick = rO < rN                                         # the objective prefers the oracle-seeded solution
SM = np.where(pick[:, None], SO, SN)                    # min-residual of {NOMP, oracle-seeded}: best this objective can certify
perM = MM.by_snr(SM, b['snr'])
head = {}
for s in snrs + ['all']:
    ix = np.arange(len(snr)) if s == 'all' else np.where(snr == s)[0]
    Bt = MM.paired_bootstrap({'N': SN[ix], 'O': SO[ix], 'M': SM[ix]}, snr[ix], 2000, 0)
    d = (Bt['O'] - Bt['N'])[:, 0]; dm = (Bt['M'] - Bt['N'])[:, 0]
    key = s if s != 'all' else 'mean'
    pn = per['NOMP (cached)'][key]['pd_paper']; po = per['oracle-seeded (3 rounds)'][key]['pd_paper']
    p10 = per['oracle-seeded (10 rounds)'][key]['pd_paper']; pm = perM[key]['pd_paper']
    miss = ix[SN[ix, 1] > 0]; fix = miss[SO[miss, 1] == 0]; lower = int((rO[fix] < rN[fix]).sum())
    head[str(s)] = dict(nomp=pn, oracle3=po, oracle10=p10, gap=po - pn, gap_ci=MM.pct_ci(d), minres=pm, minres_gap=pm - pn, minres_gap_ci=MM.pct_ci(dm),
                        n_scenes_oracle_lower_residual=int(pick[ix].sum()), n_nomp_miss=len(miss), n_oracle_fixes=len(fix), n_fix_lower_residual=lower)
    md.append(f'| {s if s != "all" else "mean over SNR"} | {pn:.4f} | {po:.4f} | {po-pn:+.4f} [{MM.pct_ci(d)[0]:+.4f}, {MM.pct_ci(d)[1]:+.4f}] | {p10:.4f} '
              f'| {pm:.4f} | {pm-pn:+.4f} [{MM.pct_ci(dm)[0]:+.4f}, {MM.pct_ci(dm)[1]:+.4f}] | {int(pick[ix].sum())} | {len(miss)} | {len(fix)} | {lower} |')
md.append('\n(residual = ||Ht - A x_LS||^2 at the decoded (u,v). "min-residual" keeps, per scene, whichever of the NOMP and oracle-seeded (3 rounds) '
          'solutions has the lower residual -- the oracle start is used only where the decoder objective itself prefers it, so its gap over NOMP is the '
          'search-failure headroom of this objective; the rest of the oracle-seeded gap comes from scenes where the oracle start sits in a local optimum near the '
          'truth that fits the data worse than the NOMP solution, i.e. the objective (noise/impairments) rather than the search is the limit.)\n')
J['official_per_snr'] = {m: {str(k): v for k, v in per[m].items()} for m in cols}
J['headroom'] = head
print(f'residual check: scenes where oracle residual < NOMP residual: {int(pick.sum())} of {len(pick)}; median residual ratio oracle/NOMP {np.median(rO / rN):.4f}', flush=True)
open(f'{HERE}/tables.md', 'w', encoding='utf-8').write('\n'.join(md))
json.dump(J, open(f'{HERE}/verify_tables.json', 'w'), indent=1, default=float)
print('\n'.join(md))
