"""Analysis of unfolded_v2/build evaluations: tables, paired bootstrap (2000 resamples within SNR groups, seed 0), literal application of
the pre-registered criteria (../design/criteria.json), figures. Reads per-scene stats from ev.py (BIG/stats). Writes results.json,
fig_pd_vs_snr.png, fig_regimes.png. stdout -> analyze.log"""
import os
os.environ.setdefault('THREADS', '4')
from ev import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = dict(notes=[])
DESIGNS = [m for m in ['A', 'B'] if os.path.exists(stats_path('official', m))]


def S_of(bank, m):
    if m in ('U-Net (Lloria et al.)', 'lurd all-neural (cached)'):
        return cached_pred_stats(bank, m)
    return load_stats(bank, m)['S']


def have(bank, m):
    if m == 'U-Net (Lloria et al.)':
        return os.path.exists(f'{C.CACHE}/pred__U-Net_paper__{bank}.npz')
    if m == 'lurd all-neural (cached)':
        return os.path.exists(f'C:/Users/Milon/iabr_deep_research_cache/dr_cache/nn_decoder/{bank}__allneural_sc.npz')
    return os.path.exists(stats_path(bank, m))


def snr_of(bank):
    return load(bank)['snr'] if bank not in _SNR else _SNR[bank]


_SNR = {}


def bank_snr(bank):
    if bank not in _SNR:
        _SNR[bank] = load(bank)['snr']
    return _SNR[bank]


def per_snr_diff(Sa, Sb, snr, B=2000):
    """per SNR group: Pd a - b with paired bootstrap CI (resampling within that group, seed 0)"""
    out = {}
    for s in sorted(set(snr.astype(int).tolist())):
        ix = np.where(snr.astype(int) == s)[0]
        Bt = C.boot_metrics({'a': Sa[ix], 'b': Sb[ix]}, snr[ix], B=B, seed=0)
        d = Bt['a'][:, 0] - Bt['b'][:, 0]
        out[s] = [pd(Sa[ix]) - pd(Sb[ix]), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    return out


def table_bank(bank, meths, ref_pairs):
    meths = [m for m in meths if have(bank, m)]
    snr = bank_snr(bank); S = {m: S_of(bank, m) for m in meths}
    pairs = [(a, b) for a, b in ref_pairs if a in S and b in S]
    pdm, ci_, D = boot_pairs(S, snr, pairs)
    per = {m: {int(k): v['pd_paper'] for k, v in C.metric_by_snr(S[m], snr).items() if k != 'mean'} for m in meths}
    strict = {m: C.metric_by_snr(S[m], snr)['mean']['pd_strict'] for m in meths}
    rmse = {m: C.metric_by_snr(S[m], snr)['mean']['rmse'] for m in meths}
    return dict(pd=pdm, ci=ci_, diffs=D, per_snr=per, pd_strict=strict, rmse=rmse), S


def pr(*a):
    print(*a, flush=True)


# ================================================================ official / replicate / T2
OFF_M = DESIGNS + ['NOMP', 'NOMP os8 rc6', 'NOMP+SC tuned', 'classical skeleton', 'classical heat decoder', 'lurd all-neural (cached)', 'U-Net (Lloria et al.)']
pairs = [(d, r) for d in DESIGNS for r in ['NOMP', 'NOMP os8 rc6', 'classical skeleton', 'classical heat decoder', 'lurd all-neural (cached)', 'U-Net (Lloria et al.)']]
if 'A' in DESIGNS and 'B' in DESIGNS:
    pairs.append(('A', 'B'))
R['official'], S_off = table_bank('official', OFF_M, pairs)
snr_off = bank_snr('official')
for d in DESIGNS:
    R['official'][f'per_snr_{d}_vs_os8rc6'] = per_snr_diff(S_off[d], S_off['NOMP os8 rc6'], snr_off)
    R['official'][f'per_snr_{d}_vs_NOMP'] = per_snr_diff(S_off[d], S_off['NOMP'], snr_off)
pr('=== official bank (8000, L=3 given, ideal hardware)')
pr('method'.ljust(28) + ''.join(f'{s:>8d}' for s in range(-10, 26, 5)) + '    mean   strict   rmse')
for m in R['official']['pd']:
    pr(m.ljust(28) + ''.join(f'{R["official"]["per_snr"][m][s]:8.4f}' for s in range(-10, 26, 5)) + f'  {R["official"]["pd"][m]:.4f}  {R["official"]["pd_strict"][m]:.4f}  {R["official"]["rmse"][m]:.4f}')
for k, v in R['official']['diffs'].items():
    pr(f'   {k:50s} {fmt(v)}')
for d in DESIGNS:
    pr(f'   per SNR {d} - NOMP os8 rc6: ' + ', '.join(f'{s}: {fmt(v)}' for s, v in R['official'][f'per_snr_{d}_vs_os8rc6'].items()))

for bank in ['rep9520', 'T2_d5']:
    ms = DESIGNS + ['NOMP', 'NOMP os8 rc6', 'NOMP+SC tuned', 'classical skeleton', 'classical heat decoder', 'lurd all-neural', 'lurd all-neural (cached)', 'U-Net (Lloria et al.)']
    R[bank], _ = table_bank(bank, ms, [(d, r) for d in DESIGNS for r in ['NOMP', 'NOMP os8 rc6', 'classical skeleton', 'U-Net (Lloria et al.)']])
    pr(f'=== {bank}'); pr('   ' + ', '.join(f'{m} {v:.4f}' for m, v in R[bank]['pd'].items()))
    for k, v in R[bank]['diffs'].items():
        pr(f'   {k:50s} {fmt(v)}')

# ================================================================ regime cells
PRE_CLASSICAL = ['NOMP', 'NOMP os8 rc6', 'NOMP+SC tuned', 'consistent NOMP', 'ML', 'annealed ML', 'EB-ML-g5', 'PN-ML all starts', 'POS-MAP all starts']
REG_M = DESIGNS + PRE_CLASSICAL + ['NOMP os8 rc6+SC tuned', 'classical skeleton', 'ML snr+5', 'ML snr-5', 'ML joint sigma', 'ML nominal 10dB',
                                   'hybrid ML from FT seeds', 'FT network only', 'classical heat decoder', 'lurd all-neural']
R['regime'] = {}
for cell in REGIME:
    ms = [m for m in REG_M if have(cell, m)]
    if not ms:
        continue
    snr = bank_snr(cell); S = {m: S_of(cell, m) for m in ms}
    pdm0 = {m: pd(S[m]) for m in ms}
    pre = [m for m in PRE_CLASSICAL if m in S]; best = max(pre, key=lambda m: pdm0[m])
    sens = pre + [m for m in ['classical skeleton', 'NOMP os8 rc6+SC tuned'] if m in S]; best2 = max(sens, key=lambda m: pdm0[m])
    if cell.startswith('M5'):
        deploy = 'ML joint sigma'; deploy2 = max(['ML joint sigma', 'ML nominal 10dB'], key=lambda m: pdm0.get(m, -1))
    else:
        deploy = 'NOMP os8 rc6+SC tuned'; deploy2 = deploy
    prs = []
    for d in DESIGNS:
        prs += [(d, best), (d, best2), (d, 'NOMP'), (d, deploy), (d, deploy2)]
        if cell.startswith('M5'):
            prs += [(d, 'ML snr+5'), (d, 'ML snr-5'), (d, 'hybrid ML from FT seeds')]
    if 'A' in S and 'B' in S:
        prs.append(('A', 'B'))
    prs = [(a, b) for a, b in dict.fromkeys(prs) if a in S and b in S and a != b]
    pdm, ci_, D = boot_pairs(S, snr, prs)
    R['regime'][cell] = dict(pd=pdm, ci=ci_, diffs=D, best_classical_prereg=best, best_classical_with_skeleton=best2, deployable=deploy, deployable_sens=deploy2,
                             pd_strict={m: C.metric_by_snr(S[m], snr)['mean']['pd_strict'] for m in ms})
    pr(f'=== {cell}: best pre-registered classical = {best} {pdm[best]:.4f}; incl. classical skeleton = {best2} {pdm[best2]:.4f}')
    pr('   ' + ', '.join(f'{m} {v:.4f}' for m, v in pdm.items()))
    for k, v in D.items():
        pr(f'   {k:50s} {fmt(v)}')

# ================================================================ unknown L (hit-F1)
def f1_boot(Fa, Fb, snr, B=2000):
    rng = np.random.default_rng(0); groups = [np.where(snr.astype(int) == s)[0] for s in sorted(set(snr.astype(int).tolist()))]
    acc = np.zeros(B); pa = pb = 0
    def f1(F):
        return 2 * F[..., 0] / (F[..., 1] + F[..., 2])
    for ix in groups:
        Rr = rng.integers(0, len(ix), (B, len(ix)))
        sa = Fa[ix][Rr].sum(1); sb = Fb[ix][Rr].sum(1)
        acc += f1(sa) - f1(sb)
        pa += f1(Fa[ix].sum(0)); pb += f1(Fb[ix].sum(0))
    acc /= len(groups)
    return [(pa - pb) / len(groups), float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


R['unknownL'] = {}
for bank in ['Lmix9550'] + [f'L{l}_snr{s}' for l in [1, 2, 4, 5, 6] for s in [0, 15]]:
    ms = [m for m in [f'{d} unknownL' for d in DESIGNS] + ['NOMP stop true sigma', 'NOMP stop est sigma'] if have(bank, m)]
    if not ms:
        continue
    snr = bank_snr(bank); F = {m: load_stats(bank, m)['f1'] for m in ms}
    f1 = {m: float(np.mean([2 * F[m][snr.astype(int) == s].sum(0)[0] / F[m][snr.astype(int) == s].sum(0)[1:].sum() for s in sorted(set(snr.astype(int).tolist()))])) for m in ms}
    fa = {m: float((F[m][:, 2].sum() - F[m][:, 0].sum()) / max(F[m][:, 2].sum(), 1)) for m in ms}
    base = [m for m in ['NOMP stop true sigma', 'NOMP stop est sigma'] if m in F]
    best = max(base, key=lambda m: f1[m]) if base else None
    D = {f'{d} unknownL - {b}': f1_boot(F[f'{d} unknownL'], F[b], snr) for d in DESIGNS if f'{d} unknownL' in F for b in base}
    R['unknownL'][bank] = dict(f1=f1, fa=fa, best_baseline=best, diffs=D, mean_n_est={m: float(F[m][:, 2].mean()) for m in ms})
    pr(f'=== unknown L {bank}: ' + ', '.join(f'{m} F1 {f1[m]:.4f} FA {fa[m]:.3f} n_est {F[m][:, 2].mean():.2f}' for m in ms))
    for k, v in D.items():
        pr(f'   {k:50s} {fmt(v)}')

# ================================================================ prior violation / L != 3 with L given
R['prior'] = {}
for bank in [f'sep_{k}deg_snr15' for k in [1, 2, 3, 5, 10, 20]] + ['nosep9560', 'sep10_9561'] + [f'L{l}_snr{s}' for l in [1, 2, 4, 5, 6] for s in [0, 15]]:
    ms = [m for m in DESIGNS + ['NOMP', 'NOMP os8 rc6', 'NOMP+SC tuned', 'classical skeleton'] if have(bank, m)]
    if not ms or not DESIGNS or DESIGNS[0] not in ms:
        continue
    t, S = table_bank(bank, ms, [(d, r) for d in DESIGNS for r in ['NOMP', 'NOMP os8 rc6']])
    R['prior'][bank] = t
    pr(f'=== {bank}: ' + ', '.join(f'{m} {v:.4f}' for m, v in t['pd'].items()))
    for k, v in t['diffs'].items():
        pr(f'   {k:50s} {fmt(v)}')

# ================================================================ OOD report-only
R['ood'] = {}
for bank, ms in GROUPS['ood']:
    ms = [m for m in DESIGNS + ms if have(bank, m)]
    if not ms or not DESIGNS or DESIGNS[0] not in ms:
        continue
    t, _ = table_bank(bank, ms, [(d, r) for d in DESIGNS for r in ms if r not in DESIGNS])
    R['ood'][bank] = t
    pr(f'=== OOD {bank}: ' + ', '.join(f'{m} {v:.4f}' for m, v in t['pd'].items()))
    for k, v in t['diffs'].items():
        if k.startswith('A - '):
            pr(f'   {k:50s} {fmt(v)}')

# ================================================================ ablations of A (subsets)
R['ablation'] = {}
ABL = ['A abl:kappa1', 'A abl:eta0', 'A abl:truesig', 'A abl:stepoff', 'A abl:selnll', 'A abl:beta0']
for bank in ['official_sub'] + [f'sub500:{c}' for c in ['M5_b1_snr15', 'M5_b2_snr15', 'M5_b1_snr0', 'M5_b2_snr0', 'M2_0.2_snr15', 'M6_30_snr15']]:
    src = 'official' if bank == 'official_sub' else bank[7:]
    ms = [m for m in ['A', 'B'] + ABL if have(bank, m)]
    if len(ms) < 2:
        continue
    if bank == 'official_sub':
        snr_full = bank_snr('official').astype(int)
        idx = np.concatenate([np.where(snr_full == s_)[0][:250] for s_ in sorted(set(snr_full.tolist()))])
    else:
        idx = np.arange(500)
    snr = bank_snr(src)[idx]
    S = {}
    for m in ['A', 'B'] + ABL:
        if have(bank, m):
            S[m] = load_stats(bank, m)['S']
        elif have(src, m):
            S[m] = load_stats(src, m)['S'][idx]
    prs = [('A', m) for m in S if m != 'A']
    pdm, ci_, D = boot_pairs(S, snr, prs)
    R['ablation'][bank] = dict(pd=pdm, diffs=D)
    pr(f'=== ablation {bank}: ' + ', '.join(f'{m} {v:.4f}' for m, v in pdm.items()))
    for k, v in D.items():
        pr(f'   {k:40s} {fmt(v)}')


# ================================================================ criteria (literal application of ../design/criteria.json)
def ci_pos(d):
    return d[1] > 0


def ci_neg(d):
    return d[2] < 0


CRIT = {}
for D_ in DESIGNS:
    c = {}
    off = R['official']['diffs']; ps = R['official'][f'per_snr_{D_}_vs_os8rc6']
    p1 = dict(vs_os4rc3=off[f'{D_} - NOMP'], vs_os8rc6=off[f'{D_} - NOMP os8 rc6'],
              sig_loss_snr_ge10=[s_ for s_ in [10, 15, 20, 25] if ci_neg(ps[s_])],
              replicate_vs_os8rc6=R.get('rep9520', {}).get('diffs', {}).get(f'{D_} - NOMP os8 rc6'))
    p1['PASS'] = bool(ci_pos(p1['vs_os4rc3']) and ci_pos(p1['vs_os8rc6']) and not p1['sig_loss_snr_ge10'] and p1['replicate_vs_os8rc6'] is not None and p1['replicate_vs_os8rc6'][0] > 0)
    c['P1'] = p1
    s2 = dict(lower_ci_vs_os8rc6=off[f'{D_} - NOMP os8 rc6'][1], sig_loss_snr_ge10=p1['sig_loss_snr_ge10'])
    s2['PASS'] = bool(s2['lower_ci_vs_os8rc6'] > -0.002 and not s2['sig_loss_snr_ge10']); c['S2'] = s2
    wins, bigloss, wins2, dwins, dloss, dwins2, dloss2, s3w = [], [], [], [], [], [], [], []
    for cell, r in R['regime'].items():
        if D_ not in r['pd']:
            continue
        d1 = r['diffs'].get(f"{D_} - {r['best_classical_prereg']}")
        if d1 and ci_pos(d1):
            wins.append(cell)
        if d1 and d1[0] < -0.02 and ci_neg(d1):
            bigloss.append(cell)
        d2 = r['diffs'].get(f"{D_} - {r['best_classical_with_skeleton']}")
        if d2 and ci_pos(d2):
            wins2.append(cell)
        dd = r['diffs'].get(f"{D_} - {r['deployable']}")
        if dd and ci_pos(dd):
            dwins.append(cell)
        if dd and ci_neg(dd):
            dloss.append(cell)
        dd2 = r['diffs'].get(f"{D_} - {r['deployable_sens']}")
        if dd2 and ci_pos(dd2):
            dwins2.append(cell)
        if dd2 and ci_neg(dd2):
            dloss2.append(cell)
        if cell.startswith('M5'):
            a, b = r['diffs'].get(f'{D_} - ML snr+5'), r['diffs'].get(f'{D_} - ML snr-5')
            if a and b and ci_pos(a) and ci_pos(b):
                s3w.append(cell)
    c['S1'] = dict(wins=wins, n_wins=len(wins), n_cells=len(R['regime']), big_sig_losses=bigloss,
                   PASS=bool(len(wins) >= 7 and not bigloss and s2['PASS']), sensitivity_wins_incl_skeleton=wins2)
    c['S1b'] = dict(wins=dwins, n_wins=len(dwins), sig_losses=dloss, PASS=bool(len(dwins) >= 10 and not dloss),
                    sensitivity_with_nominal_ML=dict(wins=dwins2, sig_losses=dloss2))
    c['S3'] = dict(wins=s3w, n_wins=len(s3w), PASS=bool(len(s3w) >= 6))
    uk = R['unknownL']; s4 = dict(PASS=False)
    if 'Lmix9550' in uk and f'{D_} unknownL' in uk['Lmix9550']['f1']:
        def ok_cell(b):
            r = uk[b]; d = r['diffs'].get(f"{D_} unknownL - {r['best_baseline']}")
            fa_ok = r['fa'][f'{D_} unknownL'] <= r['fa'][r['best_baseline']] + 0.02
            return bool(d and ci_pos(d) and fa_ok), d
        lm_ok, lm_d = ok_cell('Lmix9550')
        cells = [b for b in uk if b != 'Lmix9550']
        n_ok = sum(ok_cell(b)[0] for b in cells)
        s4 = dict(Lmix=lm_d, Lmix_ok=lm_ok, cached_cells_ok=n_ok, n_cached=len(cells), PASS=bool(lm_ok and n_ok >= 6))
    c['S4'] = s4
    lat = json.load(open(f'{HERE}/latency.json')) if os.path.exists(f'{HERE}/latency.json') else None
    if lat and all(D_ in lat[b] for b in lat):
        med = max(lat[b][D_]['batch1_median_ms'] for b in lat)
        c['S5'] = dict(median_ms_worst_bank=med, PASS=bool(med <= 120 and med < 482))
    h1 = dict(violations=[])
    for b, r in R['prior'].items():
        d = r['diffs'].get(f'{D_} - NOMP')
        if d and d[2] < -0.01:
            h1['violations'].append(b)
    lowsnr = [-10, -5, 0, 5]
    g_off = float(np.mean([R['official']['per_snr'][D_][s_] - R['official']['per_snr']['NOMP'][s_] for s_ in lowsnr]))
    shr = {}
    for b in ['nosep9560', 'sep10_9561']:
        if b in R['prior']:
            shr[b] = float(np.mean([R['prior'][b]['per_snr'][D_][s_] - R['prior'][b]['per_snr']['NOMP'][s_] for s_ in lowsnr]))
    h1['official_gain_vs_NOMP_at_-10..5dB'] = g_off; h1['same_gain_on'] = shr
    h1['prior_exploit_label'] = bool(g_off > 0 and any(v < 0.5 * g_off for v in shr.values()))
    h1['PASS'] = not h1['violations']
    c['H1'] = h1
    lab = 'NULL'
    if c['P1']['PASS'] and c['S1']['PASS'] and c['S2']['PASS']:
        lab = 'FULL'
    elif c['S2']['PASS'] and (c['S1']['PASS'] or c['S1b']['PASS'] or c['S3']['PASS'] or c['S4'].get('PASS')):
        lab = 'REGIME_ONLY'
    c['OUTCOME'] = lab
    CRIT[D_] = c
    pr(f'=== CRITERIA design {D_}: outcome {lab}')
    for k, v in c.items():
        if k != 'OUTCOME':
            pr(f'   {k}: ' + ('PASS' if v.get('PASS') else 'FAIL') + '  ' + json.dumps({kk: vv for kk, vv in v.items() if kk != 'PASS'}, default=float)[:700])
R['criteria'] = CRIT

# ================================================================ figures
if DESIGNS:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    sn = list(range(-10, 26, 5))
    sty = {'A': ('C3', 'o', '-'), 'B': ('C1', 's', '-'), 'NOMP': ('k', 'x', '--'), 'NOMP os8 rc6': ('0.4', '+', '--'), 'classical skeleton': ('C2', '^', ':'),
           'lurd all-neural (cached)': ('C0', 'v', '-.'), 'U-Net (Lloria et al.)': ('C4', 'd', '-.')}
    for m, (c_, mk, ls) in sty.items():
        if m in R['official']['per_snr']:
            ax[0].plot(sn, [R['official']['per_snr'][m][s_] for s_ in sn], color=c_, marker=mk, ls=ls, label=m)
            if m != 'NOMP':
                ax[1].plot(sn, [R['official']['per_snr'][m][s_] - R['official']['per_snr']['NOMP'][s_] for s_ in sn], color=c_, marker=mk, ls=ls, label=m)
    ax[0].set_xlabel('SNR (dB)'); ax[0].set_ylabel('Pd (|err| <= 1 deg)'); ax[0].set_title('Official bank (8000 scenes, L = 3 given)'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].axhline(0, color='k', lw=.8); ax[1].set_xlabel('SNR (dB)'); ax[1].set_ylabel('Pd minus NOMP (os4 rc3)'); ax[1].set_title('Difference to NOMP'); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f'{HERE}/fig_pd_vs_snr.png', dpi=130); plt.close()
    cells = list(R['regime'])
    if cells:
        fig, ax = plt.subplots(figsize=(14, 4.5)); w = 0.16; x = np.arange(len(cells))
        series = [('A', 'C3'), ('B', 'C1'), ('best classical (pre-reg.)', '0.3'), ('classical skeleton', 'C2'), ('NOMP', '0.75')]
        for j, (nm, c_) in enumerate(series):
            v = []
            for cell in cells:
                r = R['regime'][cell]
                key = r['best_classical_prereg'] if nm.startswith('best') else nm
                v.append(r['pd'].get(key, np.nan))
            ax.bar(x + (j - 2) * w, v, w, color=c_, label=nm)
        ax.set_xticks(x); ax.set_xticklabels([c.replace('_snr', '\n') for c in cells], fontsize=7); ax.set_ylabel('Pd'); ax.legend(fontsize=8, ncol=5)
        ax.set_title('Regime cells (1000 fresh scenes each, L = 3 given; classical methods get the true SNR / impairment model)'); ax.grid(axis='y', alpha=.3)
        plt.tight_layout(); plt.savefig(f'{HERE}/fig_regimes.png', dpi=130); plt.close()
save_json(R, f'{HERE}/results.json')
pr('wrote results.json and figures')
