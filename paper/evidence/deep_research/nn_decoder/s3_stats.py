"""Step 3: metrics for every bank: paper-style Pd, strict Pd, source Pd, conditional RMSE (common.scene_stats / metric_by_snr = notebook metrics()),
95% paired-bootstrap percentile CIs (common.boot_metrics: 2000 resamples within SNR groups, seed 0, the same indices for all methods of a bank).
Methods: all-neural SC on/off (s2_predict), IABR-Net classical heat-seeded decoder (new_results/preds_heat, else verify/cache/heat__, else s2 heatcalc),
IABR-v2 dual (pred__IABR-v2), NOMP-only (pred__NOMP_no_network), U-Net (pred__U-Net_paper), network only (pred__v2_network_only_no_physics or verify/cache/netonly__).
Writes results.json, tables.md; per-SNR (official, T2_d*) with per-SNR bootstrap of the paired differences; fig_pd_vs_snr.png.
usage: python -u s3_stats.py > s3_stats.log"""
import os, sys, json
os.environ['OMP_NUM_THREADS'] = '6'
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import numpy as np
import common as C
HERE = os.path.dirname(os.path.abspath(__file__))
DRC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder'
VER = 'D:/ai_ml_project/paper/evidence/deep_research/verify'; NEWR = C.OUT; CACHE = C.CACHE
inv = json.load(open(f'{HERE}/s0_inventory.json'))
banks = [n for n, v in inv.items() if v is not None]
order = ['official', 'T2_d1', 'T2_d2', 'T2_d5'] + sorted(n for n in banks if n not in ('official', 'T2_d1', 'T2_d2', 'T2_d5'))
AN, AN0, HEAT, DUAL, NOMP, UNET, NET, RES = 'all-neural (SC on)', 'all-neural (SC off)', 'IABR-Net classical heat decoder', 'IABR-v2 dual', 'NOMP-only', 'U-Net', 'network only', 'ResNet (Lloria et al., port)'
DIFFS = [(AN, UNET), (AN, NOMP), (AN, HEAT), (AN0, UNET), (AN0, NOMP), (AN0, HEAT), (AN, AN0), (AN, RES)]


def first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def sources(n):
    return {AN: f'{DRC}/{n}__allneural_sc.npz', AN0: f'{DRC}/{n}__allneural_nosc.npz',
            HEAT: first(f'{NEWR}/preds_heat/{n}.npz', f'{VER}/cache/heat__{n}.npz', f'{DRC}/{n}__heatcalc.npz'),
            DUAL: first(f'{CACHE}/pred__IABR-v2__{n}.npz'), NOMP: first(f'{CACHE}/pred__NOMP_no_network__{n}.npz'),
            UNET: first(f'{CACHE}/pred__U-Net_paper__{n}.npz'), NET: first(f'{CACHE}/pred__v2_network_only_no_physics__{n}.npz', f'{VER}/cache/netonly__{n}.npz'), RES: first(f'{DRC}/resnet__{n}.npz')}


SDIR = f'{DRC}/scene_stats'; os.makedirs(SDIR, exist_ok=True)


def cached_stats(n, k, p, b):
    """per-scene metric components, cached on C: keyed by bank, method and source-file mtime (resumable; scene_stats is a slow python loop)"""
    f = f'{SDIR}/{n}__{k.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace(".", "")}__{int(os.path.getmtime(p))}.npy'
    if os.path.exists(f):
        return np.load(f)
    S = C.scene_stats(C.load_preds(p), b); np.save(f, S); return S


def fci(m, c): return f'{m:.4f} [{c[0]:.4f}, {c[1]:.4f}]'
def fd(d): return f'{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]'


res = {}; md = []
for n in order:
    b = C.load_bank(n); snr = b['snr']; src = {k: v for k, v in sources(n).items() if v is not None}
    S = {k: cached_stats(n, k, p, b) for k, p in src.items()}
    B = C.boot_metrics(S, snr, B=2000, seed=0); snrs = sorted(set(snr.astype(int).tolist()))
    r = dict(N=len(snr), L=sorted(set(b['L'].tolist())), snr_groups=snrs, sources=src, methods={}, diffs={})
    for m in S:
        per = C.metric_by_snr(S[m], snr)
        r['methods'][m] = dict(mean=per['mean'], ci={k: list(C.ci(B[m][:, j])[1:]) for j, k in enumerate(C.KEYS)}, dropped=int(S[m][:, 2].sum()),
                               per_snr={str(s): per[s] for s in snrs} if len(snrs) > 1 else None)
    for a, c in DIFFS:
        if a in S and c in S:
            d = B[a] - B[c]; ma, mc = r['methods'][a]['mean'], r['methods'][c]['mean']
            r['diffs'][f'{a} - {c}'] = {k: [ma[k] - mc[k], *C.ci(d[:, j])[1:]] for j, k in enumerate(C.KEYS)}
    if len(snrs) > 1:     # per-SNR paired bootstrap of the differences (single group each)
        r['per_snr_diffs'] = {}
        for s in snrs:
            ix = np.where(snr.astype(int) == s)[0]
            Bs = C.boot_metrics({k: v[ix] for k, v in S.items()}, snr[ix], B=2000, seed=0)
            r['per_snr_diffs'][str(s)] = {f'{a} - {c}': [r['methods'][a]['per_snr'][str(s)]['pd_paper'] - r['methods'][c]['per_snr'][str(s)]['pd_paper'],
                                                         *C.ci((Bs[a] - Bs[c])[:, 0])[1:]] for a, c in DIFFS if a in S and c in S}
    res[n] = r
    a = r['methods'][AN]['mean']; a0 = r['methods'][AN0]['mean']
    line = f'{n:22s} N={len(snr):5d} L={r["L"]} AN-SC {a["pd_paper"]:.4f} AN-noSC {a0["pd_paper"]:.4f}'
    for m in [HEAT, DUAL, NOMP, UNET, RES, NET]:
        if m in S:
            line += f' | {m.split(" ")[0]} {r["methods"][m]["mean"]["pd_paper"]:.4f}'
    for k in [f'{AN} - {UNET}', f'{AN} - {NOMP}', f'{AN} - {HEAT}']:
        if k in r['diffs']:
            line += f' | d[{k.split(" - ")[1].split(" ")[0]}] {fd(r["diffs"][k]["pd_paper"])}'
    print(line, flush=True)

    md.append(f'### {n}  (N={r["N"]}, L={r["L"]}, SNR groups={snrs})\n\n| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped |\n|---|---|---|---|---|')
    for m in [AN, AN0, HEAT, DUAL, NOMP, UNET, RES, NET]:
        if m in r['methods']:
            q = r['methods'][m]
            md.append(f'| {m} | {fci(q["mean"]["pd_paper"], q["ci"]["pd_paper"])} | {fci(q["mean"]["pd_strict"], q["ci"]["pd_strict"])} | {fci(q["mean"]["rmse"], q["ci"]["rmse"])} | {q["dropped"]} |')
    md.append('\n| paired difference | d Pd paper-style | d Pd strict | d RMSE deg |\n|---|---|---|---|')
    for k, d in r['diffs'].items():
        md.append(f'| {k} | {fd(d["pd_paper"])} | {fd(d["pd_strict"])} | {fd(d["rmse"])} |')
    md.append('')


def group(n):
    if n.startswith('tune_'):
        return 'tuning banks (used to tune other stages)'
    return 'L=3 test banks' if res[n]['L'] == [3] else 'L!=3 test banks'


counts = {}
for n in order:
    gname = group(n)
    for k, d in res[n]['diffs'].items():
        for metric in ['pd_paper', 'pd_strict']:
            v = d[metric]; c = counts.setdefault(gname, {}).setdefault(k, {}).setdefault(metric, dict(above=[], below=[], contains0=[]))
            c['above' if v[1] > 0 else 'below' if v[2] < 0 else 'contains0'].append(n)
print('\nBank counts (95% CI of the paired difference above 0 / below 0 / containing 0):', flush=True)
mdc = ['| bank group | difference | metric | CI > 0 | CI < 0 | contains 0 |', '|---|---|---|---|---|---|']
for gname, dd in counts.items():
    for k, mm in dd.items():
        for metric, c in mm.items():
            print(f'  {gname:42s} {k:58s} {metric:9s} above {len(c["above"]):2d} below {len(c["below"]):2d} contains0 {len(c["contains0"]):2d}   above={c["above"]}', flush=True)
            mdc.append(f'| {gname} | {k} | {metric} | {len(c["above"])} | {len(c["below"])} | {len(c["contains0"])} |')

mds = []
for n in ['official', 'T2_d1', 'T2_d2', 'T2_d5']:
    r = res[n]; ms = [m for m in [AN, AN0, HEAT, DUAL, NOMP, UNET, RES, NET] if m in r['methods']]
    mds.append(f'#### {n}: paper-style Pd per SNR (U-Net strict Pd in parentheses)\n\n| SNR dB | ' + ' | '.join(ms) + f' | {AN} - NOMP [95% CI] | {AN} - U-Net [95% CI] |\n|' + '---|' * (len(ms) + 3))
    for s in r['snr_groups']:
        row = [f'{r["methods"][m]["per_snr"][str(s)]["pd_paper"]:.4f}' + (f' ({r["methods"][m]["per_snr"][str(s)]["pd_strict"]:.4f})' if m == UNET else '') for m in ms]
        dd = r['per_snr_diffs'][str(s)]
        mds.append(f'| {s} | ' + ' | '.join(row) + f' | {fd(dd[f"{AN} - {NOMP}"])} | {fd(dd[f"{AN} - {UNET}"])} |')
    mds.append('')
print('\n' + '\n'.join(mds), flush=True)
json.dump(dict(banks=res, counts=counts), open(f'{HERE}/results.json', 'w'), indent=1, default=float)
open(f'{HERE}/tables.md', 'w').write('## Per-bank tables\n\n' + '\n'.join(md) + '\n\n## Bank counts\n\n' + '\n'.join(mdc) + '\n\n## Per-SNR tables\n\n' + '\n'.join(mds))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
r = res['official']; s = r['snr_groups']
sty = {AN: ('C3', 'o', '-'), AN0: ('C1', 's', '--'), HEAT: ('C2', '^', '-'), DUAL: ('C4', 'v', '-'), NOMP: ('k', 'D', '-'), UNET: ('C0', 'x', '-'), NET: ('0.5', '.', ':'), RES: ('C5', '+', '-')}
for m, (c, mk, ls) in sty.items():
    if m in r['methods']:
        ax[0].plot(s, [r['methods'][m]['per_snr'][str(x)]['pd_paper'] for x in s], color=c, marker=mk, ls=ls, label=m)
ax[0].plot(s, [r['methods'][UNET]['per_snr'][str(x)]['pd_strict'] for x in s], color='C0', marker='x', ls=':', label='U-Net (strict Pd)')
ax[0].set_xlabel('SNR (dB)'); ax[0].set_ylabel('Pd (paper-style, |err| <= 1 deg)'); ax[0].set_title('official bank (N=8000, L=3)'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=7)
for k, c in [(f'{AN} - {NOMP}', 'k'), (f'{AN} - {UNET}', 'C0'), (f'{AN} - {HEAT}', 'C2')]:
    v = np.array([r['per_snr_diffs'][str(x)][k] for x in s])
    ax[1].errorbar(s, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=c, marker='o', capsize=3, label=k.replace(f'{AN} - ', 'all-neural - '))
ax[1].axhline(0, color='0.3', lw=.8); ax[1].set_xlabel('SNR (dB)'); ax[1].set_ylabel('paired d Pd (95% CI)'); ax[1].set_title('all-neural (SC on) minus reference'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{HERE}/fig_pd_vs_snr.png', dpi=130)
print('done', flush=True)
