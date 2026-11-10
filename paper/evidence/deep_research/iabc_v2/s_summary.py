"""Summary of e1_eval caches -> tables (stdout = s_summary.log), results.json, figures. Paired bootstrap: 2000 resamples within SNR
groups, seed 0 (C.boot_metrics for Pd; v2common.boot_removed for % MSE removed). Criteria from criteria.json (pre-registered)."""
import os
os.environ['OMP_NUM_THREADS'] = '4'
import json, sys
import numpy as np
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import common as C
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from v2common import boot_removed, V2, V2CACHE, SNRS, DB

crb = json.load(open(f'{V2}/crb.json')); crit = json.load(open(f'{V2}/criteria.json'))
CELLS = ['clean', 'ph5', 'ph8', 'ph20', 'ph30', 'ph45', 'g3', 'g6', 'c8_3', 'c20_4', 'c30_6']
IMP = [c for c in CELLS if c != 'clean']
EST = ['oldIABC', 'v2a', 'v2b', 'Toeplitz', 'SC tuned', 'SC open', 'SC snrgate', 'oracle ALS', 'v2a+SC tuned', 'v2b+SC tuned']
Z = {c: np.load(f'{V2CACHE}/eval__{c}.npz') for c in CELLS if os.path.exists(f'{V2CACHE}/eval__{c}.npz')}
CELLS = [c for c in CELLS if c in Z]; IMP = [c for c in IMP if c in Z]
R = dict(cells={}, banks={}, crb=crb)
comps = lambda z: [j for j, k in [(0, 'phase_deg'), (1, 'gain_db')] if float(z[k]) > 0]
PR = 'v2b+NOMP+SC tuned'


def fmt(t):
    return f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]' + ('*' if t[1] > 0 or t[2] < 0 else '')


print('# IABC-v2 evaluation summary\n')
# ---------------------------------------------------------------- residual error
print('## 1. Identifiable per-antenna error: % of pooled MSE removed [95% CI] (L=3, 250 scenes x 8 SNR per cell)')
print('| cell | comp | before RMS | ' + ' | '.join(EST) + ' |'); print('|---|---|---|' + '---|' * len(EST))
for c in IMP:
    z = Z[c]; snr = z['snr']; R['cells'][c] = dict(removed={}, removed_snr_ge5={}, rms_by_snr={}, pd={}, diffs={})
    for j in comps(z):
        key = 'mp_' if j == 0 else 'mg_'; nm = 'phase' if j == 0 else 'gain'
        b = z[key + 'none']; br = (np.rad2deg(np.sqrt(b.mean())) if j == 0 else DB * np.sqrt(b.mean()))
        cells = []
        for e in EST:
            t = boot_removed(b, z[key + e], snr); R['cells'][c]['removed'][f'{nm}|{e}'] = t; cells.append(f'{t[0]:.1f} [{t[1]:.1f},{t[2]:.1f}]')
        print(f'| {c} | {nm} | {br:.2f} {"deg" if j == 0 else "dB"} | ' + ' | '.join(cells) + ' |')
print('\n## 1b. Same, SNR >= 5 dB only (criterion i: >= 75% MSE = RMS halved); RMS before -> after for v2b')
print('| cell | comp | ' + ' | '.join(EST) + ' | v2b RMS before -> after | approx-Bayes bound (RMS % removed) |'); print('|---|---|' + '---|' * len(EST) + '---|---|')
for c in IMP:
    z = Z[c]; q = z['snr'] >= 5
    for j in comps(z):
        key = 'mp_' if j == 0 else 'mg_'; nm = 'phase' if j == 0 else 'gain'; b = z[key + 'none'][q]
        cells = []
        for e in EST:
            t = boot_removed(b, z[key + e][q], z['snr'][q]); R['cells'][c]['removed_snr_ge5'][f'{nm}|{e}'] = t; cells.append(f'{t[0]:.1f} [{t[1]:.1f},{t[2]:.1f}]')
        f = (lambda m: np.rad2deg(np.sqrt(m.mean()))) if j == 0 else (lambda m: DB * np.sqrt(m.mean()))
        bd = crit['approx_bayes_bound_snr_ge5'].get(c, {}).get(nm, {}).get('bound_rms_removed_pct', np.nan)
        print(f'| {c} | {nm} | ' + ' | '.join(cells) + f' | {f(b):.2f} -> {f(z[key + "v2b"][q]):.2f} | {bd:.1f}% |')
print('\n## 1c. Residual identifiable RMS by SNR (sqrt of pooled MSE; phase deg / gain dB) vs CRB (unknown angles) and approx-Bayes bound')
for c in IMP:
    z = Z[c]
    for j in comps(z):
        key = 'mp_' if j == 0 else 'mg_'; nm = 'phase' if j == 0 else 'gain'; f = (lambda m: np.rad2deg(np.sqrt(m.mean()))) if j == 0 else (lambda m: DB * np.sqrt(m.mean()))
        print(f'\n{c} {nm}: | method | ' + ' | '.join(f'{s}' for s in SNRS) + ' |')
        rows = {}
        for e in ['none'] + EST:
            rows[e] = [f(z[key + e][z['snr'] == s]) for s in SNRS]; print(f'| {e} | ' + ' | '.join(f'{x:.2f}' for x in rows[e]) + ' |')
        rows['CRB'] = crb['clean point']['unknown']['phase_deg' if j == 0 else 'gain_db']
        rows['approx Bayes bound'] = crb['cells'][c]['phase_deg' if j == 0 else 'gain_db']
        for e in ['CRB', 'approx Bayes bound']:
            print(f'| {e} | ' + ' | '.join(f'{x:.2f}' for x in rows[e]) + ' |')
        R['cells'][c]['rms_by_snr'][nm] = rows
# clean cell: error ADDED
z = Z['clean']
print('\n## 1d. Clean scenes (ideal hardware): identifiable error ADDED by each estimator, RMS (deg / dB) at SNR -10, 0, 10, 25 and all')
for e in EST:
    a = [f'{np.rad2deg(np.sqrt(z["mp_" + e][z["snr"] == s].mean())):.2f}/{DB * np.sqrt(z["mg_" + e][z["snr"] == s].mean()):.2f}' for s in [-10, 0, 10, 25]]
    print(f'| {e} | ' + ' | '.join(a) + f' | all {np.rad2deg(np.sqrt(z["mp_" + e].mean())):.2f}/{DB * np.sqrt(z["mg_" + e].mean()):.2f} |')
# ---------------------------------------------------------------- Pd
PDM = ['NOMP', 'oldIABC+NOMP', 'v2a+NOMP', 'v2b+NOMP', 'Toeplitz+NOMP', 'NOMP+SC tuned', 'NOMP+SC snrgate', 'v2a+NOMP+SC tuned', 'v2b+NOMP+SC tuned',
       'IABR-Net heat', 'IABR-Net heat (v2a swapped)', 'IABR-Net heat (v2b swapped)', 'CEILING', 'FULL']
print('\n## 2. Paper Pd, mean over SNR (strict Pd in brackets), L=3, 2000 scenes per cell')
print('| cell | ' + ' | '.join(PDM) + ' |'); print('|---|' + '---|' * len(PDM))
DIFFS = [('v2b+NOMP+SC tuned', 'NOMP+SC tuned'), ('v2b+NOMP', 'NOMP+SC tuned'), ('v2a+NOMP+SC tuned', 'NOMP+SC tuned'), ('v2b+NOMP+SC tuned', 'NOMP+SC snrgate'),
         ('v2b+NOMP', 'NOMP'), ('v2a+NOMP', 'NOMP'), ('oldIABC+NOMP', 'NOMP'), ('IABR-Net heat (v2b swapped)', 'IABR-Net heat'), ('IABR-Net heat (v2b swapped)', 'NOMP+SC tuned'),
         ('CEILING', 'NOMP+SC tuned'), ('CEILING', 'NOMP'), ('FULL', 'CEILING')]
BT = {}
for c in CELLS:
    z = Z[c]; snr = z['snr']; S_ = {m: z[f'S_{m}'] for m in PDM}
    per = {m: C.metric_by_snr(S_[m], snr) for m in PDM}
    bt = C.boot_metrics(S_, snr); BT[c] = bt
    R['cells'].setdefault(c, dict(pd={}, diffs={}))
    R['cells'][c]['pd'] = {m: dict(paper=per[m]['mean']['pd_paper'], strict=per[m]['mean']['pd_strict'], by_snr=[per[m][s]['pd_paper'] for s in SNRS]) for m in PDM}
    R['cells'][c]['diffs'] = {f'{a} - {b}': dict(paper=C.ci(bt[a][:, 0] - bt[b][:, 0]), strict=C.ci(bt[a][:, 1] - bt[b][:, 1])) for a, b in DIFFS}
    print(f'| {c} | ' + ' | '.join(f'{per[m]["mean"]["pd_paper"]:.3f} ({per[m]["mean"]["pd_strict"]:.3f})' for m in PDM) + ' |')
print('\n## 2b. Paired differences, paper Pd mean over SNR [95% CI] (* = CI excludes 0)')
print('| cell | ' + ' | '.join(f'{a} - {b}' for a, b in DIFFS) + ' |'); print('|---|' + '---|' * len(DIFFS))
for c in CELLS:
    print(f'| {c} | ' + ' | '.join(fmt(R['cells'][c]['diffs'][f'{a} - {b}']['paper']) for a, b in DIFFS) + ' |')
print('\n## 2c. Same, strict Pd')
print('| cell | ' + ' | '.join(f'{a} - {b}' for a, b in DIFFS[:5]) + ' |'); print('|---|' + '---|' * 5)
for c in CELLS:
    print(f'| {c} | ' + ' | '.join(fmt(R['cells'][c]['diffs'][f'{a} - {b}']['strict']) for a, b in DIFFS[:5]) + ' |')
print('\n## 2d. Per-SNR paper Pd: NOMP / NOMP+SC tuned / v2b+NOMP+SC tuned / CEILING / FULL')
for c in CELLS:
    p = R['cells'][c]['pd']
    print(f'{c:6s} ' + ' '.join(f'{s}:{p["NOMP"]["by_snr"][i]:.3f}/{p["NOMP+SC tuned"]["by_snr"][i]:.3f}/{p[PR]["by_snr"][i]:.3f}/{p["CEILING"]["by_snr"][i]:.3f}/{p["FULL"]["by_snr"][i]:.3f}' for i, s in enumerate(SNRS)))
# ---------------------------------------------------------------- banks
print('\n## 3. Official bank (8000, ideal hardware) and cached v1 banks: paper Pd and paired differences')
BPD = ['NOMP', 'oldIABC+NOMP', 'v2a+NOMP', 'v2b+NOMP', 'NOMP+SC tuned', 'v2a+NOMP+SC tuned', 'v2b+NOMP+SC tuned', 'IABR-Net heat', 'IABR-Net heat (v2b swapped)']
BD = [('v2b+NOMP+SC tuned', 'NOMP'), ('v2b+NOMP+SC tuned', 'NOMP+SC tuned'), ('v2b+NOMP', 'NOMP'), ('v2a+NOMP', 'NOMP'), ('oldIABC+NOMP', 'NOMP')]
for f in sorted(os.listdir(V2CACHE)):
    if not f.startswith('eval__bank_'):
        continue
    bank = f[len('eval__bank_'):-4]; z = np.load(f'{V2CACHE}/{f}'); snr = z['snr']; S_ = {m: z[f'S_{m}'] for m in BPD}
    pdv = {m: C.metric_by_snr(S_[m], snr)['mean']['pd_paper'] for m in BPD}; bt = C.boot_metrics(S_, snr)
    dd = {f'{a} - {b}': C.ci(bt[a][:, 0] - bt[b][:, 0]) for a, b in BD}
    R['banks'][bank] = dict(N=int(len(snr)), pd=pdv, diffs=dd, added_rms={k: [float(z['rp_' + k].mean()), float(z['rg_' + k].mean())] for k in ['oldIABC', 'v2a', 'v2b', 'Toeplitz']})
    print(f'{bank:20s} N={len(snr):5d} ' + ' '.join(f'{m} {v:.4f}' for m, v in pdv.items()))
    print('     ' + ' | '.join(f'{k}: {fmt(v)}' for k, v in dd.items()))
# ---------------------------------------------------------------- criteria
print('\n## 4. Pre-registered decision criteria (criteria.json)')
ci_ = []; fails_i = []; infeas = []
for c in ['ph8', 'ph20', 'ph30', 'ph45', 'g3', 'g6', 'c8_3', 'c20_4', 'c30_6']:
    if c not in Z:
        continue
    for j in comps(Z[c]):
        nm = 'phase' if j == 0 else 'gain'; t = R['cells'][c]['removed_snr_ge5'][f'{nm}|v2b']
        rms_rem = 100 * (1 - np.sqrt(1 - t[0] / 100)); bd = crit['approx_bayes_bound_snr_ge5'][c][nm]['bound_rms_removed_pct']
        ok = rms_rem >= 50; status = 'PASS' if ok else ('INFEASIBLE (bound < 50%)' if bd < 50 else 'FAIL')
        if status == 'FAIL':
            fails_i.append(f'{c}/{nm}')
        if status.startswith('INF'):
            infeas.append(f'{c}/{nm}')
        print(f'(i) {c:6s} {nm:5s}: v2b removes {t[0]:.1f}% MSE = {rms_rem:.1f}% RMS at SNR>=5 (bound {bd:.1f}% RMS) -> {status}')
pos = [c for c in IMP if R['cells'][c]['diffs'][f'{PR} - NOMP+SC tuned']['paper'][1] > 0]
neg = [c for c in IMP if R['cells'][c]['diffs'][f'{PR} - NOMP+SC tuned']['paper'][2] < 0]
ok_ii = len(pos) >= 5 and not neg
print(f'(ii) {PR} - NOMP+SC tuned: CI > 0 in {len(pos)}/{len(IMP)} impaired cells {pos}; CI < 0 in {neg} -> {"PASS" if ok_ii else "FAIL"}')
off = R['banks'].get('official')
ok_iii = off is not None and off['diffs'][f'{PR} - NOMP'][2] >= 0
if off:
    print(f'(iii) official: {PR} - NOMP = {fmt(off["diffs"][f"{PR} - NOMP"])}; vs NOMP+SC tuned {fmt(off["diffs"][f"{PR} - NOMP+SC tuned"])} -> {"PASS" if ok_iii else "FAIL"}')
ok_i = not fails_i
print(f'(i) overall: {"PASS" if ok_i else "FAIL"} (fails: {fails_i}; infeasible: {infeas})')
dec = 'KEEP / REPLACE with IABC-v2' if (ok_i and ok_ii and ok_iii) else 'DROP from the claims (optional module at most)'
print(f'DECISION by the pre-registered rule: {dec}')
R['criteria'] = dict(i=dict(pass_=ok_i, fails=fails_i, infeasible=infeas), ii=dict(pass_=ok_ii, pos=pos, neg=neg), iii=dict(pass_=ok_iii), decision=dec)
# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
panels = [('ph8', 0), ('ph20', 0), ('ph45', 0), ('g3', 1), ('g6', 1), ('c30_6', 1)]
sty = {'none': ('k', '-', 'no correction'), 'oldIABC': ('tab:gray', '-', 'old IABC'), 'v2a': ('tab:blue', '-', 'IABC-v2a (blind)'), 'v2b': ('tab:red', '-', 'IABC-v2b (hybrid)'),
       'Toeplitz': ('tab:olive', ':', 'Toeplitz (classical)'), 'SC tuned': ('tab:green', '-', 'ALS SC tuned'), 'SC open': ('tab:green', ':', 'ALS SC gate-open'),
       'oracle ALS': ('tab:purple', '--', 'oracle ALS (true angles)'), 'CRB': ('k', '--', 'CRB (unbiased)'), 'approx Bayes bound': ('k', ':', 'approx. Bayes bound')}
for a, (c, j) in zip(ax.ravel(), panels):
    if c not in R['cells']:
        continue
    nm = 'phase' if j == 0 else 'gain'; rows = R['cells'][c]['rms_by_snr'][nm]
    for k, (col, ls, lab) in sty.items():
        a.semilogy(SNRS, rows[k], color=col, ls=ls, marker='o' if k in ('v2a', 'v2b') else None, ms=3, label=lab)
    a.set_title(f'{c}: identifiable {nm} error'); a.set_xlabel('SNR (dB)'); a.set_ylabel('RMS ' + ('deg' if j == 0 else 'dB')); a.grid(alpha=.3, which='both')
ax[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{V2}/fig_residual_vs_snr.png', dpi=130); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
series = [('phase error (deg)', [('clean', 0), ('ph5', 5), ('ph8', 8), ('ph20', 20), ('ph30', 30), ('ph45', 45)]), ('gain error (dB)', [('clean', 0), ('g3', 3), ('g6', 6)]),
          ('combined (deg / dB)', [('clean', 0), ('c8_3', 1), ('c20_4', 2), ('c30_6', 3)])]
for a, (xl, ser) in zip(ax, series):
    ser = [s for s in ser if s[0] in R['cells']]; x = [s[1] for s in ser]
    for m, col, ls in [('NOMP', 'k', '-'), ('NOMP+SC tuned', 'tab:green', '-'), ('NOMP+SC snrgate', 'tab:green', ':'), ('oldIABC+NOMP', 'tab:gray', '-'), ('IABR-Net heat', 'tab:orange', '-'),
                       ('v2a+NOMP', 'tab:blue', '-'), ('v2b+NOMP', 'tab:red', '--'), (PR, 'tab:red', '-'), ('CEILING', 'tab:purple', '--'), ('FULL', 'tab:purple', ':')]:
        a.plot(x, [R['cells'][s[0]]['pd'][m]['paper'] for s in ser], color=col, ls=ls, marker='o', ms=3, label=m)
    a.set_xlabel(xl); a.set_ylabel('paper Pd (mean over SNR)'); a.grid(alpha=.3)
    if xl.startswith('combined'):
        a.set_xticks([0, 1, 2, 3]); a.set_xticklabels(['clean', '8/3', '20/4', '30/6'])
ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{V2}/fig_pd_vs_severity.png', dpi=130); plt.close()
# training curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
h = json.load(open(f'{V2}/train_v2a_history.json')); tr = [(r['step'], r['train']) for r in h]; va = [(r['step'], r['val']) for r in h if 'val' in r]
ax[0].semilogy(*zip(*tr), lw=.8, label='train (50-step mean)'); ax[0].semilogy(*zip(*va), 'o-', label='held-out val (seed 62999999)'); ax[0].set_title('IABC-v2a loss (MSE rad^2 / neper^2)'); ax[0].legend(); ax[0].set_xlabel('step')
if os.path.exists(f'{V2}/train_v2b_history.json'):
    h = json.load(open(f'{V2}/train_v2b_history.json'))
    ax[1].semilogy([r['step'] for r in h], [r['train'] for r in h], label='train'); ax[1].semilogy([r['step'] for r in h], [r['val'] for r in h], 'o-', label='val (seed 63999999)')
    ax[1].set_title('IABC-v2b refiner loss'); ax[1].legend(); ax[1].set_xlabel('step')
plt.tight_layout(); plt.savefig(f'{V2}/fig_training.png', dpi=130); plt.close()
json.dump(R, open(f'{V2}/results.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
print('\nwrote results.json, fig_residual_vs_snr.png, fig_pd_vs_severity.png, fig_training.png')
