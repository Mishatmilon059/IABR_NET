"""Summary of a3_test (fresh seeds 9300-9307 + official bank): % identifiable MSE removed (and RMS) at SNR >= 5 with paired bootstrap
CIs (v2common.boot_removed: 2000 resamples within SNR groups, seed 0), residual RMS per SNR, paper/strict Pd and paired Pd differences
(new_results/common.boot_metrics, 2000 within SNR, seed 0; * = 95% CI excludes 0). Comparison against the original e1_eval numbers
(iabc_v2/results.json is not re-read; the original values are quoted from s_summary.log in VERDICT.md). stdout -> a5_summary.log,
machine-readable -> a5_results.json"""
import os, json, sys
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import common as C
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/iabc_v2_adv'
DB = 20 / np.log(10); SNRS = [-10, -5, 0, 5, 10, 15, 20, 25]
CELLS = ['clean', 'ph8', 'g3', 'c8_3', 'ph20', 'g6', 'c30_6', 'ph45']
EST = ['v2b', 'v2b+SC tuned', 'SC tuned', 'SC tuned 10it', 'SC open', 'SC redetect', 'EB-ML', 'EB-ML-g5', 'EB-ML-g10', 'EB-ML-g5+SC tuned']
COMP = {'clean': [], 'ph8': [0], 'g3': [1], 'c8_3': [0, 1], 'ph20': [0], 'g6': [1], 'c30_6': [0, 1], 'ph45': [0]}
DIFFS = [('v2b+NOMP+SC tuned', 'NOMP+SC tuned'), ('v2b+NOMP', 'NOMP'), ('v2b+NOMP+SC tuned', 'NOMP'), ('NOMP+SC tuned', 'NOMP'),
         ('NOMP+SC tuned 10it', 'NOMP+SC tuned'), ('NOMP+SC redetect', 'NOMP+SC tuned'), ('EB-ML+NOMP', 'NOMP+SC tuned'),
         ('EB-ML-g5+NOMP', 'NOMP+SC tuned'), ('EB-ML-g10+NOMP', 'NOMP+SC tuned'), ('EB-ML-g5+NOMP+SC tuned', 'NOMP+SC tuned'),
         ('v2b+NOMP+SC tuned', 'EB-ML-g5+NOMP+SC tuned'), ('v2b+NOMP', 'EB-ML-g5+NOMP'), ('v2b+NOMP+SC tuned', 'EB-ML-g5+NOMP')]


def boot_removed(mb, ma, snr, B=2000, seed=0):
    rng = np.random.default_rng(seed); snr = snr.astype(int); nb = np.zeros(B); na = np.zeros(B)
    for s in sorted(set(snr.tolist())):
        ix = np.where(snr == s)[0]; R = rng.integers(0, len(ix), (B, len(ix))); nb += mb[ix][R].sum(1); na += ma[ix][R].sum(1)
    v = 100 * (1 - na / np.maximum(nb, 1e-30))
    return 100 * (1 - ma.sum() / max(mb.sum(), 1e-30)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def f3(c):
    return f'{c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]' + ('*' if c[1] > 0 or c[2] < 0 else '')


R = dict(cells={}, bank={})
print('# Adversarial re-test of IABC-v2 (fresh seeds 9300-9307, 200 scenes x 8 SNR per cell, L=3)\n')
print('## 1. % identifiable MSE removed at SNR >= 5 dB [95% CI]  (RMS % removed in parentheses)')
print('| cell | comp | before RMS | ' + ' | '.join(EST) + ' |'); print('|---|---|---|' + '---|' * len(EST))
Z = {}
for c in CELLS:
    fn = f'{OUT}/cell__{c}.npz'
    if not os.path.exists(fn):
        continue
    z = np.load(fn); Z[c] = z; snr = z['snr']; hi = snr >= 5; R['cells'][c] = dict(removed={}, rms_by_snr={}, pd={}, diffs={})
    for j in COMP[c]:
        k = 'mp' if j == 0 else 'mg'; unit = 'deg' if j == 0 else 'dB'
        b0 = z[f'{k}_none'][hi]; brms = (np.rad2deg if j == 0 else (lambda x: DB * x))(np.sqrt(b0.mean()))
        cells = []
        for m in EST:
            p = boot_removed(b0, z[f'{k}_{m}'][hi], snr[hi]); rr = 100 * (1 - np.sqrt(1 - p[0] / 100)) if p[0] < 100 else 100
            cells.append(f'{p[0]:.1f} [{p[1]:.1f},{p[2]:.1f}] ({rr:.0f})'); R['cells'][c]['removed'][f'{m}|{k}'] = p
        print(f'| {c} | {"phase" if j == 0 else "gain"} | {brms:.2f} {unit} | ' + ' | '.join(cells) + ' |')

print('\n## 2. Residual identifiable RMS per SNR (phase deg / gain dB) for the headline cells; clean = added error')
for c in ['clean', 'ph8', 'g3', 'c8_3', 'c30_6', 'ph45']:
    if c not in Z:
        continue
    z = Z[c]; snr = z['snr']
    print(f'\n{c}: | method | ' + ' | '.join(str(s) for s in SNRS) + ' |')
    for m in ['none', 'v2b', 'SC tuned', 'SC open', 'EB-ML', 'EB-ML-g5', 'v2b+SC tuned']:
        row = [f'{np.rad2deg(np.sqrt(z["mp_" + m][snr == s].mean())):.2f}/{DB * np.sqrt(z["mg_" + m][snr == s].mean()):.2f}' for s in SNRS]
        R['cells'][c]['rms_by_snr'][m] = row
        print(f'| {m} | ' + ' | '.join(row) + ' |')

print('\n## 3. Paper Pd (mean over 8 SNRs; strict in parentheses)')
PDM = None
for c in CELLS:
    if c not in Z:
        continue
    z = Z[c]; snr = z['snr']; ms = [str(m) for m in z['methods']]
    if PDM is None:
        PDM = ms; print('| cell | ' + ' | '.join(PDM) + ' |'); print('|---|' + '---|' * len(PDM))
    S_ = {m: z[f'S_{m}'] for m in ms}; per = {m: C.metric_by_snr(S_[m], snr) for m in ms}
    R['cells'][c]['pd'] = {m: dict(paper=per[m]['mean']['pd_paper'], strict=per[m]['mean']['pd_strict'], by_snr=[per[m][s]['pd_paper'] for s in SNRS]) for m in ms}
    print(f'| {c} | ' + ' | '.join(f'{per[m]["mean"]["pd_paper"]:.3f} ({per[m]["mean"]["pd_strict"]:.3f})' for m in PDM) + ' |')
    bt = C.boot_metrics(S_, snr)
    R['cells'][c]['diffs'] = {f'{a} - {b}': dict(paper=C.ci(bt[a][:, 0] - bt[b][:, 0]), strict=C.ci(bt[a][:, 1] - bt[b][:, 1])) for a, b in DIFFS}

print('\n## 4. Paired paper-Pd differences [95% CI] (* = CI excludes 0)')
print('| cell | ' + ' | '.join(f'{a} - {b}' for a, b in DIFFS) + ' |'); print('|---|' + '---|' * len(DIFFS))
for c in CELLS:
    if c in Z:
        print(f'| {c} | ' + ' | '.join(f3(R['cells'][c]['diffs'][f'{a} - {b}']['paper']) for a, b in DIFFS) + ' |')
print('\n## 4b. Same, strict Pd (first three differences)')
for c in CELLS:
    if c in Z:
        print(f'| {c} | ' + ' | '.join(f3(R['cells'][c]['diffs'][f'{a} - {b}']['strict']) for a, b in DIFFS[:3]) + ' |')

fn = f'{OUT}/bank__official.npz'
if os.path.exists(fn):
    z = np.load(fn); snr = z['snr']; ms = [str(m) for m in z['methods']]; S_ = {m: z[f'S_{m}'] for m in ms}
    pdv = {m: C.metric_by_snr(S_[m], snr)['mean']['pd_paper'] for m in ms}; bt = C.boot_metrics(S_, snr)
    BD = [('v2b+NOMP+SC tuned', 'NOMP'), ('v2b+NOMP', 'NOMP'), ('NOMP+SC tuned', 'NOMP'), ('v2b+NOMP+SC tuned', 'NOMP+SC tuned'),
          ('NOMP+SC tuned 10it', 'NOMP'), ('NOMP+SC redetect', 'NOMP'), ('EB-ML+NOMP', 'NOMP'), ('EB-ML-g5+NOMP', 'NOMP'),
          ('EB-ML-g10+NOMP', 'NOMP'), ('EB-ML-g5+NOMP+SC tuned', 'NOMP')]
    R['bank'] = dict(N=int(len(snr)), pd=pdv, diffs={f'{a} - {b}': C.ci(bt[a][:, 0] - bt[b][:, 0]) for a, b in BD},
                     added_rms={m: [float(np.rad2deg(np.sqrt(z['mp_' + m].mean()))), float(DB * np.sqrt(z['mg_' + m].mean()))] for m in EST})
    print(f'\n## 5. Official bank (N={len(snr)}, ideal hardware, true L) paper Pd')
    print(' | '.join(f'{m} {v:.4f}' for m, v in pdv.items()))
    for k, v in R['bank']['diffs'].items():
        print(f'   {k}: {f3(v)}')
    print('   added identifiable RMS on ideal hardware (deg / dB): ' + ' | '.join(f'{m} {a:.2f}/{b:.2f}' for m, (a, b) in R['bank']['added_rms'].items()))
json.dump(R, open(f'{HERE}/a5_results.json', 'w'), indent=0, default=float)
print('\nwrote a5_results.json')
