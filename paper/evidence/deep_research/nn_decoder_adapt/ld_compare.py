"""Paired comparison of the learned-decoder pipelines (this folder, ld_adapt_adc.py) against the ADC fine-tuning stage's per-scene stats
(finetune/adc/evaluate.py + supp.py: classical baselines incl. consistent NOMP, FT network only, FT dual+SC) on the SAME scenes.
The banks are regenerated from the same seeds (stress M5 / clean 5000, 5015; official bank first 250 scenes per SNR), so the NOMP per-scene
stats of both runs must be identical; this is asserted. Best classical = highest mean-over-SNR paper Pd among the classical methods on
this bank itself (favours the classical side). Paired bootstrap 2000 resamples within SNR groups, seed 0; '*' = 95% CI excludes 0.
Reads only; writes ld_compare.json (stdout -> ld_compare.log)."""
import os, sys, json
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import numpy as np
import common as C
PC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache'
MINE, ADC = f'{PC}/nn_decoder_adapt', f'{PC}/finetune_adc'
HERE = os.path.dirname(os.path.abspath(__file__))
CLASSICAL = ['NOMP', 'NOMP+SC', 'NOMP+SC tuned', 'FFT-peak seeds', 'FFT-peak seeds+SC', 'classical dual', 'classical dual+SC', 'NOMP os8 rc6',
             'multi-start K6', 'consistent NOMP', 'FFT parabolic os4', 'FFT parabolic os16', 'consistent parabolic']
REFS = ['BEST classical', 'NOMP', 'consistent NOMP', 'FT network only', 'FT dual+SC']
OURS = ['FT-CNN + adapted learned decoder', 'FT-CNN + original learned decoder', 'original CNN + original learned decoder',
        'FT-CNN + classical heat decoder']
banks = [(f'M5_b{b}_snr{s}', 'm5', f'b{b}_snr{s}') for s in (15, 0) for b in (4, 3, 2, 1)] + \
        [(f'clean_snr{s}', 'clean', f'clean_snr{s}') for s in (15, 0)] + [('official_subset', 'official', 'official')]
out = {}
for mine, aset, abank in banks:
    zm = np.load(f'{MINE}/stats_run__{mine}.npz'); za = np.load(f'{ADC}/stats2_{aset}__{abank}.npz')
    idx = np.load(f'{MINE}/official_idx.npy') if aset == 'official' else np.arange(len(zm['snr']))
    snr = zm['snr']; assert np.array_equal(za['snr'][idx].astype(int), snr.astype(int))
    S = {k: zm[k] for k in zm.files if k != 'snr'}
    for k in za.files:
        if k != 'snr':
            S['ADC:' + k] = za[k][idx]
    nomp_same = bool(np.array_equal(S['NOMP'], S['ADC:NOMP']))
    assert nomp_same, mine
    pdm = {k: C.metric_by_snr(v, snr)['mean']['pd_paper'] for k, v in S.items()}
    cls = [k for k in CLASSICAL if 'ADC:' + k in S]
    best = max(cls, key=lambda k: pdm['ADC:' + k])
    Bt = C.boot_metrics(S, snr, B=2000, seed=0)
    rec = dict(n=len(snr), nomp_identical=nomp_same, best_classical=best, pd={k.replace('ADC:', ''): v for k, v in pdm.items()}, diffs={})
    print(f'=== {mine} (n={len(snr)}) NOMP identical: {nomp_same}; best classical: {best} {pdm["ADC:" + best]:.4f}; '
          f'consistent NOMP {pdm.get("ADC:consistent NOMP", float("nan")):.4f}; FT network only {pdm["ADC:FT network only"]:.4f}; '
          f'FT dual+SC {pdm["ADC:FT dual+SC"]:.4f}')
    for a in OURS:
        line = f'   {a:42s} Pd {pdm[a]:.4f}'
        for r in REFS:
            rk = 'ADC:' + (best if r == 'BEST classical' else r)
            if rk not in S:
                continue
            d = Bt[a][:, 0] - Bt[rk][:, 0]
            v = [pdm[a] - pdm[rk], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
            rec['diffs'][f'{a} - {r}'] = v
            line += f' | vs {r} {v[0]:+.4f} [{v[1]:+.4f},{v[2]:+.4f}]{"*" if v[1] > 0 or v[2] < 0 else ""}'
        print(line, flush=True)
    out[mine] = rec
json.dump(out, open(f'{HERE}/ld_compare.json', 'w'), indent=1, default=float)
print('done')
