"""Assembles RESULTS_NEW.md from the raw outputs (bank_json/*.json via e13_e6_tables.md, e4/*.json, e5_output.txt, e5_results.json, e8_latency_threads*.json).
All numbers in the generated tables are read from files; the narrative sentences quote numbers that are re-checked by check_results.py."""
import json, os
OUT = 'D:/ai_ml_project/paper/evidence/new_results'
L = []
w = L.append
w('''# RESULTS_NEW: new CPU experiments for the paper (E13+E6, E4, E5, E8)

Folder: `D:\\ai_ml_project\\paper\\evidence\\new_results\\`. Nothing outside this folder was modified. Every number below was produced by a script in this folder; the raw outputs are listed per section. Machine: ASUS VivoBook X515EP, Intel Core i5-1135G7 (4 cores / 8 threads), 7.7 GB RAM, no GPU, Windows 11; Python 3.12.10, numpy 2.3.5, scipy 1.17.0.

## 0. Common settings

- Decoder = `notebook_src.py` lines 528-647 (python slice `[527:647]`), compiled from the source text, unchanged. `CFG`: `nomp_os = 4`, `nomp_rc = 3` (the values used by the original runs). Network-free decoder = `mode='greedy'` on the raw antenna-domain channel `Ht` of `outputs/cache/net__<bank>.npz` (no IABC, no network). NOMP+selfcal = the same, followed by `selfcal(lam, tau) = (1.0, 5.0)` (the hybrid's tuned setting; NOT re-tuned for the network-free decoder; nothing was tuned on any test bank in this work).
- Evaluator = `prepare_for_metric` / `get_ang_difference` compiled from `TVT_Blob_Inference.py`; `metrics()` / `by_snr()` are `exec`-ed verbatim from `notebook_src.py` lines 315-343 for the cross-check; `common.py: scene_stats/metric_by_snr` reproduces them per scene (checked: official bank, U-Net / IABR-v2 / NOMP, all four quantities identical to the printed digits, `logs`/task output; official U-Net 0.7366 / 0.6940 / 0.6035 / 0.3611).
- Metric definitions (as in `metrics()`): `pd_paper` = good components / (good + bad components) over scenes with at least L detections (scenes with fewer are dropped); `pd_strict` = dropped scenes count 2L misses; `pd_source` = fraction of paths with both components within 1 degree (dropped scenes count as misses); `rmse` = root mean square of the errors of the components within 1 degree (conditional on detection, degrees). Per SNR group, then the mean over the 8 SNR groups for `official` and `T2_*` banks; single-SNR banks have one group.
- Bootstrap: 2000 resamples, seed 0, scenes resampled with replacement within each SNR group, identical resample indices for all methods (paired); percentile 95 percent interval of the mean-over-SNR metric.
- Sanity check on the re-decode: for all banks the recomputed greedy decode equals the cached `NOMP_no_network` predictions (max abs difference 4.44e-16 rad; `logs/decode_*.log`).

## 1. E13 + E6: all banks, four methods, strict / paper-style Pd, pd_source, RMSE, paired intervals

Scripts: `e13_decode.py` (NOMP+selfcal decode, saves `preds/selfcal_pred__<bank>.npz`), `e13_stats.py` (metrics and bootstrap, writes `bank_json/<bank>.json`), `make_md.py` (tables in `e13_e6_tables.md`). Raw outputs: `logs/decode_*.log`, `logs/stats_*.log`, `bank_json/*.json`, `preds/*.npz`, `e13_e6_tables.md`. Methods: U-Net = `U-Net_paper` cache (the U-Net of [TVT], PyTorch port, cached predictions); hybrid = `IABR-v2` cache (CNN + decoder); NOMP = `NOMP_no_network` cache; NOMP+selfcal = new.

Banks processed: 51 (official 8000; T2_d1, T2_d2, T2_d5 with 8000 each; phase_d{0,1,2,5} x SNR{0,15}; gain_g{0.5,1,2,4} x SNR{0,15}; oodphase_d{10,15} x SNR{0,15}; L{1,2,4,5,6,7,8,10} x SNR{0,15}; sep_{30,20,10,5,3,2,1}deg_snr15; tail_snr{-25,-20,30,35}; 500 scenes each unless stated; L is constant within every bank, so the 8-SNR banks are the only multi-group ones).

Banks NOT processed, with the reason:
- `dldoa_test`: cached predictions exist for U-Net / IABR-v2 / NOMP, but the bank file is not on this machine (`find` finds only `outputs/dldoa_test_inference/dldoa_test_pd_rmse.png`); the NOMP+selfcal decode and the paired intervals need the bank (truth angles).
- `nuis_p{-20,-10,0}_snr{0,15}` and `v1_clean_snr{0,15,25}`: bank files exist, but there are no cached predictions for the U-Net / IABR-v2 / NOMP methods and no `net__` cache (the CNN would have to be run; not done).
- `tune_*` banks: tuning banks, excluded on purpose (never test banks).
- Banks with a variable number of paths do not exist among the above (all L constant).

### 1.1 Headline (official 8000-scene bank, mean over 8 SNR)

| quantity | U-Net | hybrid (IABR-v2) | NOMP (network-free) | NOMP+selfcal |
|---|---|---|---|---|
| paper-style Pd | 0.7366 | 0.7551 | 0.7581 | 0.7576 |
| strict Pd | 0.6940 | 0.7551 | 0.7581 | 0.7576 |
| pd_source | 0.6035 | 0.6681 | 0.6741 | 0.6720 |
| conditional RMSE (deg) | 0.3611 | 0.3337 | 0.3277 | 0.3288 |

Paired differences on the official bank (paper-style Pd; strict Pd gives the same values except for the U-Net row: NOMP - U-Net strict = +0.0641 [+0.0602, +0.0680]; RMSE differences are conditional on detection):

| difference | point [95% paired interval] |
|---|---|
| NOMP - U-Net | +0.0215 [+0.0181, +0.0248] |
| hybrid - NOMP | -0.0030 [-0.0043, -0.0016] |
| NOMP+selfcal - hybrid | +0.0025 [+0.0014, +0.0036] |
| NOMP+selfcal - NOMP | -0.0005 [-0.0014, +0.0005] |

U-Net dropped scenes (fewer than L detections), official: 538 of 8000 (matches the earlier audit), T2_d1 476, T2_d2 533, T2_d5 427 (full list at the end of `e13_e6_tables.md`).

### 1.2 Pattern over the 51 banks (counts of banks whose paired interval excludes 0; `logs/counts.txt`, script `summarize_counts.py`)

| difference | interval > 0 | interval < 0 | contains 0 | (paper-style Pd; strict Pd gives for NOMP - U-Net 48 / 0 / 3, the others identical) |
|---|---|---|---|---|
| NOMP - U-Net | 39 | 0 | 12 | |
| hybrid - NOMP | 1 (gain_g4_snr15) | 17 | 33 | |
| NOMP+selfcal - hybrid | 19 | 1 (gain_g4_snr15) | 31 | |
| NOMP+selfcal - NOMP | 8 | 1 (phase_d5_snr0) | 42 | |

The 51 intervals per difference are not corrected for multiple comparisons; the banks are separate data draws, 500 scenes each (single draw per bank). The bank where the hybrid is clearly ahead of every network-free variant is gain_g4_snr15 (hybrid 0.9123, NOMP+selfcal 0.8950, NOMP 0.8697; hybrid - NOMP +0.0427 [+0.0327, +0.0530], NOMP+selfcal - hybrid -0.0173 [-0.0257, -0.0100]), as in the earlier ablation. The banks where NOMP+selfcal exceeds NOMP are the impaired-gain/phase banks at 15 dB (gain_g2: +0.0140, gain_g4: +0.0253, oodphase_d15: +0.0097) and the close-separation banks (sep 1-5 degrees: +0.046 to +0.113); E4 below shows that the latter is not calibration.

All four tables with their intervals (every bank), the strict-Pd, pd_source and RMSE versions, the per-SNR tables for official and T2 banks and the dropped counts are in `e13_e6_tables.md` (included below verbatim).
''')
w(open(f'{OUT}/e13_e6_tables.md', encoding='utf-8').read())

# ------------------------------------------------------------------ E4
w('''
## 2. E4: control for the self-calibration gain on banks with no impairment

Script: `e4_control.py`; raw outputs: `e4/e4__<bank>.json` (all numbers), `e4/preds__<bank>.npz`, `logs/e4_sep.log`, `logs/e4_official.log`. Banks: the seven `sep_*deg_snr15` banks (2 paths, 500 scenes each, SNR 15 dB, NO phase or gain impairment) and the official bank (8000 scenes). Input `Ht`, L known, paper-style Pd (identical to strict Pd here: no dropped scenes), paired bootstrap as above.

What the code does (read from `notebook_src.py` lines 597-647): `selfcal(y, U, V, lam, tau, rc=2, iters=2)` runs `iters = 2` outer iterations; each estimates per-antenna corrections, applies them scaled by `lam` where the model/residual energy exceeds `tau`, and then calls `cyclic(yc, U, V, rc=2)`. With `lam = 0` the correction is exp(0) = 1 and the data are unchanged, but the two outer iterations still perform 2 x 2 = 4 extra cyclic re-refinement rounds (each round: 2 Newton steps for every path against the data minus the other paths). The brief said "two extra rounds"; the code gives four. `decode()` skips `selfcal` when `lam = 0` (line 645), so variant (iii) calls `selfcal(..., 0.0, 5.0)` directly after `nomp_greedy`.

Variants: (i) greedy, os = 4, rc = 3 (= cached NOMP, verified identical on every bank); (ii) greedy + selfcal (1.0, 5.0); (iii) greedy + selfcal (0.0, 5.0); (iv) greedy with rc = 5 and rc = 8 (rc acts inside every greedy stage); (v) greedy with os = 8 (rc = 3).

Paper-style Pd (mean over scenes; interval in the difference table):

''')
banks = ['sep_1deg_snr15', 'sep_2deg_snr15', 'sep_3deg_snr15', 'sep_5deg_snr15', 'sep_10deg_snr15', 'sep_20deg_snr15', 'sep_30deg_snr15', 'official']
V = ['greedy', 'sc(1,5)', 'sc(0,5)', 'rc5', 'rc8', 'os8']
w('| bank | (i) greedy rc3 os4 | (ii) +selfcal(1,5) | (iii) +selfcal(0,5) [no correction] | (iv) rc5 | (iv) rc8 | (v) os8 |\n|---|---|---|---|---|---|---|')
E4 = {b: json.load(open(f'{OUT}/e4/e4__{b}.json')) for b in banks}
for b in banks:
    w(f'| {b} | ' + ' | '.join(f'{E4[b]["variants"][v]["point"]["pd_paper"]:.4f}' for v in V) + ' |')
w('\nPaired differences in Pd, point [95% interval] (`*` = interval excludes 0):\n')
DN = [('ii-i', '(ii) - (i): selfcal(1,5) - greedy'), ('iii-i', '(iii) - (i): selfcal(0,5) - greedy'), ('ii-iii', '(ii) - (iii): calibration effect given the same refinement'), ('rc5-i', 'rc5 - (i)'), ('rc8-i', 'rc8 - (i)'), ('os8-i', 'os8 - (i)')]
w('| bank | ' + ' | '.join(n for _, n in DN) + ' |\n|---|' + '---|' * len(DN))
for b in banks:
    cells = []
    for k, _ in DN:
        v, lo, hi = E4[b]['diffs'][k]['pd_paper']; cells.append(f'{v:+.4f} [{lo:+.4f}, {hi:+.4f}]' + ('*' if lo > 0 or hi < 0 else ''))
    w(f'| {b} | ' + ' | '.join(cells) + ' |')
w('''
**Conclusion (one sentence).** On the unimpaired close-separation banks the gain of the self-calibration stage is extra cyclic refinement, not calibration: at 5 degrees separation Pd rises from 0.8970 (greedy) to 0.9430 with selfcal (1.0, 5.0) but to 0.9595 with lam = 0 (no correction applied, only the four extra cyclic rounds) and to 0.9640 with plain rc = 8, so the calibration part of the stage lowers Pd by 0.0165 [0.0090, 0.0250] relative to the refinement-only variant (the same sign at 1, 2 and 3 degrees: -0.0120, -0.0480, -0.0400, all intervals excluding 0), while a doubled oversampling (os = 8) changes nothing (+0.0005 [0.0000, +0.0015] at 5 degrees).

Further readings, each tied to the table: (a) the effect grows with decreasing separation (rc8 - greedy: +0.067 at 5, +0.149 at 3, +0.179 at 2, +0.117 at 1 degree) and vanishes at 10 degrees and above, where Pd is 0.995-0.9955 for every variant; (b) on the official bank the differences are at most 0.0019 in magnitude (greedy 0.7581, selfcal(1,5) 0.7576, selfcal(0,5) 0.7595, rc8 0.7598); (c) the calibration part does not help on unimpaired data (it is gated by the estimated SNR and shrunk by lam, but at lam = 1 it still reduces Pd at 1-5 degrees and on the official bank by 0.0019 [0.0011, 0.0027]); (d) rc = 8 is better than rc = 5 and rc = 3 at every separation up to 5 degrees, so the cyclic refinement has not converged at rc = 3 on closely spaced paths (no larger rc was run, so the ceiling is not established); (e) this control concerns the NOMP+selfcal rows only: it does not decide how much of the hybrid's advantage at gain error 4 dB comes from the network (gain_g4_snr15, where the calibration effect is real: NOMP+selfcal - NOMP = +0.0253 [+0.0177, +0.0330], section 1.2), and the "+0.084" quoted for the IABC bypass at 5 degrees was not re-examined here. Earlier project text that attributes the 5-degree gain (0.897 to 0.943) to the self-calibration should say "the self-calibration stage, which contains extra cyclic rounds" and cite this control.
''')

# ------------------------------------------------------------------ E5
raw = open(f'{OUT}/e5_output.txt', encoding='utf-8').read()
w('''
## 3. E5: diagnostics re-run for the network-free decoder

Script: `e5_diag.py`; raw outputs: `e5_output.txt` (reproduced verbatim below), `e5_results.json`, `e5_perscene.npz`, `logs/e5.log`. Scenes: the first 250 per SNR of the official bank (2000 scenes, SNR -10..25), input `Ht`, L = 3, (a) greedy only, (b) greedy + selfcal (1.0, 5.0). **The metric in this section is the component-wise score of spec 10.2 (fraction of the 6 components within 1 degree after the L1 assignment over the 3! permutations); it is NOT paper-style Pd** and must not be put in a table with it. For cross-reference, paper-style Pd of the same 2000 scenes is 0.7565 (greedy) and 0.7552 (greedy+selfcal), while the component-wise score is 0.7548 and 0.7542.

Settings and replication notes. `flags()` and the permutation rule are copied from `diag.py`; the 1-degree criterion, `ls_gains`, `residual_energy`, `newton`, `cyclic` are the notebook's. The objective for (b) is evaluated on the self-calibrated data `yc` returned by a verbatim copy of `selfcal()` (`selfcal_ext`, checked identical to `selfcal()`); the same statistic on the raw `Ht` is also printed. Iteration control: from the greedy estimate, 30 Newton iterations and 15 cyclic rounds (diag.py B) and greedy with rc = 15 in every stage; for (b) `selfcal(iters=6, rc=10)`.

**Joint Fisher information.** Model y = sum_{l=1..3} x_l a(u_l, v_l) + w, a[n,m] = exp(-j n u + j m v), n, m = 0..15, w circular complex Gaussian with E|w|^2 = sigma^2 per entry; 12 real parameters (Re x_l, Im x_l, u_l, v_l); derivative matrix D (256 x 12); FIM = (2/sigma^2) Re(D^H D) (Slepian-Bangs); CRB = FIM^-1; evaluated at the true angles and at the least-squares gains at the truth; sigma^2 = (residual energy at the truth)/(256-3) per scene, as in diag3.py. Validation (printed in the output): analytic derivative versus finite difference, relative error 3.2e-10; Monte Carlo (40 high-SNR, non-end-fire, well separated scenes, 100 noise draws each, ML started at the truth): empirical std / CRB std, median 0.994, 10-90% range 0.897-1.085 over 240 cells; FIM condition number median 579, maximum 2.5e6, no non-finite CRB entry. The probability of meeting the 1-degree criterion is computed as in diag3.py `p_within` from N(pi cos(theta), CRB sd^2) for psi (sd of u) and phi (sd of v), "wrap" = mass beyond +-pi lost (physical: u wraps to the other end-fire, a fold error), "clamp" = clamped at +-pi (hypothetical fix, E7). The single-path numbers of diag3.py are printed beside them for contrast.

**Correction to diag2.py.** The statistic `dE = (RA - RT)/sig2` with `sig2 = RT/(256-9)` was compared with the chi-square(6) 99% value 16.81, but for circular complex noise the likelihood-ratio statistic is 2*(RA - RT)/sigma^2 ~ chi-square(6); the band for `dE` should be 8.4. The printed null-scale check confirms this (truth-started ML, SNR >= 15: quantiles of T = 2(RT - RC)/sigma^2 at 5/50/95/99% are 1.45/5.45/12.80/16.55 against chi-square(6) 1.64/5.35/12.59/16.81). The taxonomy is printed with both conventions; the classification changes by at most two scenes (greedy: 12 vs 14 optimiser misses), so the conclusion does not depend on it.

Findings (numbers from the output below):
- Iteration-count control: ten times more iterations change the component-wise score of the greedy decoder from 0.7548 to 0.7568 (rc = 15 in every stage: 0.7572; 2.7% of scenes move their u estimate by more than 0.05 degree); selfcal with 3x iterations and 5x rounds: 0.7542 to 0.7551. The decoder is converged to within 0.003 on these scenes. Starting at the true angles gives 0.8186 (greedy objective) and 0.8166 (with selfcal).
- Failure taxonomy: of the 1120 failing scenes of the greedy decoder (56.0%), 1100 (98%) have an estimate that fits the data at least as well as the true angles, 8 are inside the band, 12 (1%) are optimiser misses (diag2 convention; 14 with the corrected scale); for greedy+selfcal 1129 of 1137 (99%) fit better than the truth, 1 optimiser miss. This is the same picture as for the heatmap-initialised decoder in diag2 (96% objective-limited), now slightly stronger: the residual errors are objective-limited (noise), not optimisation failures.
- Stratification: at SNR >= 10 dB, greedy scenes without an end-fire path succeed (all 6 components) in 0.880 of 648 scenes against 0.463 of 352 with an end-fire path (selfcal: 0.873 / 0.438). By power rank, the weakest of the 3 paths (median path SNR -4.8 dB) scores 0.644, the middle one 0.771, the strongest 0.849 (selfcal: 0.642 / 0.772 / 0.849).
- Observed versus CRB limit (component-wise score): all SNR, observed 0.755 (greedy) and 0.754 (selfcal) against joint-CRB wrap 0.798 / clamp 0.804 (single-path approximation: 0.802 / 0.807). The joint bound is below the single-path approximation at every SNR (e.g. -10 dB 0.415 vs 0.418; 25 dB 0.984 vs 0.987), i.e. the single-path approximation was slightly optimistic, by 0.002-0.005 per SNR (0.004 in the all-SNR mean for wrap, 0.003 for clamp). The observed score stays below the joint bound at every SNR; the gap is 0.162 at -10 dB, 0.075 at -5 dB, 0.044 at 0 dB and 0.011 at 25 dB. The joint CRB-limit prediction is a local, asymptotic quantity; the gap at low SNR is consistent with the threshold effect (the bound is not attainable there) and was not further analysed.
- End-fire loss: 819 of 12000 components have a true |cos| > 0.995; their observed success is 0.319 (greedy) and 0.309 (selfcal) against a joint-CRB prediction of 0.331 (wrap) and 0.409 (clamp); the other 11181 components score 0.787 against 0.833. 222 of the 387 fold errors (error > 90 degrees; 13.1% of the 2943 wrong components) lie on end-fire components (selfcal: 218 of 385); 19.0% of all failed components are end-fire components. If the end-fire components were as accurate as the rest, the greedy score would be 0.7867 instead of 0.7548. The wrap-versus-clamp difference of the prediction (0.331 vs 0.409 on end-fire components) is the part a fold fix could recover at best; the observed value is close to the wrap prediction.
- Contradiction check against earlier project statements: the earlier diag3 end-fire figures (observed 0.282, prediction 0.327-0.406) were for the heatmap-initialised, IABC-corrected decoder; for the network-free decoder the observed end-fire success is 0.319 and the prediction with the joint FIM is 0.331-0.409, a smaller shortfall relative to the wrap prediction. The single-path CRB used in diag3 was optimistic, as stated above.

Raw output (`e5_output.txt`, verbatim):

```
''' + raw + '''
```
''')

# ------------------------------------------------------------------ E8
e1 = json.load(open(f'{OUT}/e8_latency_threads1.json')); e6 = json.load(open(f'{OUT}/e8_latency_threads6.json'))
w('''
## 4. E8: batch-1 CPU latency of the network-free decoder (L = 3)

Script: `e8_latency.py`; raw outputs: `e8_latency_threads1.json`, `e8_latency_threads6.json`, `logs/e8_t1.log`, `logs/e8_t6.log`. Machine: ASUS VivoBook X515EP, Intel Core i5-1135G7 @ 2.40 GHz (4 cores / 8 threads), 7.7 GB RAM, Windows 11, Python 3.12.10, numpy 2.3.5 (plain NumPy, float64/complex128, no compiled or GPU code). Each timed call decodes one scene (batch = 1, 16 x 16 channel, L = 3, os = 4, rc = 3); 10 official-bank scenes at SNR 15 dB x 10 repeats = 100 runs after 3 warm-up runs; median, 5th and 95th percentile. Threads were limited through OMP/MKL/OPENBLAS_NUM_THREADS before importing numpy (only the BLAS calls are affected; the FFT and the einsum loops are single-threaded anyway). The laptop was not idle (an IDE, a browser and an antivirus process were running), so the values are indicative.

| decoder | 1 thread median (p5-p95) ms | 6 threads median (p5-p95) ms |
|---|---|---|''')
for nm in ['greedy (os=4, rc=3)', 'greedy + selfcal (1.0, 5.0)']:
    a, b = e1[nm], e6[nm]
    w(f'| {nm} | {a["median_ms"]:.2f} ({a["p5_ms"]:.2f}-{a["p95_ms"]:.2f}) | {b["median_ms"]:.2f} ({b["p5_ms"]:.2f}-{b["p95_ms"]:.2f}) |')
w('''
The thread setting has no measurable effect (batch-1 operations are too small to be parallelised). This is a measured NumPy-implementation latency on this laptop, not an edge-device or optimised-implementation figure; no U-Net or CNN latency was measured here (weights not run).

## 5. Open items and cautions

- `dldoa_test`, `nuis_*`, `v1_clean_*` not processed (section 1).
- Results are single draws of each bank and a single trained network; the intervals cover test-scene sampling only, not training-seed variance.
- (lam, tau) = (1.0, 5.0) was tuned for the hybrid; no re-tuning for the network-free decoder was done (E2 is not part of this run). E4 shows that for close separations a different setting (lam = 0 with the extra rounds) would score higher, which is a statement about this stage, not a tuned result.
- E5 uses 2000 scenes of the official bank; the full 8000-scene variant of the diagnostics and the E7 fold fix were not run.
''')
open(f'{OUT}/RESULTS_NEW.md', 'w', encoding='utf-8').write('\n'.join(L))
print('written', sum(len(x) for x in L))
