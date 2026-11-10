# Can a CNN fine-tuned on low-resolution-ADC data give the decoder better starting points than NOMP? (finetune/adc)

Folder: `D:/ai_ml_project/paper/evidence/deep_research/finetune/adc`. Every number below is printed in a log in this folder
(`diag.log`, `diag2.log`, `train.log`, `eval_{m5,fresh,clean,official}.log`, `supp_{m5,fresh,clean,official}.log`, `latency.log`,
`report.log`); `results.json` collects all of them. Figures: `fig_adc.png` (Pd vs ADC bits), `fig_train.png` (loss curve).

## Short answer

* **No, not as "better starting points".** Feeding the fine-tuned CNN's heatmap (or its corrected channel) into the physics decoder
  (Newton + cyclic least squares, +self-calibration) gives the same Pd as NOMP within about 0.01 in 6 of 8 cells; the exceptions are
  b = 2 and 1 at 15 dB (FT heat+SC 0.281 / 0.225 vs NOMP 0.263 / 0.179), where it still loses to (b2) or only ties (b1) the best classical method. The reason is that with a coarse ADC the **LS objective itself is biased**. The
  oracle-seeded "+0.40 gap" is not search loss: the oracle solution has a *lower* LS residual than NOMP's in only 0.4% (b1, 15 dB) /
  1.2% (b1, 0 dB) of scenes, and LS refinement *moves ideal seeds away from the truth* (ideal 64-grid seeds: 0.658 Pd without
  refinement, 0.430 after Newton/cyclic, b1 15 dB). A better seed is pulled back to the same biased optimum.
* **Yes, as a direct estimator at 1-2 bit.** The fine-tuned network read out *without* the physics decoder (heatmap peaks + parabolic
  interpolation) beats every classical baseline (11 methods + 2 hybrids; best chosen per cell on the test bank itself) at
  b = 2 and b = 1 at 15 dB by +0.13 to +0.18 Pd, and at b = 2 at 0 dB by +0.016 to +0.025; all replicated on fresh seeds (1000 scenes
  per cell), all 95% CIs exclude 0. The original CNN read the same way does not (0.18-0.25), so the gain comes from fine-tuning.
  At b = 1 / 0 dB the margin is not significant on the stress bank (+0.004 [-0.010, +0.018]) and is only borderline on the fresh bank (+0.012 [+0.001, +0.022]).
* **It loses at 3-4 bit and without ADC** (-0.08 to -0.32 Pd; the network-only read-out is limited to about 0.45-0.61). A receiver
  knows its own ADC resolution, so the usable system is a **bit-aware switch**: use the network-only read-out when b <= 2, the classical
  decoder otherwise. The classical choice is the quantisation-consistent NOMP at 15 dB and plain NOMP at 0 dB.
* **Cost:** none on the official bank when the fine-tuned net is used with the decoder (FT dual+SC - orig dual+SC +0.0003
  [-0.0001, +0.0008]; vs NOMP -0.0026 [-0.0040, -0.0013], the same as the original CNN's -0.0030) and none on the clean banks. Latency:
  6.1 ms/scene for the network-only path vs 6.5 ms for NOMP and 74 ms for consistent NOMP (1 thread, batch 1).
* **New classical baseline that matters:** quantisation-consistent NOMP (de-quantisation by projection onto the known ADC cells,
  classical, no learning) is the best classical method in most ADC cells (+0.05 to +0.20 over NOMP at 15 dB). It *hurts* at 3-4 bit at 0 dB
  (-0.03 to -0.07). Any paper claim must be made against it, not against NOMP alone.

## Setup

* Banks: stress M5 (`stress_sim.base_scenes(500, L=3, snr, seed)` -> `generate(base, 'M5', b)`, peak AGC), b = 4,3,2,1, SNR 15 / 0 dB,
  seeds 5015 / 5000, the same as `stress/run_stress.py`. **Y identity confirmed** (`diag.log`): NOMP on the regenerated Y reproduces
  `stress/cache/preds__M5_<b>_snr<s>.npz` exactly (max diff 0.0 rad), `stress_sim.adc` equals the independent `adv_common.adc_peak` (max
  diff 0), and Y == adc(clean Y). Fresh replicate: the same generator with **new seeds 8315 / 8300**, 1000 scenes per cell (seeds used
  nowhere else, checked by grep). Clean banks: stress clean, seeds 5015/5000, 500 scenes. Official: frozen `eval_bank.npz`, all 8000 scenes.
* All methods are given the true L = 3. Metric: paper-style Pd (|err| <= 1 deg, mean over SNR groups). Every method returns 3 estimates
  per scene (checked, `supp_*.log` "n-estimates histogram"), so Pd is not inflated by dropped estimates (pd_strict == pd_paper).
* Paired bootstrap: 2000 resamples within SNR groups, seed 0. The "best classical" method of each cell is chosen **on the test bank
  itself** (this favours the classical side).
* Classical methods (13): NOMP (os 4, rc 3); NOMP+SC default (lam 1, tau 5); NOMP+SC tuned (lam 1, tau 2, 4 iters, frozen from
  stress/adversarial, tuned on 9900/9915); (i) FFT-peak seeds (top-3 local maxima of 4x |FFT2(Hant)| as the heatmap of decode 'heat')
  +/- SC; (ii) classical dual (greedy vs FFT-peak seeds by residual) +/- SC; (iii) NOMP os 8 rc 6; (iv) multi-start K6 (top-6 FFT
  maxima, all 20 subsets + greedy, lowest residual); **consistent NOMP** (8 iterations: fit the L-path model, clip it into the known
  quantisation cells of every Re/Im sample, re-run NOMP / cyclic on the de-quantised data, keep the lowest cell inconsistency; fixed a
  priori, ADC banks only); FFT parabolic os4 / os16 (the classical counterpart of the network-only read-out: FFT local maxima +
  parabolic interpolation, no refinement); consistent parabolic (FFT-parabolic seeds -> consistent NOMP).
* Network methods: original CNN (final.pt) heat+SC, dual+SC, network only; fine-tuned CNN (cache/adc_ft.pt) heat+SC, dual+SC, NOMP on
  its Hc (+SC), network only (`parabolic_heat_decode`, notebook lines 650-668), FT heat + consistent (FT heat+SC solution as the start
  of consistent NOMP). Reference: oracle-seeded (true (u,v) -> 3 cyclic rounds; sees the ground truth).

## Step 1: why NOMP fails, and classical fixes (diag.log, diag2.log, eval_m5.log, supp_m5.log)

| (500 scenes, stress seeds) | clean 15 | b4 15 | b2 15 | b1 15 | clean 0 | b2 0 | b1 0 |
|---|---|---|---|---|---|---|---|
| NOMP Pd | 0.929 | 0.773 | 0.263 | 0.179 | 0.690 | 0.356 | 0.169 |
| first greedy detection within 1 deg of strongest path | 0.896 | 0.702 | 0.052 | 0.020 | 0.666 | 0.226 | 0.062 |
| median cell distance of first detection | 0.009 | 0.041 | 0.209 | 0.260 | 0.038 | 0.122 | 0.232 |
| true paths with a 4x-FFT local max within 0.5 cell, in top 3 | 0.915 | 0.915 | 0.857 | 0.755 | 0.873 | 0.754 | 0.529 |
| oracle solution has LOWER LS residual than NOMP's | 0.24 | 0.154 | 0.034 | 0.004 | 0.112 | 0.026 | 0.012 |
| oracle-seeded Pd (3 cyclic rounds) | 0.938 | 0.789 | 0.381 | 0.575 | 0.744 | 0.513 | 0.512 |
| ideal 64-grid seeds, no refinement | 0.658 | 0.658 | 0.658 | 0.658 | 0.662 | 0.662 | 0.662 |
| ideal 64-grid seeds + Newton/cyclic | 0.928 | 0.783 | 0.374 | 0.430 | 0.729 | 0.484 | 0.419 |

Diagnosis: at 1-2 bit the coarse search is *not* lost, because the true paths still sit at FFT local maxima (75-86% in the top 3 at 15 dB). NOMP's estimates are
near misses (59% / 29% of the wrong estimates are within 0.25 cell at b2 / b1, 15 dB). The quantiser distorts the beamspace, so the
LS-optimal (u,v) is shifted: the oracle-started solution converges to a median of 0.13-0.20 beam cells from the truth (1 deg is about
0.14 cell at broadside), and 10 or 30 cyclic rounds give the same Pd as 3. The oracle "gap" (+0.40 / +0.34 at b1) is a *different local
optimum* near the truth, and its residual is higher than NOMP's in 99.6% / 98.8% of scenes. The objective prefers NOMP's wrong
answer, so no better search in this objective can recover the gap. Seeding (i)-(iii) and multi-start (iv) therefore change nothing
(within -0.03..+0.002 of NOMP everywhere). Only methods that change the objective help: consistent NOMP (+0.05..+0.20 at 15 dB,
+0.05 at 0 dB for b <= 2, but -0.03 / -0.07 at b4 / b3, 0 dB), or a direct estimator that bypasses LS.

## Step 2: fine-tuning (train_ft.py, train.log, train_hist.json, fig_train.png)

From `outputs/checkpoints/IABRv2/final.pt` (not modified). Every batch is half the original training distribution
(`iabr2_sim.make_batch`, notebook CFG: L 1..9, SNR -15..24 dB, 20% clean, phase <= 8 deg, gain <= 3 dB) and half the same samples passed
through `stress_sim.adc(Y, b, 'peak')`, b ~ U{1,2,3,4}, before `to_ri`. Loss: the original loss (focal heatmap + 50 x IABC MSE). Adam,
batch 128, 200-step linear warm-up, cosine 3e-4 -> 1e-5, grad-norm clip 5, 4500 steps (576k samples), 6 threads, 66.9 min, run detached.
Data seeds `default_rng([31337, step])`, monitor batch `default_rng([31338, 0])` (256 samples; loss only, not used for selection; the final
weights are used). Output `cache/adc_ft.pt` (state_dict, 830 kB); resumable checkpoint `cache/adc_ft_ckpt.pt`. Earlier attempts 1/2
(shared-CPU load, 11 s and 2 s/step) were aborted without a checkpoint (`train_attempt{1,2}_aborted.log`).
Fixed monitor batch, heat focal loss: ADC half 0.946 -> 0.790 (most of the drop happens in the first 700 steps), original half
0.470 -> 0.472 (unchanged); IABC MSE unchanged (4.15e-3 / 4.98e-3). Training loss (50-step means) 0.99 -> 0.89.

## Step 3: evaluation (eval_*.log, supp_*.log, report.log)

Headline per cell: the best classical method (13 methods, picked per cell on the test bank) vs the fine-tuned network-only read-out and
vs the best fine-tuned network+decoder variant. Paired 95% CI; * = CI excludes 0. m5 = stress seeds (500/cell), fresh = new seeds (1000/cell).

| set | cell | best classical | Pd | FT network only | diff [CI] | orig network only | best FT+decoder | diff [CI] | oracle-seeded |
|---|---|---|---|---|---|---|---|---|---|
| m5 | b4_snr15 | consistent NOMP | 0.843 | 0.546 | -0.297 [-0.318,-0.277]* | 0.562 | FT heat + consistent 0.837 | -0.006 [-0.012,+0.000] | 0.789 |
| m5 | b3_snr15 | consistent NOMP | 0.699 | 0.609 | -0.090 [-0.110,-0.070]* | 0.431 | FT heat + consistent 0.676 | -0.023 [-0.034,-0.012]* | 0.538 |
| m5 | b2_snr15 | consistent NOMP | 0.430 | 0.559 | +0.130 [+0.110,+0.148]* | 0.244 | FT heat + consistent 0.448 | +0.018 [+0.003,+0.034]* | 0.381 |
| m5 | b1_snr15 | consistent NOMP | 0.225 | 0.399 | +0.174 [+0.155,+0.192]* | 0.176 | FT heat + consistent 0.310 | +0.085 [+0.072,+0.098]* | 0.575 |
| m5 | b4_snr0 | NOMP | 0.663 | 0.532 | -0.131 [-0.149,-0.115]* | 0.531 | FT NOMP on Hc 0.661 | -0.002 [-0.005,+0.002] | 0.726 |
| m5 | b3_snr0 | classical dual+SC | 0.589 | 0.500 | -0.089 [-0.108,-0.069]* | 0.538 | FT dual+SC 0.587 | -0.001 [-0.005,+0.002] | 0.672 |
| m5 | b2_snr0 | consistent NOMP | 0.407 | 0.432 | +0.025 [+0.009,+0.041]* | 0.355 | FT heat + consistent 0.366 | -0.041 [-0.054,-0.028]* | 0.513 |
| m5 | b1_snr0 | consistent parabolic | 0.226 | 0.229 | +0.004 [-0.010,+0.018] | 0.155 | FT heat + consistent 0.215 | -0.011 [-0.022,+0.002] | 0.512 |
| fresh | b4_snr15 | consistent NOMP | 0.850 | 0.527 | -0.323 [-0.337,-0.308]* | 0.547 | FT heat + consistent 0.840 | -0.009 [-0.014,-0.005]* | 0.795 |
| fresh | b3_snr15 | consistent NOMP | 0.712 | 0.612 | -0.100 [-0.114,-0.086]* | 0.438 | FT heat + consistent 0.692 | -0.021 [-0.028,-0.013]* | 0.536 |
| fresh | b2_snr15 | consistent NOMP | 0.424 | 0.579 | +0.155 [+0.142,+0.168]* | 0.250 | FT heat + consistent 0.453 | +0.029 [+0.018,+0.039]* | 0.380 |
| fresh | b1_snr15 | consistent NOMP | 0.223 | 0.401 | +0.178 [+0.165,+0.192]* | 0.182 | FT heat + consistent 0.306 | +0.083 [+0.074,+0.094]* | 0.571 |
| fresh | b4_snr0 | NOMP os8 rc6 | 0.675 | 0.551 | -0.124 [-0.136,-0.112]* | 0.552 | FT NOMP on Hc 0.674 | -0.000 [-0.004,+0.003] | 0.737 |
| fresh | b3_snr0 | multi-start K6 | 0.603 | 0.519 | -0.083 [-0.095,-0.071]* | 0.547 | FT dual+SC 0.601 | -0.001 [-0.004,+0.002] | 0.683 |
| fresh | b2_snr0 | consistent NOMP | 0.420 | 0.436 | +0.016 [+0.005,+0.028]* | 0.372 | FT heat + consistent 0.384 | -0.036 [-0.044,-0.026]* | 0.529 |
| fresh | b1_snr0 | consistent NOMP | 0.217 | 0.229 | +0.012 [+0.001,+0.022]* | 0.153 | FT heat + consistent 0.215 | -0.002 [-0.010,+0.007] | 0.504 |

Full Pd tables for every method and cell (m5, fresh, clean, official, official by SNR) are in `report.log`; network minus best classical
with CIs for every method are in the second table of each set there.

Other comparisons (paired 95% CI, stress M5 banks):
* FT vs original CNN through the decoder: FT dual+SC - orig dual+SC is within +/-0.002 in all 8 cells (no CI excludes 0). FT heat+SC -
  orig heat+SC: +0.048 [+0.038, +0.058] (b1 15), +0.013 [+0.000, +0.026] (b2 15), but -0.016 (b3 15) and -0.034 (b2 0). The fine-tuned heatmap
  helps the heat-seeded decoder only in a few cells. Even its best cell (b1 15, 0.225) only *ties* consistent NOMP.
* FT network only - orig network only (`res2_m5.json`): +0.316 / +0.223 at b2 / b1, 15 dB; +0.178 at b3 15; +0.077 / +0.074 at b2 / b1, 0 dB;
  -0.017 (b4 15), +0.001 (b4 0), -0.038 (b3 0); +0.008 [+0.005, +0.010] on the official bank. So fine-tuning is what creates the 1-2 bit gain.
* Oracle-seeded is not an upper bound at 2-4 bit / 15 dB. It is *below* consistent NOMP (b4 -0.054, b3 -0.161, b2 -0.048), because it
  converges to the biased LS optimum.

Official bank (8000 scenes, L=3) and clean banks (regression):

| method | official Pd | clean 15 dB | clean 0 dB |
|---|---|---|---|
| NOMP | 0.7581 | 0.929 | 0.690 |
| best classical (NOMP os8 rc6) | 0.7598 | 0.931 | 0.690 |
| orig dual+SC | 0.7551 | 0.925 | 0.690 |
| FT dual+SC | 0.7555 | 0.924 | 0.689 |
| orig heat+SC | 0.7503 | 0.918 | 0.685 |
| FT heat+SC | 0.7508 | 0.918 | 0.687 |
| orig network only | 0.4506 | 0.448 | 0.535 |
| FT network only | 0.4584 | 0.464 | 0.546 |
| oracle-seeded | 0.8261 | 0.938 | 0.744 |

FT dual+SC - orig dual+SC: official +0.0003 [-0.0001, +0.0008], clean 15 -0.0007 [-0.0020, +0.0007], clean 0 -0.0003 [-0.0037, +0.0027]
-> **no regression**. FT dual+SC - NOMP on official -0.0026 [-0.0040, -0.0013] (the original CNN: -0.0030), and NOMP os8 rc6 beats NOMP by
+0.0017 [+0.0011, +0.0023]: the classical baseline still wins on ideal hardware.

Latency (`latency.log`; 1 thread, batch 1, median of 40 scenes, b1 15 dB bank): NOMP 6.5 ms, NOMP+SC tuned 12.8, classical dual 8.8,
NOMP os8 rc6 10.1, multi-start K6 50.9, consistent NOMP 74.4; CNN forward 5.8, FT network only (CNN + parabolic) 6.1, FT heat+SC 12.5,
FT dual+SC 18.3 ms.

## Step 4: verdict

* **Where the fine-tuned network beats every classical baseline** (CI excludes 0 on both the stress and the fresh seeds), always as the
  network-only read-out:
  b = 2, 15 dB: +0.130 [+0.110, +0.148] (stress), +0.155 [+0.142, +0.168] (fresh), 0.56-0.58 vs 0.42-0.43 for consistent NOMP and 0.26-0.27 for NOMP;
  b = 1, 15 dB: +0.174 [+0.155, +0.192] / +0.178 [+0.165, +0.192], 0.40 vs 0.22 (NOMP 0.18);
  b = 2, 0 dB: +0.025 [+0.009, +0.041] / +0.016 [+0.005, +0.028] (small).
  b = 1, 0 dB: tie on the stress bank (+0.004 [-0.010, +0.018]), borderline on the fresh bank (+0.012 [+0.001, +0.022]). This is **not claimed**.
  The hybrid FT heat + consistent also beats the best classical method at b2/b1 15 dB (+0.018..+0.085, replicated), but it is below network-only there.
* **Where it does not:** b = 3, 4 (both SNRs) and no ADC. There the network-only read-out loses by 0.08-0.32, and every network+decoder variant
  is at most equal to the best classical method (within 0.002 at 0 dB, -0.006..-0.023 vs consistent NOMP at 15 dB).
* **Mechanism:** this is *not* "better starting points for the decoder". With a coarse ADC, the decoder's LS objective prefers wrong
  answers. Any seed, including the truth, is refined towards them. The network wins by **not** being refined: it learns a direct
  quantisation-aware map from the 1-2 bit beamspace to angles. Its ceiling is the 32x32 heatmap read-out. Network-only Pd falls with SNR on the
  official bank (0.55 at 0-5 dB, 0.44 at 25 dB), and this read-out limit is also why it loses at 3-4 bit.
* **Cost:** no Pd cost on the official / clean banks when the fine-tuned weights are used with the decoder (FT dual+SC == orig dual+SC),
  so one set of weights can serve both regimes with a bit-aware switch (network-only if b <= 2, else decoder/classical). Latency of the
  network-only path is 6.1 ms/scene, the same as NOMP and 12x faster than consistent NOMP.
* **Caveats:** the ADC model is the stress simulator's (peak AGC, no clipping, uniform mid-rise quantiser), and the network was trained on
  it. There is no out-of-model ADC test (dither, clipping AGC, different b mix). L is given to every method. Consistent NOMP is a classical method
  designed here a priori, without tuning. A stronger quantisation-aware classical estimator (e.g. 1-bit ML / GAMP with a probit likelihood) was not
  implemented, and it is the obvious next baseline to try before any paper claim. The best classical method is picked per cell on the test bank, which
  favours the classical side.

## Files

Scripts: `acommon.py` (helpers, classical methods incl. consistent NOMP / multi-start), `diag.py`, `diag2.py` (step 1 diagnosis),
`train_ft.py` (step 2), `evaluate.py` (step 3; decoders on chunks of 400 scenes), `supp.py` (no-refinement classical counterparts,
orig network only, consistent parabolic; added after the first M5 results), `latency.py`, `run_all.py`, `run_supp.py` (detached sequential
drivers), `make_report.py` (tables, results.json, figures), `write_report.py` (this file's assembly). Results: `res_<set>.json` (evaluate),
`res2_<set>.json` (evaluate + supp, used for the tables), `diag.json`, `diag2.json`, `latency.json`, `train_hist.json`, `results.json`.
Weights: `cache/adc_ft.pt` (fine-tuned state_dict, used by the next stage), `cache/adc_ft_ckpt.pt` (resume checkpoint).
Large caches (regenerable from the fixed seeds):
`C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/finetune_adc/`
(`preds_<set>__<bank>.npz`, `stats_<set>__<bank>.npz`, `stats2_<set>__<bank>.npz`). The small `cache/*_smoke.npz` files and
`res_m5_smoke*.json` are code tests (40 scenes) and are not results.
Seeds: stress 5000/5015 (reused for the Y identity check), fresh 8300/8315 (new), training [31337, step], monitor [31338, 0], bootstrap 0.
