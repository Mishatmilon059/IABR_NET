"""Assembles REPORT.md from REPORT_head.tmp (step 1 text) + the headline table in report.log + the text below. No computation."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
s = open(f'{HERE}/REPORT_head.tmp', encoding='utf-8').read()
old = """LS-optimal (u,v) is shifted by about 0.15-0.3 beam cells. This is more than the 1-deg tolerance, and Newton/cyclic refinement converges
to that biased optimum from any start. The oracle "gap" (+0.40 / +0.34 at b1) exists only because 3 cyclic rounds started *at the truth* do not travel all the way to the biased
optimum. It is not a better optimum of the same objective."""
new = """LS-optimal (u,v) is shifted: the oracle-started solution converges to a median of 0.13-0.20 beam cells from the truth (1 deg is about
0.14 cell at broadside), and 10 or 30 cyclic rounds give the same Pd as 3. The oracle "gap" (+0.40 / +0.34 at b1) is a *different local
optimum* near the truth, and its residual is higher than NOMP's in 99.6% / 98.8% of scenes. The objective prefers NOMP's wrong
answer, so no better search in this objective can recover the gap."""
assert old in s
s = s.replace(old, new)
s += """
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

"""
rl = open(f'{HERE}/report.log', encoding='utf-8').read().split('\n')
i = [k for k, l in enumerate(rl) if l.startswith('| set | cell |')][0]
tab = []
for l in rl[i:]:
    if not l.startswith('|'):
        break
    tab.append(l)
s += '\n'.join(tab) + '\n'
s += """
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
"""
open(f'{HERE}/REPORT.md', 'w', encoding='utf-8').write(s)
print('wrote REPORT.md', len(s))
