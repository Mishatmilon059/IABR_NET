# Adversarial verdict: all-neural IABR-Net, full-scale evaluation (`nn_decoder/`)

I tried to refute every claim with my own code. Each number below comes from a script in this folder; its stdout is in the `.log` file next to it.

| script | what it does |
|---|---|
| `a1_replicate.py` | Independent replication on four banks: official (all 8000 scenes), T2_d5, gain_g4_snr15, sep_3deg_snr15. Checks that the caches are aligned with the scenes, recomputes all-neural (SC on and off), recomputes NOMP-only and the classical heat decoder from the cached CNN outputs, scores U-Net from its cache, and computes CIs with its own metric and its own paired bootstrap. |
| `a2_mechanism.py` | Checks what the decoder actually computes: call counts, probe features compared with the analytic Hessian, the MLP step compared with the Newton step, an ablation that puts Newton in place of the MLP, and the overlap between the training scenes and the test scenes. |
| `a3_bankcounts.py` | Recounts the bank-count table independently over all 51 test banks. |
| `a4_latency.py` | Latency spot check of the decoders alone (batch 1, 1 thread). |
| `a5_resnet_check.py` | Rescores the cached ResNet-port predictions with my own metric. |

**Seeds.** I introduced one new seed: 880001, used only for the fresh-draw control in a2 step 5. It does not overlap with any earlier seed. Everything else is deterministic, and the bootstrap uses seed 0 (2000 resamples within SNR groups).

**Caches.** Large prediction caches are under `C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder_adv/`. They can be regenerated with a1, a2 and a3.

**Attempt 1 of a2.** The first run of a2 crashed (`a2_mechanism_attempt1.*`). The Newton-step ablation failed in complex64: when two Newton-refined paths merge, the 1e-6 ridge in `ls_gains_t` is below float32 resolution and the solve becomes singular. I re-ran the ablation in complex128. The learned model itself always runs in complex64, as it does in s2.

## 1. Code audit (truth, seeds, alignment, evaluator, leakage)

* **No truth at inference: CONFIRMED.**
  * `StagedSC.forward_all(H, U0, V0)` takes only Hc and the seeds.
  * Truth (`ug, vg`) appears only in `matched_loss` inside the training loops of `exp.py`, `exp3.py` and `exp5.py`.
  * The number of paths L is the true L. This is the case for every method, including U-Net and ResNet, which keep their top-L blobs.
* **Seeds only from the CNN heatmap: CONFIRMED.** The seeds come from `heat_peaks(heat, L)`, the notebook's 3x3 wrap-around maximum filter on the cached sigmoid heatmap. There is no FFT or greedy search anywhere.
* **lurd.pt and use_sc are used as stated: CONFIRMED.**
  * The model loads with 4,803 parameters, λ = 1.0232 and `training = False`, so the SC gate is the hard gate.
  * Hook counts per forward pass at L=3:
    * SC on: 48 sequential MLP calls and 2 ALS self-calibration calls.
    * SC off: 24 MLP calls and 0 self-calibration calls.
* **Same scenes and same L for all methods, and caches aligned scene by scene: CONFIRMED.**
  * I re-ran the CNN from Y16 on every 16th scene: 500 official scenes, 500 T2_d5 scenes, and 32 scenes each from the two small banks.
  * Agreement with `net__<bank>.npz`:
    * heat: max |Δ| between 3e-4 and 1.2e-3 (GPU against CPU arithmetic);
    * Hc and Ht: max relative difference ≤ 5e-7.
  * Control: shifting the scenes by one index gives a heat difference of 0.98, so a misaligned cache would show up clearly.
  * My recomputed NOMP-only predictions match `pred__NOMP_no_network` for every scene (max 2.5e-14°).
  * My recomputed all-neural predictions match the s2 cache bit for bit (max 0°).
* **Evaluator: CONFIRMED.**
  * My own metric (Hungarian matching on unwrapped Euclidean distance, as in `permute_pairs`, with |wrapped error| ≤ 1°) gives exactly the same good/bad/dropped totals as `common.scene_stats`. Examples:
    * official all-neural: 36037 / 11963 / 0;
    * official U-Net: 33313 / 11459 / 538.
  * My own bootstrap reproduces every CI to 4 decimals.
* **Could official-bank scenes be in training? NO (CONFIRMED clean).**
  * The learned decoder was trained in `exp5.py` on `gen_batch`: fresh `iabr2_sim` scenes, `default_rng(55)`, 1500 × 64 scenes. The CNN does not draw from that rng.
  * I regenerated the exact 96,000 training scenes.
  * Nearest training scene to each official scene (L∞ distance over the 6 sorted angles):
    * official scenes: minimum 2.18°, median 9.42°;
    * control of 8000 fresh draws (seed 880001): minimum 2.14°, median 9.41°.
  * No official scene is closer to a training scene than a random new scene would be. The official bank is the authors' frozen bank (stored seed 42), not produced by `gen_batch`.
  * The CNN (`final.pt`) was trained by `make_batch` on fresh simulator draws. I did not regenerate those draws (training seed not re-derived). The sampler is the same and the angles are continuous, so the same argument applies.
* **Design selection on the test set: caveat, not a refutation.**
  * `exp.py`, `exp3.py` and `exp5.py` used the official bank's first 250 scenes per SNR (2000 scenes) to compare designs: V1, V2 and V3, SC on or off, and the reported `results5.json`. No weight was fit on them.
  * On the 6000 official scenes never used for these choices, every conclusion holds (`a1`):

    | comparison | held-out 6000 scenes | design subset (2000 scenes) |
    |---|---|---|
    | all-neural Pd | 0.7519 | 0.7473 |
    | vs U-Net | +0.0152 [+0.0112, +0.0191] | +0.0112 |
    | vs NOMP-only | −0.0067 [−0.0088, −0.0048] | −0.0092 |
    | vs classical decoder | +0.0003 [−0.0005, +0.0012] | +0.0009 |

## 2. Replicated numbers (independent code)

**Official bank** (N=8000, L=3; paper-style Pd, mean over SNR groups)

| method | claimed | replicated |
|---|---|---|
| all-neural, SC on | 0.7508 | 0.7508 |
| all-neural, SC off | 0.7457 | 0.7457 |
| classical heat decoder (recomputed) | 0.7503 | 0.7503 |
| NOMP-only (recomputed) | 0.7581 | 0.7581 |
| U-Net, paper-style / strict | 0.7366 / 0.6940, 538 dropped | 0.7366 / 0.6940, 538 dropped |
| ResNet, paper-style / strict (cache rescored) | 0.7103 / 0.6728, 481 dropped | 0.7103 / 0.6728, 481 dropped |

Official bank, paired differences (all-neural SC on minus the reference):

| comparison | claimed | replicated |
|---|---|---|
| − U-Net | +0.0142 [+0.0108, +0.0176] | +0.0142 [+0.0108, +0.0176] |
| − ResNet | +0.0404 [+0.0369, +0.0439] | +0.0404 [+0.0369, +0.0439]; strict +0.0780 [+0.0737, +0.0821] |
| − NOMP-only | −0.0073 [−0.0091, −0.0056] | −0.0073 [−0.0091, −0.0056] |
| − classical decoder | +0.0005 [−0.0002, +0.0011] | +0.0005 [−0.0002, +0.0011] |
| SC on − SC off | +0.0051 [+0.0038, +0.0062] | +0.0051 [+0.0038, +0.0062] |
| SC off − classical decoder | − | −0.0046 [−0.0059, −0.0033] |

Official bank, agreement with the classical decoder:
* Share of scenes with max difference < 0.01° / < 0.1° / < 1°: 0.088 / 0.855 / 0.976. Median 0.029°.
* Scenes with every angle correct: both 3463, all-neural only 36, classical only 45. Both are as claimed.

**Official bank per SNR:** all eight rows of the claimed table are reproduced exactly (all-neural, NOMP and U-Net Pd, plus both CI columns).

**Other banks** (all-neural SC on minus reference)

| bank | AN Pd | − U-Net | − NOMP-only | − classical decoder | SC on − SC off |
|---|---|---|---|---|---|
| T2_d5 | 0.7417 | +0.0140 [+0.0107, +0.0172] | −0.0051 [−0.0068, −0.0034] | +0.0004 [−0.0003, +0.0012] | +0.0054 [+0.0041, +0.0066] |
| gain_g4_snr15 | 0.9100 | +0.0974 [+0.0817, +0.1130] | **+0.0403 [+0.0293, +0.0513]** | −0.0010 [−0.0043, +0.0023] | +0.0253 [+0.0170, +0.0330] |
| sep_3deg_snr15 (L=2) | 0.3070 | −0.2356 [−0.3066, −0.1651]; strict +0.21 (U-Net drops 412 of 500) | −0.4045 [−0.4310, −0.3775] | +0.0080 [−0.0035, +0.0195] | +0.0235 [+0.0120, +0.0360] |

Every value agrees with the claim within rounding (the claims are identical to 4 decimals).

**Bank counts (a3, all 51 test banks recomputed): identical to the claim.**

| comparison | L=3 banks (above / below / contains 0) | L≠3 banks (above / below / contains 0) |
|---|---|---|
| vs U-Net, paper-style | 14 / 0 / 14 | 7 / 2 / 14 (below: sep_3deg, sep_5deg) |
| vs U-Net, strict | 26 / 0 / 2 | 21 / 0 / 2 |
| vs NOMP-only | 1 / 12 / 15 (only gain_g4_snr15 is above) | 0 / 13 / 10 |
| vs classical decoder | 1 / 0 / 27 | 1 / 1 / 21 |

* On L=3 banks, the one bank above the classical decoder is phase_d5_snr15: +0.0033 [+0.0007, +0.0063]. With 28 banks, a single bank excluding 0 is what chance alone would be expected to produce.
* On L≠3 banks the classical-decoder exceptions are sep_5deg (above) and sep_1deg (below).
* Largest losses to NOMP-only on L≠3 banks:
  * many paths at 15 dB: L5 −0.021, L6 −0.021, L7 −0.030, L8 −0.030, L10 −0.039;
  * close pairs: sep_2deg −0.248, sep_3deg −0.405, sep_5deg −0.345.

## 3. What the decoder computes (code compared with the description)

| claim | verdict | evidence |
|---|---|---|
| only one shared MLP 7→64→64→2 plus λ (1.0232); step (π/32)·tanh(MLP) | CONFIRMED | `lurd.pt` keys: 4,802 MLP weights + λ. `max_step` = 0.0982 = π/32, which equals the notebook Newton clip 2π/64. |
| MLP inputs: gradient/score, 4 probe second differences at 0.02 and 0.08 rad, normalised coherence, "all analytic" | CONFIRMED | `RefinerV3.forward` |
| "no FFT search" | CONFIRMED | seeds come from `heat_peaks` |
| "no Newton formula" | CONFIRMED, literally | the Newton formula is not in the code |
| "no Hessian" | **OVERSTATED** | At d = 0.02 the probe feature (f(u+d)+f(u−d)−2f)/f equals d²·(∂²f/∂u²)/f: Pearson 1.00000 with the analytic Hessian diagonal, median relative error 0.33% (0.99966 / 5.1% at d = 0.08). The MLP therefore receives the gradient and the diagonal of the Hessian; only the cross term ∂²f/∂u∂v is missing. |
| the learned step rule | in effect a Newton step | At the heatmap seeds the MLP step correlates with the clipped 2-D Newton step at 0.92 (du) and 0.91 (dv), with the same sign 94% of the time. Replacing the MLP by the clipped Newton step in the same unrolled structure (no learned parameter; λ = 1) gives official Pd 0.7503. That is exactly the classical decoder: difference 0.0000 [0.0000, 0.0000]. The learned MLP adds +0.0005 [−0.0002, +0.0011] over it, which is not significant. |
| fixed structure | CONFIRMED, but under-described | See the description of the self-calibration stage below this table. |
| "all-neural" | **MISLEADING** | Only the step rule is learned. Peak picking, LS gains, residual subtraction, the self-calibration algorithm, the identifiable-subspace projection and the SNR gate are hand-designed classical components. |

What "2 gated self-calibration iterations" actually contains: each iteration is the notebook's classical `selfcal` rewritten in torch:
* an LS model of Hc from the current paths;
* an SNR estimate with a hard gate at τ = 5;
* 3 alternating rank-one LS fits of the per-row and per-column distortions;
* removal of the mean log-gain, the mean phase and the linear phase ramp;
* shrinkage by λ;
* then 2 cyclic rounds.

The only difference from the classical version is that λ was learned (1.0232 instead of 1.0).

## 4. Latency (a4, decoders alone, batch 1, 1 thread, median of 40 runs)

| decoder | median (ms) |
|---|---|
| learned decoder, SC on | 98.0 (claimed: learned decoder alone 97.5) |
| learned decoder, SC off | 48.8 |
| **the same torch structure with the Newton step instead of the MLP, SC on** | 38.1 |
| classical heat decoder (numpy, SC on) | 6.0 |
| NOMP (notebook) | 5.5 |

* The latency claims replicate. The learned decoder makes 48 MLP calls with SC on (24 with SC off), and each call evaluates the score 9 times (1 centre and 8 probes).
* About 32 ms of the gap to the classical decoder is implementation: an unrolled complex64 torch loop at batch 1 against vectorised numpy. The remaining ~60 ms is the MLP and its probes.
* The NOMP latency to quote is the notebook version (6.89 ms in s4), because that is the version that produced the Pd numbers. The `verify` version (4.80 ms) is a separate re-implementation.

## Verdict per claim

| # | claim | verdict |
|---|---|---|
| 1 | All four parts done, ResNet port done and validated, scripts finished without errors, s1 reproduced `results5.json` exactly | **CONFIRMED.** s1 diffs are +0.0000; the `.err` files are empty; ResNet cache rescored 0.7103 / 0.6728. I did not re-run the ResNet port itself, nor check it against the authors' pickle. |
| 2 | Official table (Pd, strict Pd, dropped) for all-neural on/off, classical decoder, NOMP-only, U-Net, ResNet | **CONFIRMED** with exact replication (see §2). I did not re-score the RMSE column, IABR-v2 dual (0.7551) or network-only (0.4507) here; those come from verify/. |
| 3 | "beats both published networks (U-Net and ResNet) on their own benchmark" | **NEEDS REWORDING** (true on average, not at every SNR). See the corrected sentence below. |
| 4 | "still below NOMP-only" | **CONFIRMED:** −0.0073 [−0.0091, −0.0056]; held-out 6000 scenes −0.0067 [−0.0088, −0.0048]. |
| 5 | "gives the same results as the same CNN with the classical decoder" / "reproduces the classical decoder rather than improving on it" | **CONFIRMED, and stronger than stated.** +0.0005 [−0.0002, +0.0011]; the Newton-for-MLP ablation in the identical structure gives exactly the classical Pd (Δ = 0). 85.5% agreement < 0.1°; 36 vs 45. |
| 6 | Paired differences (vs U-Net, ResNet, NOMP, classical; SC on vs off) | **CONFIRMED**, all to 4 decimals. |
| 7 | Per-SNR official table | **CONFIRMED**, all 8 rows. |
| 8 | T2_d1 / T2_d2 / T2_d5 table | **CONFIRMED.** T2_d5 replicated in full; T2_d1 and T2_d2 differences against U-Net, NOMP and the classical decoder replicated in a3. |
| 9 | Bank counts, gain_g4_snr15 +0.040, L≠3 loss ranges, tune banks reported separately | **CONFIRMED** (exact recount). Not recounted: "SC off below the classical decoder on 14/28 L=3 banks" (SC off was below on all 3 L=3 banks checked) and ResNet 4/0/0 (official bank only). |
| 10 | Latency table and "the learned decoder is the slowest part: ~48 small sequential torch steps" | **NEEDS REWORDING** (numbers replicate; the attribution is incomplete). See the corrected sentence below. |
| 11 | Model facts: parameter counts; one MLP (7→64→64→2) + λ = 1.0232; step (π/32)·tanh; inputs | **CONFIRMED.** CNN 191,702 (IABC 458), decoder 4,803, total 196,505; U-Net 31,261,441 (`run_log.txt`); ResNet 466,321 (s5 log). |
| 12 | "There is no FFT search, no Hessian and no Newton formula" | **NEEDS REWORDING.** See the corrected sentence below. |
| 13 | "All-neural" pipeline / "learned decoder" | **NEEDS REWORDING.** See the corrected sentence below. |
| 14 | No new seeds and no tuning in this stage | **CONFIRMED for this stage.** Caveat: the earlier design choices were made on 2000 of the 8000 official scenes; on the other 6000 every conclusion holds. |
| 15 | Leakage: official scenes cannot be in training | **CONFIRMED** (regenerated exp5 training set: nearest scene ≥ 2.18°, the same as the fresh-draw control). |

Corrected sentences:

* **Claim 3:** "On the authors' official test set, the pipeline beats U-Net by +0.014 [+0.011, +0.018] paper-style Pd (+0.057 strict) and ResNet by +0.040 [+0.037, +0.044], averaged over SNR. Against U-Net the per-SNR gain is significant at −10, −5, 15, 20 and 25 dB, but not at 0, 5 or 10 dB, where the CIs contain 0. The classical-decoder IABR-Net achieves the same margin, so the gain comes from the CNN front end, not from the learned decoder."
* **Claim 10:** "The learned decoder is the slowest part (98 ms at batch 1 against 6 ms for the classical numpy decoder). With SC on it makes 48 sequential MLP calls, each with 9 score evaluations. About a third of the gap is the unrolled torch implementation: the same structure with a Newton step takes 38 ms. Compare against NOMP notebook 6.9 ms, the version that produced the Pd numbers."
* **Claim 12:** "There is no FFT search and no explicit Newton formula. The MLP receives the analytic gradient and finite-difference second differences that equal the diagonal of the Hessian (Pearson 1.000 at d = 0.02 rad); only the mixed term is withheld. Its learned step correlates 0.91–0.92 with the clipped Newton step. Substituting the Newton step reproduces the classical decoder exactly (official Pd 0.7503)."
* **Claim 13:** "CNN + unrolled NOMP-style decoder with a learned step rule (4,803 parameters). Seed selection by heatmap local maxima, LS gain re-estimation, residual subtraction and the ALS self-calibration (projection, hard SNR gate τ = 5) are fixed classical components; only the step MLP and the self-calibration strength λ are learned."

**Bottom line.** No numerical claim is refuted; every number I recomputed reproduces exactly. Four descriptive claims need rewording: "all-neural", "no Hessian", the latency attribution, and "beats U-Net" without per-SNR qualification. The central finding, that the learned decoder equals the classical decoder and stays below NOMP-only, is supported more strongly than stated: replacing the MLP with the Newton step gives the classical decoder's Pd in every bootstrap resample (Δ = 0.0000 [0.0000, 0.0000]).
