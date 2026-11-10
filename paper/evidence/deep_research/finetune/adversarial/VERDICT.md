# Adversarial check of the fine-tuning claims: verdict

Folder: `D:/ai_ml_project/paper/evidence/deep_research/finetune/adversarial/`. Every number below comes from a script in this folder; its stdout is in the `.log` file of the same name. Paired bootstrap: 2000 resamples within SNR groups, seed 0. `*` means the 95% CI excludes 0. Pd = `pd_paper`. Every method gets the true L = 3 and the same Y.

## Bottom line

- **The headline ADC win is REFUTED.** At 1–2 ADC bits, a classical maximum-likelihood estimator under the exact quantiser likelihood (no learning, its one constant set a priori and confirmed on separate seeds) beats the fine-tuned CNN read out directly in all four cells:
  - +0.121\* (b1, 15 dB)
  - +0.089\* (b2, 15 dB)
  - +0.036\* (b1, 0 dB)
  - +0.038\* (b2, 0 dB)
  - The network's lead over the ADC stage's own classical set (consistent NOMP) does reproduce exactly. That set was simply not the best classical method.
- **The imperfection "learned seeds" wins are REFUTED for phase noise (M6).** A phase-noise-aware Gaussian ML (PN-ML, no learning, known phase-noise level) beats the learned-seed path at M6 30° / 15 dB (−0.026\*) and M6 30° / 0 dB (−0.011\*). It ties at M6 20° / 15 dB (−0.007, n.s.).
- **The M2 position-error win SURVIVES.** It held against every classical method I built (+0.044\* vs NOMP on new seeds). My position-error MAP estimator did not help: it scored −0.012\* vs NOMP.
- **What the network is really worth here is starting points for a physics-model ML estimator, not a replacement for one.** Using the network output as an extra starting point of the classical ML gives the best Pd of all in every cell tested:
  - ADC b1 / 15 dB: +0.032\* over ML from classical starts; +0.018\* over a wider classical restart search
  - M6 30° / 15 dB: +0.045\* (hybrid with all starts); 20° / 15 dB: +0.029\*; 30° / 0 dB: +0.012\*
  - M2 0.2: POS-MAP started from the learned seeds gains +0.038\* over POS-MAP from classical starts, but stays below the plain learned-seed path
  - The ADC hybrid gain replicates on a new bank (s9, seed 9515): +0.036* at b1 and +0.011* at b2, both at 15 dB.
- **The no-regression claims are CONFIRMED in the narrow sense stated.**
  - ADC-FT dual+SC vs the original IABR-v2 dual: +0.0003 on the official bank.
  - Adapted vs original learned decoder: +0.0001.
  - Every learned decoder path still sits below plain NOMP on ideal hardware, by −0.003\* to −0.009\*.

## 1. Audit of the scripts (step 1)

`s1_audit.py` / `s1_audit.log`, `t0_unit.py` / `t0_unit.log`, `t1_unit_s8.py` / `t1_unit_s8.log`, plus reading the code.

- **Training seeds are disjoint from test and tuning seeds.**
  - ADC fine-tuning uses `default_rng([31337, step])` (`finetune/adc/train_ft.py`); the monitoring batch uses `[31338, 0]`.
  - The imperfection stage uses `[71000000(+1, +2), step]` (`finetune/imp/train_imp.py`, `icom.py`).
  - The learned-decoder adaptation uses `[61000000, step]` (`nn_decoder_adapt/ld_adapt_adc.py`).
  - These are 2-word SeedSequence entropies; all test and tuning banks use scalar seeds (5000/5015, 8300/8315, 8400/8415, 87xx, 9415/9425/9515). A grep over 1341 text files shows each training seed only in its own stage's files. First-draw check: the streams differ (`s1_audit.log`).
- **No test bank is used to pick checkpoints or thresholds.** All three training scripts save the final weights. The monitoring loss is logged only (`train_ft.py` lines 10–11, 104; `train_imp.py` lines 12, 97; `ld_adapt_adc.py` line 141).
  - The "best classical" method in the earlier stages is picked on the test bank. This favours the classical side, so it does not inflate the network wins.
- **Checkpoints are the right ones** (md5 in `s1_audit.log`).
  - `final.pt` and `lurd.pt` are unchanged.
  - `adc_ft.pt` differs from `final.pt` by 14% relative L2; `imp_ft.pt` by 7.7%.
  - `ld_adc_final_run.pt` is the adapted decoder (4,803 params).
- **The generators match `stress_sim.generate`.**
  - `train_ft.batch` applies the same `SS.adc(Y, b, 'peak')`.
  - `adc(generate('clean'))` equals `generate('M5', b)` exactly for b = 1..4 (`t0_unit.log` (2)).
  - The ML model's mean at the true parameters equals noiseless Y to 1.8e-14 (`t0_unit.log` (1)).
  - The M2 model in s8 equals noiseless `generate('M2', 0.2)` to 9e-15 (`t1_unit_s8.log` (1)).
  - The noise std `sqrt(10^(-SNR/10)/2)` per real component in `stress_sim.base_scenes` is the σ that the ML estimators assume.
- **ADC cells can be recovered from Y and b alone** (peak AGC, no clipping). 100% of unquantised samples fall inside the recovered cell for b = 1..4 (`t0_unit.log` (2)).
- **Analytic gradients and curvatures match finite differences** for the quantiser NLL, the M2 Jacobian and the PN-ML gradient: relative error ≤ 1.4e-9 (`t0_unit.log` (3), `t1_unit_s8.log`).
- **The classical baselines are fair.**
  - Tuned self-calibration = `selfcal(H, U, V, 1.0, 2.0, rc=2, iters=4)` (`acommon.SC_TUNED`, line 32), the setting from `stress/adversarial/replicate.py`.
  - The FFT-peak and dual paths match `finetune/imp/evaluate.py`: top-3 4× FFT maxima → Newton → cyclic rc3; dual = lower antenna-domain residual.
- **Same Y for all methods.** Within each script every method reads the same `Y` array. My NOMP numbers on the official bank reproduce the frozen ones exactly (0.7581, `s5_official.log`).
- **No code bug found that changes a published number.**
  - Known reporting defect: `finetune/adc/write_report.py` reads a temp file that has since been deleted, so it cannot be rerun. The ADC stage already disclosed this.
  - The M3 claim in `finetune/imp/REPORT.md` credits the FT trunk + v2b, but the gain belongs to v2b (section 4).

## 2. ADC: the quantiser-ML baseline (step 4, priority)

**Estimator** (`mlq.py`, `advc.ml_quant`): exact interval-censored Gaussian log-likelihood. For each Re/Im sample, the term is log(Φ((hi−μ)/s) − Φ((lo−μ)/s)). Cells come from the receiver's own ADC (b bits, peak AGC); the noise σ is known.
- Levenberg–Marquardt over (u, v, complex gains) with exact curvature.
- 13 classical starts: consistent NOMP, NOMP, consistent parabolic, and all 3-subsets of the top-5 FFT local maxima. A censored-data EM start is added. Per scene, the lowest NLL wins; no ground truth is used.
- The only constant, κ (noise-std inflation), was tuned on separate banks (seeds 8730/8745, 300 scenes, `s2_tune.log`). It came out at κ = 1.0, the a-priori value, so the estimator is effectively untuned.

**Test:** fresh banks, seeds 8715 (15 dB) / 8700 (0 dB), 1000 scenes per cell (`s3_adc_test.log`).

| cell | FT network only | consistent NOMP (ADC-stage best) | ML (all starts) | ML − network only | network only − consistent NOMP (replication of the claim) |
|---|---|---|---|---|---|
| b1, 15 dB | 0.419 | 0.237 | **0.540** | **+0.121 [+0.106, +0.137]\*** | +0.182\* |
| b2, 15 dB | 0.593 | 0.451 | **0.682** | **+0.089 [+0.076, +0.103]\*** | +0.141\* |
| b1, 0 dB | 0.239 | 0.226 | **0.275** | **+0.036 [+0.025, +0.046]\*** | +0.013\* |
| b2, 0 dB | 0.438 | 0.422 | **0.476** | **+0.038 [+0.026, +0.049]\*** | +0.016\* |

- **One start is enough.** ML from the consistent-NOMP start alone still beats the network: +0.109\*, +0.083\*, +0.027\*, +0.032\*.
- **Robust to a wrong noise level at 15 dB.** With the assumed SNR off by ±5 dB (single start), ML still wins:
  - b1, 15 dB: +0.070\* / +0.072\*
  - b2, 15 dB: +0.023\* / +0.013\*
  - At 0 dB a wrong noise level makes it tie or lose: b1 +0.012\* / −0.003; b2 +0.006 / −0.022\*.
- **The learned decoder path loses to ML.** FT-CNN + adapted learned decoder − ML (all starts): −0.267\*, −0.322\*, −0.067\*, −0.082\*. The nn_decoder_adapt claim (b1 / 15 dB beats the best classical method) is refuted. Against consistent NOMP it still replicates (+0.036\*).
- **Latency** (1 thread, batch 1, `s6_latency.log`):

| method | ms per scene |
|---|---|
| FT network only | 6.2 |
| NOMP | 5.8 |
| consistent NOMP | 76.9 |
| ML from consistent NOMP (1 start) | 93.8 |
| ML (all starts) | 482 |

  The network keeps a 15–80× speed advantage at a cost of 0.04–0.12 Pd. That is a latency trade-off, not an accuracy win.
- **Distance to the ML objective's own limit.** Starting ML from the true angles gives 0.626 / 0.719 / 0.439 / 0.540, so ML (all starts) is 0.04–0.16 below that limit.

**Hybrid** (network output as one more ML start; reported separately, not part of the classical set):
- ML from FT network seeds − ML (all starts): +0.032\* (b1 / 15 dB); +0.005, −0.001 and −0.004 in the other cells (all n.s.).
- **Can a wider classical search close it?** (`s7_restarts.py`, same bank, b1 / 15 dB) The wider search uses K = 7 FFT maxima (35 subsets), the other starts, EM, and 8 jittered restarts (±½ cell) around the winner.
  - Wide search − ML (all starts): +0.014 [+0.009, +0.020]\*
  - ML from FT seeds − wide search: +0.018 [+0.010, +0.027]\*
  - Wide search − network only: +0.135\*
  - So more classical restarts close about 40% of the hybrid gap but not all of it.

**Replication on a new bank** (`s9_hybrid_rep.py`, seed 9515, 15 dB, 1000 scenes, same code, nothing tuned):

| cell | FT network only | consistent NOMP | ML (all starts) | ML − network only | ML from FT seeds − ML | ML all starts + FT seeds − ML |
|---|---|---|---|---|---|---|
| b1, 15 dB | 0.412 | 0.232 | 0.525 | **+0.113 [+0.097, +0.128]\*** | +0.028 [+0.018, +0.038]\* | **+0.036 [+0.028, +0.044]\*** |
| b2, 15 dB | 0.574 | 0.439 | 0.665 | **+0.092 [+0.078, +0.106]\*** | +0.004 [−0.003, +0.010] | **+0.011 [+0.007, +0.016]\*** |

- **The refutation replicates:** quantiser ML beats network only by +0.11 / +0.09.
- **The network lead over consistent NOMP replicates too:** +0.180\* / +0.135\*.
- **The hybrid gain replicates at b1 / 15 dB.** Adding the network output as one more ML start lifts Pd by +0.028\* to +0.036\*, and by +0.011\* at b2 / 15 dB.

## 3. Phase noise (M6) and position error (M2): physics-model classical estimators (step 3)

**Simple fixes** (`s4_imp.log`, new seeds 8775 / 8760, 1000 scenes): self-cal with 8 iterations, NOMP os16 rc6 + self-cal, multi-start K6, the FFT-peak classical analogue. None of these closes any gap; all are equal to or below NOMP os8 rc6. The imp-stage wins replicate against that set:

| cell | learned seeds − best simple classical |
|---|---|
| M6 30°, 15 dB | +0.043\* |
| M6 20°, 15 dB | +0.020\* |
| M2 0.2, 15 dB | +0.034\* |
| M6 30°, 0 dB | +0.017\* |

**Physics estimators** (`s8_physics_ml.py`, new seeds 9415 (15 dB) / 9425 (0 dB), 1000 scenes; no learning; constants = known impairment level, 30 LM iterations, step clip 0.1; starts NOMP, NOMP os8 rc6, FFT-peak seeds, multi-start K6; lowest objective kept).

- **PN-ML.** A Gaussian likelihood using the exact second moments of Y = μe^{jΦ} + n. Per sample, the residual is split into the component along the model value (variance σ² + c_r|m|²) and across it (variance σ² + c_t|m|²). The weights are recomputed at every iteration, and each step is accepted only if it lowers the exact objective.
- **POS-MAP.** Joint LM over the angles, gains and per-element position errors, with a Gaussian prior at the known σ_pos.
  - Main version: the constant and ramp components of the position errors are removed, because the gains and angles absorb them.
  - The full-d version was the a-priori design. It scored below NOMP on the smoke seed 9435, which is what motivated the restriction (`s8_smoke_attempt1.log`, `s8_smoke_attempt2.log`). Both versions are reported.

| cell | NOMP os8 rc6 | learned seeds (FT seeds on Ht+SC) | physics ML (all starts) | learned − physics ML | physics ML (level ×0.5) | physics ML + learned seeds | hybrid − physics ML |
|---|---|---|---|---|---|---|---|
| M6 30°, 15 dB | 0.768 | 0.810 | **PN 0.835** | **−0.026 [−0.037, −0.013]\*** | 0.830 | **0.880** | +0.045\* |
| M6 20°, 15 dB | 0.870 | 0.887 | **PN 0.894** | −0.007 [−0.017, +0.003] | 0.891 | **0.923** | +0.029\* |
| M6 30°, 0 dB | 0.591 | 0.607 | **PN 0.618** | **−0.011 [−0.021, −0.002]\*** | 0.621 | **0.630** | +0.012\* |
| M2 0.2, 15 dB | 0.615 | **0.660** | POS 0.604 (full d 0.603) | +0.056 (point estimate; vs POS ×0.5: **+0.055 [+0.045, +0.066]\***) | 0.604 | 0.633 | +0.029\* |

- **M6:** physics ML beats the learned seeds. The learned path's advantage at 30° (+0.042\* over NOMP os8 rc6) becomes −0.026\*. The result holds with the phase-noise level mis-specified by ×0.5. PN-ML costs 45–61 ms/scene (4 starts, 6 threads, batch 200).
- **M2:** POS-MAP is below NOMP (−0.012\*), so the learned-seed win survives. The position-error MAP is the weakest point of my classical side. A better position-calibration method, for example with more snapshots or a global search, might still close the gap; this is not shown.
- **Hybrid:** starting the physics ML from the learned seeds gives the best Pd in every M6 cell.

## 4. M3 (large phase/gain)

The imp stage reported "FT trunk + v2b" at +0.066\* over EB-ML-g5 (45°, 0 dB). On new seeds (`s4_imp.log`, `s4b_m3pair.log`):
- FT v2b heat+SC: +0.062\* vs EB-ML-g5.
- FT v2b heat+SC − v2b+NOMP+SC tuned: **+0.001 [−0.002, +0.004]**.
- The fine-tuned trunk's own paths are far below v2b+NOMP+SC: FT dual+SC −0.051\*, FT heat+SC −0.046\*.
- The win belongs to the IABC-v2b corrector. Fine-tuning adds nothing (the imp stage's own text agrees).

## 5. Ideal hardware / regression (`s5_official.log`)

| comparison | official 8000 (reproduction) | fresh official-like 2000 (seeds 8780–8787) |
|---|---|---|
| ADC-FT dual+SC − orig dual+SC | +0.0003 [−0.0001, +0.0008] | +0.0006 [−0.0004, +0.0018] |
| ADC-FT dual+SC − NOMP | −0.0026\* | −0.0034\* |
| ADC-FT network only − NOMP | −0.300\* | −0.296\* |
| imp-FT dual+SC (own IABC) − NOMP | −0.0080\* | −0.0082\* |
| imp-FT learned seeds − NOMP | −0.0068\* | −0.0040\* |
| imp-FT no-correction dual+SC − NOMP | −0.0001 | +0.0013 |
| imp-FT v2b dual+SC − NOMP | −0.0010 | −0.0001 |
| ADC-FT + adapted LD − ADC-FT + original LD | +0.0001 [−0.0009, +0.0010] | +0.0004 |
| ADC-FT + adapted LD − NOMP | −0.0071\* | −0.0064\* |
| NOMP os8 rc6 − NOMP | +0.0017\* | +0.0015\* |

## 6. Verdict per claim

| # | claim (stage) | verdict | evidence (new seeds unless noted) |
|---|---|---|---|
| A1 | FT network only beats every classical method at b2 / 15 dB, b1 / 15 dB, b2 / 0 dB (ADC) | **REFUTED** | quantiser ML better by +0.089\*, +0.121\*, +0.038\*; holds with ±5 dB noise mismatch at 15 dB. The lead over the ADC-stage classical set reproduces (+0.14 to +0.18\*) |
| A2 | b1 / 0 dB not claimed (ADC) | consistent | ML +0.036\* over the network |
| A3 | Through the decoder, fine-tuning changes nothing; no regression on official / clean (ADC) | **CONFIRMED** | +0.0003 [−0.0001, +0.0008] official; +0.0006 fresh. Both are −0.003\* vs NOMP |
| A4 | Bit-aware switch (network only for b ≤ 2) is the usable system (ADC) | **REFUTED as an accuracy claim** | quantiser ML is better at b ≤ 2. The switch survives only as a low-latency option (6 ms vs 94–482 ms) |
| D1 | Adapted learned decoder beats the best classical method at b1 / 15 dB (nn_decoder_adapt) | **REFUTED** | −0.267\* vs quantiser ML (still +0.036\* vs consistent NOMP) |
| D2 | Adapted decoder: no regression vs the original learned decoder (nn_decoder_adapt) | **CONFIRMED** | +0.0001 [−0.0009, +0.0010] official; costs −0.007\* vs NOMP, as reported |
| I1 | Learned seeds win at M6 30° / 15 dB, +0.046\* (imp) | **REFUTED** | reproduces vs simple classical (+0.042\*) but PN-ML is better: −0.026\* |
| I2 | Learned seeds win at M6 20° / 15 dB, +0.016\* (imp) | **REFUTED** (no longer a win) | +0.017\* vs simple classical; −0.007 (n.s.) vs PN-ML |
| I3 | Learned seeds win at M6 30° / 0 dB, +0.016\* (imp) | **REFUTED** | −0.011\* vs PN-ML |
| I4 | Learned seeds / FT heat win at M2 σ 0.2 / 15 dB, +0.035\* (imp) | **CONFIRMED (against the classical methods built here)** | +0.044\* vs NOMP, +0.055\* vs POS-MAP. A stronger position-calibration baseline is still missing |
| I5 | M3 wins (FT trunk + v2b) (imp) | **REFUTED as a fine-tuning win** | FT v2b − v2b+NOMP+SC = +0.001 (n.s.); the gain is IABC-v2b's |
| I6 | Fine-tuned pipelines cost 0.006–0.012 Pd on ideal hardware (imp) | **CONFIRMED** | own-IABC dual −0.008\*; learned seeds −0.0068\* official / −0.0040\* fresh |
| H1 | (new) Network output as a seed for a physics-model ML gives the best Pd | **CONFIRMED for ADC at 15 dB (b1 strongly, b2 small); M6 single bank only** | ADC b1 / 15 dB +0.032\* (8715), +0.036\* (9515); b2 +0.011\* (9515); +0.018\* vs wide search; M6 +0.012\* to +0.045\* |

**H1 is CONFIRMED** (replicated on seed 9515). The network output is a useful extra starting point for the quantiser ML: +0.036\* at b1 / 15 dB and +0.011\* at b2 / 15 dB, over ML from classical starts. A wider classical restart search closes only part of the gap at b1 / 15 dB (+0.018\* remains, seed 8715). Not shown at 0 dB, where ML from FT seeds − ML is −0.001 and −0.004 (n.s.). For M6 it holds on one new bank each (+0.012\* to +0.045\*) and has not been replicated on a second bank.

## Caveats

- Every method gets the true L = 3.
- The ML estimators get the noise variance; PN-ML and POS-MAP also get the impairment level. A mis-specified level (×0.5) is tested for M6 / M2, and the noise level ±5 dB for ADC. The network was trained over a range of levels.
- Only the training ADC model was tested (peak AGC, uniform quantiser).
- **GAMP / Bussgang:** neither was implemented. The exact-likelihood LM with EM and many starts is a stronger estimator for this problem size (it is the target that GAMP approximates).
- ML (all starts) is 15–80× slower than the network.
- PN-ML latency was measured at batch 200 with 6 threads, not 1 thread at batch 1.
- POS-MAP's design was changed once after the smoke seed (no test seed was involved). Both versions are reported, and both lose to NOMP.

## Seeds used in this stage

All were checked with grep as unused in any `.py` / `.ipynb` under `D:/ai_ml_project` before use; some appear as numbers inside `.md` / `.json` tables.

| purpose | seeds |
|---|---|
| ADC test | 8700 (0 dB), 8715 (15 dB) |
| ADC κ tuning | 8730, 8745 |
| imperfection replicate | 8760, 8775 |
| official-like | 8780–8787 |
| physics-ML test | 9415 (15 dB), 9425 (0 dB) |
| physics-ML smoke | 9435 |
| hybrid replicate | 9515 |
| s7 jitter | `default_rng([8715, 7])` |
| bootstrap | 0 |

## Files

- **Scripts and logs:** `advc.py`, `mlq.py`, `t0_unit.*`, `t1_unit_s8.*`, `s1_audit.*`, `s2_tune.*` (+ `tune.json`, `tune_parts/`), `s3_adc_test.*`, `s4_imp.*`, `s4b_m3pair.*`, `s5_official.*`, `s6_latency.*`, `s7_restarts.*`, `s8_physics_ml.*` (+ `s8_smoke*.log`), `s9_hybrid_rep.*`, `run_all.*`.
- **Per-scene caches** (regenerable from the seeds): `C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/finetune_adv/`.
