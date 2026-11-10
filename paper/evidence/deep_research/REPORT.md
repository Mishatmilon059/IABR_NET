# Joint AoA/AoD in analog mmWave MIMO: is the network needed? Final research report

Folder: `D:/ai_ml_project/paper/evidence/deep_research/` (written 2026-10-16). This report only collects results: every number is copied from a logged report or log in the sub-folders of Section 9, and each section names its sources. No new experiment and no new seed was used for it. References are used only if `lit/` marks them verified; the one exception is Weiss & Friedlander (1990), listed in `lit/when_dl_helps.md` as "partial (Crossref metadata only)", which is cited because it is the origin of gain/phase self-calibration.

Notation. Pd = paper-style detection rate (both angles within 1 deg; scenes with fewer than L estimates dropped; mean over SNR groups). Strict Pd counts dropped scenes as misses; it differs only for U-Net/ResNet. Every method gets the true L. [a, b] = paired-bootstrap 95% CI (2000 resamples within SNR groups, seed 0); `*` = CI excludes 0; only starred differences are claims. SC = gain/phase self-calibration; "SC tuned" = lam 1, tau 2, 4 iterations (tuned on seeds 9900/9915). Official bank = the authors' 8000-scene test set (L=3, SNR -10..25 dB, ideal hardware).

## 1. The question and the short answer

**Question.** Does the IABR network (CNN heatmap + IABC impairment corrector + decoder) do anything a classical receiver cannot? The classical receiver is NOMP (Newtonized OMP; Mamandipoor, Ramasamy & Madhow, IEEE TSP 2016) plus gain/phase self-calibration (Weiss & Friedlander 1990).

- **On ideal hardware NOMP alone is the best method we have.** Official bank: NOMP 0.7581; IABR-v2 dual 0.7551 (-0.0030*), all-neural 0.7508 (-0.0073*), IABR-Net heat 0.7503 (-0.0078*); published U-Net 0.7366, ResNet 0.7103. Better seeds could add at most +0.0051 [+0.0041, +0.0061]; the rest of the oracle gap is the objective under noise, not the search.
- **The original IABC corrects some gain error and almost no phase error, and costs Pd on ideal hardware.** It removes ~50% of the identifiable gain MSE at 3 dB but 10% of the phase MSE at 8 deg (1-6% beyond its range); official bank -0.0082*. Cause: its phase feature is valid only for one path at high SNR, so it learned to ignore it.
- **IABC-v2b is a strong corrector, but not a necessary one.** It removes 90-97% of the identifiable MSE at SNR >= 5 dB (69-83% RMS), near the bound, and is neutral on ideal hardware (-0.001). A classical joint-ML + shrinkage self-calibration (EB-ML-g5) matches or beats it downstream in 6 of 7 impaired cells; v2b wins only at 45 deg phase errors (+0.021*) and in speed (~6.5 vs ~37 ms).
- **The "all-neural" pipeline is CNN + a decoder unfolded from NOMP with a learned step.** It beats U-Net (+0.0142*) and ResNet (+0.0404*), is below NOMP (-0.0073*), and equals the classical decoder: putting the Newton step back in place of the MLP gives the same Pd (0.0000).
- **NOMP breaks under position errors, large phase errors, 1-2-bit phase shifters, low-resolution ADCs and phase noise.** Most of these have a classical fix: known-codebook NOMP (phase shifters, 93-100% of the gap), joint-ML self-calibration (large phase/gain), quantiser maximum likelihood (ADC), phase-noise-aware ML (phase noise).
- **A network alone never beat the best classical estimator after adversarial testing.** The ADC "network read-out wins" claim was refuted by quantiser ML (+0.04 to +0.12* better); the phase-noise wins were refuted by phase-noise-aware ML. The only surviving stand-alone win is position error sigma 0.2 at 15 dB (+0.044* vs NOMP), and a strong classical position-calibration baseline is still missing.
- **What survived is the hybrid: network output as extra starting points for a physics-model ML.** 1-bit ADC, 15 dB: +0.032* and +0.036* (replicated) over multi-start quantiser ML, +0.018* over a wider classical restart search; 2-bit +0.011*; not at 0 dB. Phase-noise hybrid +0.012* to +0.045* (one bank only).
- **Bottom line.** "The network is needed" cannot be written for ideal hardware. The evidence supports (A) a strong-baseline re-evaluation that names and credits NOMP and self-calibration, and (B) a smaller "learned seeds for physics-model ML under hardware impairments" result.

## 2. Is NOMP alone enough?

Sources: `verify/REPORT.md` (an independently written NOMP and metric reproduce the cached predictions scene by scene on 10 banks, max difference 5.6e-14 rad), `nn_decoder/REPORT.md`, `iabc_v2/s_summary.log`, `finetune/adversarial/s5_official.log`.

| method (official bank, N=8000) | Pd [95% CI] | strict Pd | RMSE deg | params |
|---|---|---|---|---|
| NOMP-only (known L, no network) | 0.7581 [0.7544, 0.7618] | 0.7581 | 0.3277 | 0 |
| NOMP os8 rc6 (more oversampling / refinement) | +0.0017* over NOMP | | | 0 |
| NOMP + SC default (1, 5) | 0.7576 [0.7540, 0.7611] | 0.7576 | 0.3288 | 0 |
| NOMP + SC tuned | 0.7565 (-0.002*) | 0.7565 | | 0 |
| IABR-v2 dual (CNN + IABC + dual-seeded classical decoder) | 0.7551 [0.7513, 0.7587] | 0.7551 | 0.3337 | 191,702 |
| all-neural (CNN + decoder unfolded from NOMP, SC on) | 0.7508 [0.7468, 0.7545] | 0.7508 | 0.3324 | 196,505 |
| IABR-Net heat (CNN + classical heat-seeded decoder) | 0.7503 [0.7463, 0.7540] | 0.7503 | 0.3329 | 191,702 |
| U-Net (Lloria et al., IEEE TVT 2026; validated port) | 0.7366 [0.7327, 0.7405] | 0.6940 | 0.3611 | 31,261,441 |
| ResNet (Lloria et al.; validated port) | 0.7103 [0.7063, 0.7146] | 0.6728 | 0.3762 | 466,321 |
| network only (heatmap read-out) | 0.4507 | 0.4507 | 0.5660 | 191,702 |

Paired differences: IABR-Net heat - NOMP -0.0078 [-0.0095, -0.0061]; IABR-v2 dual - NOMP -0.0030 [-0.0043, -0.0016]; NOMP - U-Net +0.0215 [+0.0181, +0.0248] (strict +0.0641 [+0.0602, +0.0680]). Cached ablations of IABR-v2 on the official bank (`verify/REPORT.md`): without IABC 0.7580, without self-calibration 0.7503, greedy init only 0.7548, heatmap init only 0.7503. Removing the IABC brings v2 back to NOMP's level.

Per SNR (Pd):

| SNR dB | -10 | -5 | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| NOMP | 0.2618 | 0.5063 | 0.6907 | 0.8198 | 0.8993 | 0.9448 | 0.9662 | 0.9758 |
| IABR-v2 dual | 0.2630 | 0.5067 | 0.6915 | 0.8187 | 0.8955 | 0.9390 | 0.9582 | 0.9687 |
| U-Net (strict) | 0.2225 (0.2027) | 0.4708 (0.4143) | 0.6788 (0.5967) | 0.8129 (0.7478) | 0.8885 (0.8325) | 0.9244 (0.9032) | 0.9440 (0.9280) | 0.9508 (0.9270) |
| oracle-seeded (true angles) | 0.5655 | 0.6363 | 0.7400 | 0.8452 | 0.9143 | 0.9490 | 0.9743 | 0.9843 |

**Oracle headroom.** Starting the same Newton + cyclic decoder at the true angles gives 0.8261 (+0.0680), mostly at -10 dB (+0.3037), and mostly in scenes where the truth-started solution fits the data *worse* than NOMP's. Keeping the oracle start only where the decoder's own objective prefers it gives 0.7632, i.e. +0.0051 [+0.0041, +0.0061]. Any better seed, learned or not, can recover about half a point of Pd at most on ideal hardware.

**Across the 10 verified banks** the networked decoders are clearly ahead of NOMP only on `gain_g4_snr15` (heat +0.0413, dual +0.0427). On closely spaced paths they lose heavily (sep 5 deg: heat -0.369, dual -0.038; sep 2 deg: heat -0.247, dual -0.133).

**Latency** (1 thread, batch 1; `nn_decoder/s4_latency.log`): NOMP 4.80 ms (independent version) / 6.89 ms (notebook version, the one behind the Pd numbers); CNN only 5.88 ms; CNN + classical decoder 12.27 ms; CNN + learned decoder 101.4 ms. The earlier `verify/latency.log` (NOMP 8.53 ms, CNN + decoder 24.71 ms) was measured under machine load; compare ratios, not absolute times.

## 3. How much impairment does IABC remove, and what does it buy?

### 3.1 Original IABC (458 parameters; trained on phase <= 8 deg, gain <= 3 dB)
Sources: `iabc/s5_summary.log`, `iabc/tables.md` (seeds 7700-7707, L=3, 200 scenes per SNR); adversarial re-test `iabc/adversarial/VERDICT.md` (fresh seeds 8800+k). Share of the identifiable per-antenna MSE removed, all SNR (-10..25 dB) / SNR >= 10 dB:

| cell | in range | before (RMS) | IABC | SC tuned | SC gate open, SNR >= 10 dB |
|---|---|---|---|---|---|
| phase 8 deg | yes | 4.31 deg | 10.0% / 17.5% | 28.2% / 70.5% | 70.5% |
| phase 20 deg | no | 10.76 deg | 3.3% / 5.1% | 53.0% / 94.6% | 94.9% |
| phase 45 deg | no | 24.22 deg | 1.0% / 1.4% | 26.3% / 52.0% | 97.0% |
| gain 2 dB / 3 dB | yes | - / 1.66 dB | 41.3 / 49.7% (66.6 / 76.9%) | 48.2 / 53.6% (89.1 / 94.5%) | 89.0 / 94.7% |
| gain 6 dB | no | 3.33 dB | 37.6% / 50.0% | 50.1% / 93.0% | 97.6% |
| 30 deg / 6 dB (phase / gain) | no | 16.15 deg / 3.33 dB | 5.5 / 33.5% (8.9 / 44.3%) | 17.9 / 18.0% (34.9 / 35.3%) | 92.9 / 96.2% |

- Per SNR (0 / 10 / 15 / 25 dB): 3 dB gain cell IABC 27 / 72 / 77 / 80%, SC tuned 4 / 86 / 95 / 99%, oracle ALS -32 / 87 / 96 / 100%; 8 deg phase cell IABC 3 / 12 / 17 / 22%, SC tuned -10 / 19 / 75 / 97%. With no noise at all it still only takes the phase error from 4.35 to 3.76 deg (`s1_signcheck.log`), so the weak phase correction is not caused by noise.
- Adversarial rewording: "half of the gain error" is an MSE share (RMS 1.68 -> 1.18 dB), only 9% at 1 dB, and it falls with the number of paths (58% at L=1, 45% at L=6). The "oracle ALS" is an unshrunk fit, not a bound; at 0 dB a shrunk classical SC (lam 0.5, chosen on separate seeds) removes 38.7% of the gain MSE, +11.1 points [+8.2, +13.7] more than IABC.

**What it buys downstream** (Pd, mean over SNR, 1600 scenes per cell; `s5_summary.log` C, `iabc/adversarial/VERDICT.md`):

| cell | IABC+NOMP - NOMP | IABR-Net heat - NOMP | (IABC+NOMP+SC tuned) - (NOMP+SC tuned) | same, fresh seeds, vs SNR-gated SC |
|---|---|---|---|---|
| clean | -0.009* | -0.008* | -0.002 [-0.004, +0.000] | fresh clean IABC+NOMP - NOMP -0.013* |
| 8 deg / 3 dB | +0.009* | +0.015* | +0.006 [+0.002, +0.010]* | +0.003 (n.s.) |
| 6 dB gain | +0.022* | +0.037* | +0.015 [+0.009, +0.019]* | +0.014* |
| 30 deg / 6 dB | +0.022* | +0.022* | +0.030 [+0.023, +0.036]* | +0.021 [+0.016, +0.027]* |

- Official bank: IABC+NOMP 0.7499 vs NOMP 0.7581, -0.0082 [-0.0096, -0.0069]*. Over 55 cached banks the CI is above 0 in 1 (`gain_g4_snr15`, +0.015), contains 0 in 27, below 0 in 27 (closely spaced paths -0.14 to -0.28; 30-35 dB SNR -0.025 / -0.030).
- IABR-Net heat beats NOMP at 6 dB gain (+0.034* fresh; up to +0.07-0.11 at 15-25 dB SNR) but is worse below 0 dB SNR and at 45 deg (-0.013*); its 30 deg / 6 dB edge over SC disappears against SNR-gated SC (-0.002 [-0.009, +0.007]).

### 3.2 IABC-v2
Sources: `iabc_v2/s_summary.log`, `d1_diagnose.log`, `d2_crb.log`, `criteria.json`, `t3_latency.log`; adversarial `iabc_v2/adversarial/VERDICT.md` (fresh seeds 9300-9370).

- **Diagnosis of the old IABC (confirmed on fresh seeds).** Its phase feature (row phase after de-rotating by the strongest beam) is exact only for one path at high SNR: at 25 dB it removes 83.7% of the phase error with L=1 but adds 650-2900% with L >= 3. The network learned to ignore it: the phase removed falls 43.9 -> 24.4 -> 7.7% (slope 0.25 -> 0.15 -> 0.05) for L = 1 / 3 / 9; outside the training range the slope is 0.03 (20 deg) and 0.01 (45 deg).
- **Cramer-Rao bound** (L=3, RMS over 32 antennas, unknown angles; independent re-derivation ratio 1.000000, attained by an ML estimator, MSE/CRB 0.98-1.02): phase 38.4 / 12.7 / 4.24 / 0.737 deg and gain 6.07 / 2.02 / 0.67 / 0.117 dB at SNR -10 / 0 / 10 / 25 dB. Knowing the angles barely helps (12.3 vs 12.7 deg at 0 dB). For 8 deg phase errors at SNR <= 5 dB the bound exceeds the error itself (for 3 dB gain errors only at <= 0 dB): no unbiased corrector helps there.
- **Ceiling** (Pd, 2000 scenes per cell): perfect correction of the identifiable part gives 45 deg 0.598 (NOMP 0.513, NOMP+SC tuned 0.523); 30/6 0.621 (0.574 / 0.583); 8 deg 0.743 (0.739 / 0.740). Also removing the invisible linear phase ramp would give 0.754 / 0.705 / 0.753, so the ramp causes most of the loss at large phase errors. For gain, "perfect" correction is not an upper bound (6 dB: 0.704 vs 0.721 for NOMP+SC tuned) because dividing out the gain amplifies noise on weak antennas.
- **New corrector.** v2a: 597k-parameter transformer over 32 antenna tokens (rows/columns of Ht and of the covariance matrices). v2b: v2a -> NOMP -> classical ALS -> 257k-parameter refiner (854k total). Same output as the old IABC. v2b+NOMP 5.8 ms per scene vs NOMP 1.6 ms (batched timing).

% of identifiable MSE removed at SNR >= 5 dB (v2b original / fresh replication; EB-ML = classical joint ML + James-Stein shrinkage, adversarial stage):

| cell | old IABC | v2b | SC tuned | EB-ML (classical) | approx. bound, % RMS |
|---|---|---|---|---|---|
| 8 deg | 15 | 68.4 / 67.4 (RMS 43-44) | 50 | 59.4 | 49 |
| 20 deg | 5 | 91.0 / 90.6 | 85 | 89.2 | 71 |
| 45 deg | 1 | 97.1 / 97.2 | 43 | 95.4 | 85 |
| 3 dB | 73 | 90.4 / 90.3 | 84 | 88.8 | 71 |
| 6 dB | 50 | 96.5 / 96.5 | 80 | 96.0 | 83 |
| 30 deg / 6 dB | 8 / 44 | 92.8 / 96.1 | 26 / 25 | 92.5 / 95.6 | 79 / 83 |

Downstream Pd (mean over SNR; fresh-seed values where re-run):

| cell | (v2b+NOMP+SC tuned) - (NOMP+SC tuned) | v2b+NOMP - NOMP | v2b pipeline - EB-ML-g5 pipeline |
|---|---|---|---|
| clean | -0.000 [-0.002, +0.002] | -0.002 | +0.000 |
| 8 deg / 3 dB | +0.003 [-0.002, +0.007] | +0.015* | -0.006* |
| 20 deg | +0.006* | +0.014* | +0.001 |
| 45 deg | +0.057 [+0.049, +0.064]* | +0.067* | +0.021* |
| 6 dB | +0.008* | +0.031* | -0.014* |
| 30 deg / 6 dB | +0.049 [+0.041, +0.058]* | +0.057* | -0.009* |

- Official bank: v2b+NOMP 0.7576 (-0.001 [-0.002, +0.001]); v2b+NOMP+SC tuned 0.7559 (-0.002 [-0.004, -0.001]*, mostly the SC stage itself: NOMP+SC tuned is -0.002*); EB-ML-g5+NOMP -0.000 [-0.001, +0.000]. Closely spaced paths still lose (v2b+NOMP - NOMP: 3 deg -0.105*, 5 deg -0.015*; not re-run adversarially). On clean scenes v2b adds 1.54 deg / 0.21 dB.
- Pre-registered rule (`criteria.json`): (i) error removal pass; (ii) Pd vs NOMP+SC tuned pass (7 of 10 cells; 4 of 7 tested on fresh seeds, none negative); (iii) no loss on the official bank **fail by 0.002**. Decision by the rule: drop from the claims, optional module at most.
- Adversarial: code, seeds, sign, projection and CRB confirmed; the comparator set was too weak (SC tuned shuts its own gate at large impairments: 22% removed at 30/6, 42% at 45 deg vs 88% / 96% with the gate open). EB-ML-g5 (gate on its own blind SNR >= 5 dB) beats NOMP+SC tuned in all 7 impaired cells, loses nothing on the official bank, and is better than v2b in 4 cells, equal in 2, worse only at 45 deg; it costs ~37 ms per scene vs ~6.5 ms for the v2b estimate.

### 3.3 Keep / replace / drop
- **Drop the old IABC as a contribution**: essentially no phase correction, -0.008* on the official bank, worse in 27 of 55 banks.
- **Do not claim any corrector is needed on ideal hardware**: neither v2b nor EB-ML-g5 changes official Pd.
- **Replace it by v2b only as an optional, fast learned alternative to classical ML/shrinkage self-calibration**, reported next to EB-ML-g5: v2b is better only at >= 45 deg phase errors and ~6x faster. State the limits: no correction is possible below ~5 dB SNR (bound), the phase ramp keeps Pd far below clean for >= 20 deg, and closely spaced paths still lose.

## 4. The all-neural pipeline (CNN + decoder unfolded from NOMP) vs U-Net vs NOMP
Sources: `nn_decoder/REPORT.md`, `nn_decoder/adversarial/VERDICT.md` (every number re-derived to 4 decimals with independent code; no number refuted, four descriptions reworded).

**What it is.** CNN `IABRv2(32,10,32)` (`final.pt`, 191,702 parameters incl. the 458-parameter IABC) -> heatmap + corrected channel Hc -> top-L heatmap peaks as seeds -> a decoder **unfolded from NOMP's Newton + cyclic refinement** (`lurd.pt`, 4,803 parameters).
- Learned: one shared MLP (7 -> 64 -> 64 -> 2) whose output step is (pi/32)·tanh(MLP), plus one self-calibration strength lam = 1.0232.
- MLP inputs (all analytic): gradient of the correlation score divided by the score, four probe second differences (offsets 0.02 / 0.08 rad), normalised coherence. The second differences equal the Hessian diagonal (Pearson 1.00000 at 0.02 rad); only the cross term is withheld. The learned step correlates 0.91-0.92 with the clipped Newton step, same sign 94% of the time.
- Fixed classical parts: heatmap-peak seeding, 6 pre-steps, 3 cyclic rounds with LS gains and subtraction of the other paths, 2 gated self-calibration iterations (projection onto the identifiable subspace, hard gate tau = 5). Correct description: "CNN plus a NOMP-style unrolled decoder with a learned step rule" - not "all-neural", not "no Hessian".

| official bank (N=8000) | Pd difference [95% CI] |
|---|---|
| all-neural SC on (0.7508) - U-Net (0.7366) | +0.0142 [+0.0108, +0.0176]; strict +0.0568 [+0.0530, +0.0605] |
| - ResNet (0.7103) | +0.0404 [+0.0369, +0.0439] |
| - NOMP-only (0.7581) | -0.0073 [-0.0091, -0.0056] |
| - same CNN + classical heat decoder (0.7503) | +0.0005 [-0.0002, +0.0011] (no difference) |
| SC on - SC off (0.7457) | +0.0051 [+0.0038, +0.0062] |
| clipped Newton step in place of the MLP, same unrolled structure | Pd 0.7503 = classical decoder; difference 0.0000 [0.0000, 0.0000] |

- Per SNR vs U-Net: significant at -10, -5, 15, 20, 25 dB (+0.039, +0.033, +0.012, +0.009, +0.012), not at 0, 5, 10 dB. Vs NOMP: tied below 0 dB, significantly below from 0 dB up (-0.009 to -0.013 at 10-25 dB). The classical-decoder IABR-Net has the same margin over U-Net, so the gain comes from the CNN front end feeding a NOMP-type refinement.
- Held-out check (6000 official scenes never used for design choices): vs U-Net +0.0152*, vs NOMP -0.0067*, vs classical decoder +0.0003 (n.s.). 85.5% of scenes agree with the classical decoder within 0.1 deg on every angle. No train/test leakage (nearest training scene >= 2.18 deg away, same as a fresh random draw).
- Bank counts (CI above 0 / below 0 / contains 0), L=3 banks (28): vs U-Net 14/0/14 (strict 26/0/2), vs NOMP 1/12/15, vs classical decoder 1/0/27; L != 3 banks (23): vs U-Net 7/2/14, vs NOMP 0/13/10, vs classical 1/1/21. The single win over NOMP is `gain_g4_snr15` (+0.0403 [+0.0293, +0.0513]); closely spaced pairs lose -0.25 to -0.40.
- Phase-error banks (Lloria et al. Table II setting): T2_d1 / d2 / d5 Pd 0.7500 / 0.7430 / 0.7417; vs U-Net +0.014 / +0.010 / +0.014*, vs NOMP -0.007 / -0.009 / -0.005*.
- U-Net and ResNet ports match the authors' repository results within 0.0002 / 0.0004 Pd per SNR.

**Latency** (CPU, 1 thread, ms per scene, batch 1 / batch 256): NOMP 4.80 / 0.76 (notebook version 6.89 / 1.48); CNN only 5.88 / 2.15; CNN + classical decoder 12.27 / 3.88; CNN + learned decoder SC on 101.4 / 8.14, SC off 53.0 / 5.57. Adversarial re-measurement: learned decoder alone 98.0 ms vs 6.0 ms for the numpy classical decoder; the same torch structure with a Newton step takes 38.1 ms, so about a third of the gap is the unrolled torch implementation and the rest is 48 sequential MLP calls (9 score evaluations each).

**Conclusion.** The learned decoder reproduces NOMP's refinement; it adds neither accuracy nor speed. The pipeline is a fair entry that beats the published networks on their own benchmark, but NOMP must be listed as the reference and beats it.

## 5. Where NOMP breaks

Sources: `stress/RESULTS.md` (seeds 5000 / 5015, 500 scenes per bank, L=3, all losses paired against the clean bank built from the same scenes) and `stress/adversarial/VERDICT.md` (independent forward model matching to <= 1.4e-13; fresh seeds 9000 / 9015, 1000 scenes). The classical-fix column comes from the later stages (`finetune/imp/REPORT.md`, `finetune/adversarial/VERDICT.md`, `iabc_v2/adversarial/VERDICT.md`).

| impairment (severity) | SNR dB | NOMP Pd loss vs clean (orig / fresh) | oracle-seeded gap over NOMP | original network vs working SC (fresh) | best classical fix found later |
|---|---|---|---|---|---|
| M1 coupling rho 0.3 | 15 | -0.062 / -0.057 | +0.023 | "beats NOMP" (+0.017) did not reproduce (+0.003, n.s.) | none needed beyond NOMP; fine-tuning gave no gain |
| M2 position error sigma 0.2 (half-wavelengths) | 15 / 0 | -0.302 / -0.325; -0.171 / -0.182 | +0.062 / +0.103 | no win | none: position MAP lost to NOMP (-0.012*) |
| M3 phase 20 deg | 15 | -0.101 / -0.121 | - | beats NOMP (heat +0.018*), not SC (+0.003) | SC / EB-ML |
| M3 phase 45 deg | 15 / 0 | -0.324 / -0.340; -0.207 / -0.196 | +0.056 / +0.112 | no win | EB-ML-g5 (+0.036* over SC tuned, mean over SNR); v2b better still (+0.021*) |
| M3 gain 6 dB | 15 | -0.116 / -0.115 | - | heat +0.028 [+0.020, +0.036]*, dual +0.019* vs SC tuned (claimed ~+0.07) | EB-ML-g5 (+0.024* over SC tuned, mean over SNR; beats v2b) |
| M3 phase 30 deg + gain 4 dB | 15 | -0.228 / -0.235 | - | refuted vs SC tuned (heat -0.004, dual -0.006) | SC tuned / EB-ML |
| M4 phase shifters 2 bit | 15 / 0 | -0.174 / -0.194; -0.115 / -0.115 | - | dual +0.012* / +0.006* (small); heat collapses (-0.091*) | known-codebook MD-NOMP: 0.924 vs clean 0.929 |
| M4 phase shifters 1 bit | 15 / 0 | -0.634 / -0.653; -0.499 / -0.497 | +0.263 / +0.337 | no win | MD-NOMP: 0.902 / 0.657 (NOMP 0.295 / 0.191) |
| M5 ADC 4 bit | 15 | -0.156 / -0.167 | - | - | quantisation-consistent NOMP / quantiser ML |
| M5 ADC 1 bit | 15 / 0 | -0.750 / -0.756; -0.521 / -0.537 | +0.396 / +0.343 | - | quantiser ML: 0.540 / 0.275 (NOMP 0.191 / 0.163) |
| M6 phase noise 30 deg | 15 / 0 | -0.172 / -0.173; -0.109 / -0.097 | +0.073 / +0.095 | no win (dual - NOMP -0.001); default SC hurts (-0.010) | phase-noise-aware ML: 0.835 / 0.618 |
| M8 near field 0.5 m | 15 | -0.067 / -0.058 | +0.045 | heat collapses (-0.061*) | not studied (16-element array is far-field at practical ranges) |

- **First severity with a loss above 0.05 Pd** (15 / 0 dB): coupling 0.3 / none; position 0.05 / 0.1; phase 20 / 30 deg; phase shifters 2 / 2 bit; ADC 4 / 3 bit; phase noise 20 / 30 deg; near field 0.5 m / none. Coupling, near field and position 0.05 are borderline (CIs near -0.05).
- **Oracle gaps are upper bounds, not lost search.** For 1-bit ADCs the oracle solution has a lower LS residual than NOMP's in only 0.4% (15 dB) / 1.2% (0 dB) of scenes, and LS refinement moves ideal 64-grid seeds from 0.658 to 0.430 Pd (`finetune/adc/diag.log`, `diag2.log`): the LS objective is biased by the quantiser. Under the exact quantiser likelihood, seeds do matter again (Section 6.1).
- **The default SC gate (tau 5) was unfair to NOMP**: it opens on the network-corrected channel (80% of scenes at 6 dB) but not on the raw one (2%), and never at 0 dB. Every "beats NOMP+SC" claim must use tuned, SNR-gated or EB-ML self-calibration.
- **IABR-v2 dual equals plain NOMP run on the IABC-corrected channel** within 0.001 in every fresh cell: its gains come from the correction, not the heatmap seeds.
- **Unknown L (M7)**: a blind CFAR threshold gets L right in 0.924 of scenes at 15 dB (fresh) with recall equal to known-L NOMP (0.835); counting heatmap peaks is worse at 15 dB (-0.067*) and only borderline better at 0 dB (+0.022 [+0.003, +0.042]). AIC never stops (always L = 10).

## 6. Can the network do what NOMP cannot? (fine-tuning)

### 6.1 CNN fine-tuned on low-resolution ADC data
Sources: `finetune/adc/REPORT.md` (stage), `finetune/adversarial/VERDICT.md` with `s3_adc_test.log`, `s6_latency.log`, `s7_restarts.log`, `s9_hybrid_rep.log` (adversarial).

- **Training.** From `final.pt` (untouched): each batch half the original distribution, half the same samples through a b ~ U{1..4}-bit peak-AGC ADC; 4500 steps, batch 128, 66.9 min; seeds `[31337, step]`; weights `finetune/adc/cache/adc_ft.pt`.
- **Stage result.** Read out directly ("network only", no decoder), the fine-tuned CNN beat every classical method the stage tried, incl. a new quantisation-consistent NOMP: +0.130* / +0.174* at b2 / b1, 15 dB and +0.025* at b2, 0 dB (stress seeds), replicated on seeds 8300/8315 (+0.155*, +0.178*, +0.016*). Through the decoder, fine-tuning changed nothing (official +0.0003 [-0.0001, +0.0008]).
- **Adversarial result: the stand-alone win is REFUTED.** A classical maximum-likelihood estimator under the exact quantiser likelihood (interval-censored Gaussian, Levenberg-Marquardt, 13 classical starts + EM; its one constant came out at the a-priori value on separate seeds) beats network only in all four cells (seeds 8715 / 8700, 1000 scenes):

| cell | NOMP | consistent NOMP | FT network only | quantiser ML (all starts) | ML - network only | ML from network seeds - ML | ML from truth (limit) |
|---|---|---|---|---|---|---|---|
| 1 bit, 15 dB | 0.191 | 0.237 | 0.419 | 0.540 | +0.121 [+0.106, +0.137]* | +0.032 [+0.022, +0.042]* | 0.626 |
| 2 bit, 15 dB | 0.289 | 0.451 | 0.593 | 0.682 | +0.089 [+0.076, +0.103]* | +0.005 (n.s.) | 0.719 |
| 1 bit, 0 dB | 0.163 | 0.226 | 0.239 | 0.275 | +0.036 [+0.025, +0.046]* | -0.001 (n.s.) | 0.439 |
| 2 bit, 0 dB | 0.373 | 0.422 | 0.438 | 0.476 | +0.038 [+0.026, +0.049]* | -0.004 (n.s.) | 0.540 |

- ML from the single consistent-NOMP start still wins (+0.027* to +0.109*). At 15 dB it still wins with the assumed SNR off by +-5 dB; at 0 dB a wrong noise level makes it tie or lose.
- **Hybrid (network output as an extra ML start): CONFIRMED at 15 dB.** New bank (seed 9515): ML all starts + network seeds - ML all starts = +0.036 [+0.028, +0.044]* (1 bit), +0.011 [+0.007, +0.016]* (2 bit). A wider classical restart search (35 FFT subsets + 8 jittered restarts) closes about 40% of the 1-bit gap; +0.018 [+0.010, +0.027]* remains. Not shown at 0 dB.
- **Latency** (1 thread, batch 1, ms/scene): network only 6.2, NOMP 5.8, consistent NOMP 76.9, ML one start 93.8, ML all starts 482. The hybrid (network + one ML start) is about 6.2 + 93.8 = 100 ms by addition (not timed as one pipeline), about 1/5 of multi-start ML. The network alone is 15-80x faster than ML at a cost of 0.04-0.12 Pd: a latency trade-off, not an accuracy win.
- **Caveats.** One ADC model (peak AGC, uniform quantiser); true L given; ML knows the noise variance; GAMP/Bussgang not implemented.

### 6.2 Learned decoder adapted to ADC data
Sources: `nn_decoder_adapt/REPORT.md`; adversarial in `finetune/adversarial/`. FT CNN + the decoder unfolded from NOMP, retrained 1500 steps (seeds `[61000000, step]`, 25.4 min).
- Stage result (stress / new seeds): vs NOMP +0.076* / +0.081* (1 bit, 15 dB) and +0.073* / +0.066* (2 bit, 15 dB); vs the best classical method +0.029* / +0.036* at 1 bit / 15 dB only, and significantly worse in 5 of 8 / 7 of 8 ADC cells; vs FT network only worse at every b <= 2 (-0.03 to -0.24*); official subset -0.009 [-0.013, -0.005]* vs NOMP. No regression vs the original learned decoder (+0.0001 [-0.0009, +0.0010], official).
- **Adversarial: the 1-bit / 15 dB win is REFUTED** (-0.267* vs quantiser ML; still +0.036* vs consistent NOMP).

### 6.3 Severe array imperfections
Sources: `finetune/imp/REPORT.md`; adversarial `finetune/adversarial/s4_imp.log`, `s8_physics_ml.log`, `s4b_m3pair.log`.
- **Training.** 3300 steps (61.7 min, resumed once from step 3000), 40% original data + 60% M1/M2/M3/M6 impairments, seeds `[71000000, step]`, weights `finetune/imp/cache/imp_ft.pt`. The M1 monitoring loss did not improve (0.535 -> 0.539).
- **M4 physics control (no network).** NOMP run in the measurement domain with the receiver's known quantised codebooks (MD-NOMP) closes 93-100% of the gap: 2 bit / 15 dB 0.924 (fresh 0.936) vs NOMP 0.755 and clean 0.929; 1 bit / 15 dB 0.902 (0.909) vs 0.295; 1 bit / 0 dB 0.657 (0.669) vs 0.191. At 1 bit the LS transform fails (codebook rank 12 of 16, Pd 0.02). **No network is needed for M4.**
- **Stage wins replicated on fresh seeds 8400/8415** ("learned seeds" = fine-tuned heatmap peaks as decoder starts on the raw channel, then SC), each vs the best of 9 classical methods: M6 30 deg / 15 dB +0.046*, M6 20 deg / 15 dB +0.016*, M6 30 deg / 0 dB +0.016*, M2 sigma 0.2 / 15 dB +0.035*, M8 0.5 m / 15 dB +0.008* (out of distribution). Three stress-seed wins did not replicate. Wins were found post hoc among 11 learned variants.
- **Ablation** (fresh seeds, vs NOMP+SC): the gain comes from the seeds, not the learned correction. M2: seeds only +0.037*, correction only +0.004; M6 30 deg: seeds only +0.050*, correction only +0.006 (value of the correction -0.016*); only M3 30 deg / 4 dB gets value from the correction (+0.038*).
- **Adversarial physics-model baselines** (seeds 9415 / 9425, 1000 scenes; impairment level known, also tested at x0.5):

| cell | NOMP os8 rc6 | learned seeds | physics ML (classical) | learned - physics ML | physics ML + learned seeds | hybrid - physics ML |
|---|---|---|---|---|---|---|
| M6 30 deg, 15 dB | 0.768 | 0.810 | PN-ML 0.835 | -0.026 [-0.037, -0.013]* | 0.880 | +0.045* |
| M6 20 deg, 15 dB | 0.870 | 0.887 | PN-ML 0.894 | -0.007 (n.s.) | 0.923 | +0.029* |
| M6 30 deg, 0 dB | 0.591 | 0.607 | PN-ML 0.618 | -0.011 [-0.021, -0.002]* | 0.630 | +0.012* |
| M2 sigma 0.2, 15 dB | 0.615 | 0.660 | POS-MAP 0.604 | +0.055 [+0.045, +0.066]* (vs POS-MAP x0.5) | 0.633 | +0.029* |

- The M6 learned-seed wins are **REFUTED** by phase-noise-aware ML (PN-ML; 45-61 ms/scene at batch 200, 6 threads). The hybrid is best, but on one bank only. M2 survives (+0.044* vs NOMP on new seeds), but the position-error MAP lost to NOMP (-0.012*), so a strong classical baseline is still missing there.
- **M3 (large phase/gain):** every replicated win belongs to IABC-v2b, not to fine-tuning: FT trunk + v2b - (v2b+NOMP+SC tuned) = +0.001 [-0.002, +0.004]. **M1 (coupling):** no gain; fine-tuned pipelines lose up to 0.02* at 15 dB.

### 6.4 Regressions on ideal hardware
Source: `finetune/adversarial/s5_official.log` (official 8000 / fresh official-like 2000, seeds 8780-8787). ADC-FT dual+SC - original dual+SC +0.0003 [-0.0001, +0.0008] / +0.0006; ADC-FT dual+SC - NOMP -0.0026* / -0.0034*; imp-FT dual+SC (own IABC) - NOMP -0.0080* / -0.0082*; imp-FT learned seeds - NOMP -0.0068* / -0.0040*; imp-FT with v2b - NOMP -0.0010 / -0.0001; ADC-FT + adapted learned decoder - NOMP -0.0071* / -0.0064*. Every learned decoder path sits below plain NOMP on ideal hardware, and no switch that avoids this was built for the imperfection path.

### 6.5 What survived adversarial verification

| claim | status |
|---|---|
| NOMP-only best on the official bank; U-Net/ResNet below it | CONFIRMED (independent NOMP, metric, ports; all-neural re-derivation) |
| Original network beats a working SC at 6 dB gain, 15 dB | CONFIRMED, smaller: +0.028 heat / +0.019 dual (claimed about +0.07) |
| Network beats SC at 30 deg + 4 dB; network beats NOMP under coupling 0.3 | REFUTED |
| Original IABC numbers | CONFIRMED, with reworded claims (3.1) |
| IABC-v2 error removal and Pd vs SC tuned | numbers CONFIRMED; "network needed" REFUTED; EB-ML-g5 matches or beats v2b in 6 of 7 cells |
| All-neural decoder numbers | CONFIRMED; reworded ("all-neural", "no Hessian", latency, per-SNR U-Net caveat) |
| FT network only beats every classical method at 1-2-bit ADC | REFUTED by quantiser ML (+0.036* to +0.121*) |
| Bit-aware switch as the accurate system | REFUTED; survives only as a low-latency option (6 ms vs 94-482 ms) |
| Adapted learned decoder beats best classical at 1 bit / 15 dB | REFUTED (-0.267* vs quantiser ML) |
| Fine-tuning through the decoder causes no regression (ADC, adapted decoder) | CONFIRMED |
| Learned seeds win under phase noise (M6) | REFUTED vs PN-ML (20 deg: tie) |
| Learned seeds win under position error sigma 0.2, 15 dB (M2) | CONFIRMED against the classical methods built (strong baseline missing) |
| M3 wins from fine-tuning | REFUTED (they belong to IABC-v2b) |
| M4 needs no network (MD-NOMP) | stage result on stress and fresh seeds; not re-tested adversarially (physics control, no learning) |
| Network output as a seed for physics-model ML gives the best Pd | CONFIRMED for ADC at 15 dB (replicated); M6 on one bank only; not at 0 dB |

## 7. Why recent DL AoA/AoD papers do not compare with NOMP

Source: `lit/baselines_survey.md` (21 learned-method papers checked; "V" = verified there: metadata plus the supporting text read) and `lit/when_dl_helps.md`. What the sources state is kept apart from what we infer.

**Stated in the sources.**
- **Baseline lineage.** The benchmark paper (Lloria et al., IEEE TVT 2026, R1, V) chose TSDCE and DFT-CEA because earlier work showed TSDCE beat LS and OMP, so the chain of baselines never reaches NOMP or ANM; its ResNet predecessor (PIMRC 2024, R3, V) has 0 NOMP hits. The beamspace / hybrid channel-estimation papers (R4-R6, V) compare only with on-grid OMP/AMP; R6 notes that grid dictionaries miss off-grid angles.
- **Speed.** R23 (V) calls NOMP's processing "considerably long"; R26 (V) calls NOMP/RELAX time-consuming for large model order. Counter-evidence: the NOMP paper (R27, V) reports NOMP faster than AST, Lasso and MUSIC. Our measurement: NOMP 4.8-6.9 ms per scene, about the same as one CNN forward pass (5.9 ms).
- **Framing.** Covariance / multi-snapshot DOA papers (R7-R9, V) benchmark against the MUSIC family. R14 (V) says count estimators rely on an eigendecomposition, although NOMP already had a CFAR stopping rule (R27).
- **When a strong parametric method is included, the network loses or ties at high SNR** (R13, R14, R22, R23 V; R21 abstract only). Only R23 and R24 (both V) run NOMP, and both keep Newton/NOMP refinement inside their own pipeline; R23 reports "nearly the same accuracy" as NOMP. No unfolded or learned NOMP was found (unfolded OMP exists: MOMPnet, R32, abstract only).
- **General critiques.** Bjornson & Giselsson (IEEE SPM 2020, R29, V): a trained network cannot beat a known optimal algorithm; DL helps mainly to cut complexity or handle nonlinear hardware. Lin (SIGIR Forum 2018, R30, V): gains often come from weak or untuned baselines (information retrieval, not signal processing). Null result: no published critique names the missing NOMP/ANM baselines in DL AoA/AoD work.
- `when_dl_helps.md`: DL/hybrid estimators beat model-based ones when the baseline does not model something real, with few snapshots or low SNR, with sub-sampled arrays, or on runtime against convex/Bayesian solvers; against a gridless ML-type estimator with the same model and information the gain shrinks or disappears [12][14][15]. Network-initialised likelihood refinement (Barthelme & Utschick, IEEE TSP 2021 [14]; Deep-MLE, ICASSP 2022 [31], abstract only) helps mainly by removing outliers - the same pattern as our ADC hybrid.

**Our inference (not proven).**
- NOMP comes from the frequency-estimation / radar literature, while mmWave DL papers inherit OMP/AMP baselines from compressed-sensing channel estimation.
- Side information is not the reason: L is already given to the baselines in R1, R8, R14 and R16, so NOMP with known L is a like-for-like comparison.
- A weak-baseline publication incentive (R30) fits the pattern but is not shown.
- Our own data add one practical reason: on this benchmark NOMP with known L beats the published U-Net by more (+0.0215) than our network does (+0.0142), which a paper proposing a network has no incentive to show.

## 8. Honest journal framing

Every framing must (1) name NOMP (Mamandipoor, Ramasamy & Madhow, "Newtonized Orthogonal Matching Pursuit: Frequency Estimation Over the Continuum", IEEE TSP 64(19):5066-5081, 2016) as the reference estimator and run it; (2) credit gain/phase self-calibration to Weiss & Friedlander (Circuits Syst. Signal Process. 9(3):271-300, 1990; only Crossref metadata verified so far - read the paper before submission); (3) describe our learned decoder as **unfolded from NOMP**: NOMP's Newton step replaced by a small MLP, the rest of NOMP's structure kept.

**Why hiding them would fail review.**
- Every IABR decoder (heat, dual, learned) is NOMP's Newton + cyclic refinement; signal-processing reviewers will recognise it, and putting the Newton step back gives the same Pd to 4 decimals (0.7503).
- NOMP is standard and quick to run; a reviewer who runs it finds it beats every network pipeline on the official bank (by 0.0030-0.0078) and beats U-Net by more than our network does. A table without it misstates the state of the art.
- Claiming per-antenna gain/phase correction as new is a novelty error given the 1990 lineage; uncredited self-calibration would also hide that the default gate was unfair to NOMP (Section 5).
- Weak-baseline critiques (Bjornson & Giselsson 2020; Lin 2018) are well known; an omitted NOMP baseline is a fairness rejection on its own.

### A (strongest evidence). "Newtonized OMP is a hard baseline for deep joint AoA/AoD estimation in analog mmWave MIMO"
- **Central claim.** With the same pilots and known L, NOMP with no network beats the published U-Net (+0.0215*, strict +0.0641*) and ResNet, and our own CNN hybrids including a decoder unfolded from NOMP (-0.0030* to -0.0078* for the networks). Its search leaves only +0.0051 Pd on ideal hardware. Where hardware breaks the model, classical model fixes recover most of the loss: Weiss-Friedlander-type self-calibration made robust (joint ML + shrinkage + blind-SNR gate), known-codebook NOMP for phase shifters, quantiser ML for ADCs, phase-noise-aware ML.
- **Tables/figures.** `verify/REPORT.md` (official and per-bank tables, oracle headroom); `nn_decoder/REPORT.md`, `nn_decoder/fig_pd_vs_snr.png`; `nn_decoder/adversarial/VERDICT.md` (Newton-for-MLP ablation, held-out 6000); `stress/RESULTS.md` + `stress/fig_M1.png`...`fig_M8.png`, `stress/adversarial/VERDICT.md`; `iabc_v2/s_summary.log`, `iabc_v2/fig_pd_vs_severity.png`, `iabc_v2/d2_crb.log` (CRB, ceiling, phase-ramp argument); `finetune/imp/fig_M4.png` + `m4_physics.log`; `finetune/adversarial/VERDICT.md`.
- **Attack: "a negative result with no new method."** Answer: a matched-information comparison no published analog-mmWave paper has made (`lit/baselines_survey.md` Section 3); a split of the remaining loss into search loss (+0.005) and objective loss; an identifiability + CRB analysis of per-antenna impairments incl. the unremovable phase ramp; a stress map of where NOMP breaks with a classical fix for each.
- **Attack: "your U-Net port is wrong / NOMP was tuned on the test set."** Answer: ports match the authors' repository within 0.0002 / 0.0004 Pd per SNR and Table II within sampling noise; strict Pd is reported; NOMP settings are fixed (4x oversampled FFT, 3 cyclic rounds) and an independent re-implementation reproduces it scene by scene.
- **Still needed:** Pd vs wall-clock on identical hardware for all methods; the unknown-L comparison (M7) written up with NOMP's CFAR stopping; a stronger position-error calibration baseline; ideally measured data.

### B (second). "Learned starting points for physics-model ML under hardware impairments"
- **Central claim.** Where hardware breaks NOMP's model, the best receiver is a physics-model ML estimator (quantiser likelihood, phase-noise likelihood) started from classical seeds plus the output of a CNN fine-tuned on that impairment. 1-bit ADC / 15 dB: +0.032* and +0.036* (replicated) over multi-start quantiser ML, +0.018* over a wider classical restart search; 2-bit +0.011*. Phase noise 20-30 deg: +0.012* to +0.045* over PN-ML. The network alone is never the best (quantiser ML +0.036* to +0.121* above it), but it is 15-80x faster. The fine-tuned CNN through the NOMP-unfolded decoder causes no regression on ideal hardware (+0.0003).
- **Tables/figures.** `finetune/adversarial/s3_adc_test.log`, `s7_restarts.log`, `s9_hybrid_rep.log`, `s8_physics_ml.log`, `s6_latency.log`, `VERDICT.md`; `finetune/adc/REPORT.md`, `fig_adc.png`, `diag.log`, `diag2.log` (LS objective biased by the quantiser); `nn_decoder_adapt/REPORT.md`; `finetune/imp/fig_M6.png`, `fig_M2.png`.
- **Attack: "the gains are small and partly within reach of more classical restarts."** Answer: report the wider-search result (+0.018* remains) and present latency (one ML start from network seeds, ~100 ms, vs 482 ms for multi-start ML) as the main benefit.
- **Attack: "same ADC model in training and test; true L; known noise variance; no GAMP."** Answer: state them; the ML side also gets the noise variance, and the +-5 dB mismatch test is reported.
- **Still needed:** replicate the phase-noise hybrid on a second bank; time the hybrid as one pipeline; out-of-model ADCs (clipping AGC M5c, dither); 0 dB (no gain shown); unknown L; a GAMP/Bussgang baseline if a reviewer asks; for M2, a strong position-calibration baseline before any claim.

### C (weakest). "Learned per-antenna gain/phase calibration for analog arrays: a fast alternative to ML self-calibration"
- **Central claim.** IABC-v2b removes 90-97% of the identifiable MSE at SNR >= 5 dB (69-83% RMS), near a Bayesian bound, is neutral on ideal hardware (-0.001), and is about 6x faster than a classical joint-ML + shrinkage self-calibration in the Weiss-Friedlander lineage (6.5 vs 37 ms); it is better than that classical method only at >= 45 deg phase errors (+0.021*).
- **Tables/figures.** `iabc_v2/s_summary.log`, `fig_residual_vs_snr.png`, `fig_pd_vs_severity.png`; `iabc_v2/adversarial/VERDICT.md`, `a5_summary.log` (EB-ML); `iabc_v2/d2_crb.log`, `crb.json`; `iabc_v2/d1_diagnose.log` and `iabc/adversarial/VERDICT.md` (why the old IABC failed).
- **Attack: "the classical EB-ML equals or beats it in 6 of 7 cells, and your own pre-registered criterion (iii) failed."** Answer: present it as a speed/accuracy trade-off with EB-ML in the same table, plus the CRB and identifiability analysis; do not claim necessity.
- **Still needed:** a latency-matched EB-ML (fewer LM iterations); real calibration data; adversarial re-runs of the close-spacing and IABR-swap results.

## 9. File index

All paths relative to `D:/ai_ml_project/paper/evidence/deep_research/`.

| folder | key files | content |
|---|---|---|
| `verify/` | `REPORT.md`, `tables.md`, `run_nomp.log`, `check_metric.log`, `latency.log` | independent NOMP and metric; official and 10-bank tables; oracle headroom |
| `stress/` | `RESULTS.md`, `run_stress.log`, `m7_order.log`, `fig_M1.png`...`fig_M8.png`, `results.json`, `stress_sim.py` | where NOMP breaks (M1-M8), unknown L |
| `stress/adversarial/` | `VERDICT.md`, `replicate.log`, `xcheck.log`, `m7_check.log` | fresh-seed replication, fair SC baseline |
| `iabc/` | `tables.md`, `s5_summary.log`, `s1_signcheck.log`, `s4_banks.log` / `.json`, `results.json`, `fig_iabc_reduction.png`, `fig_iabc_vs_snr.png`, `fig_pd_vs_severity.png` | original IABC error removal and Pd |
| `iabc/adversarial/` | `VERDICT.md`, `a1_replicate.log`, `a2_analyze.log`, `a3_shrink.log`, `a4_clean_added.log` | IABC replication and rewording |
| `iabc_v2/` | `s_summary.log`, `results.json`, `criteria.json`, `crb.json`, `d1_diagnose.log`, `d2_crb.log`, `t3_latency.log`, `fig_residual_vs_snr.png`, `fig_pd_vs_severity.png`, `fig_training.png`, `cache/v2a_best.pt`, `cache/v2b_best.pt` | IABC-v2 |
| `iabc_v2/adversarial/` | `VERDICT.md`, `a5_summary.log`, `a5_results.json`, `a1_crb_check.log`, `a_classical.py` (EB-ML) | v2 adversarial; classical EB-ML |
| `nn_decoder/` | `REPORT.md`, `tables.md`, `results.json`, `fig_pd_vs_snr.png`, `s4_latency.log`, `s5_resnet.log`, `s6_agreement.log` | all-neural pipeline; U-Net/ResNet comparison |
| `nn_decoder/adversarial/` | `VERDICT.md`, `a1_replicate.*`, `a2_mechanism.*`, `a3_bankcounts.*`, `a4_latency.*` | decoder mechanism; Newton-for-MLP ablation |
| `finetune/adc/` | `REPORT.md`, `results.json`, `diag.log`, `diag2.log`, `latency.log`, `fig_adc.png`, `fig_train.png`, `cache/adc_ft.pt` | ADC fine-tuning (`write_report.py` cannot be rerun: its temp input was deleted; `REPORT.md` is complete) |
| `nn_decoder_adapt/` | `REPORT.md`, `ld_adapt_adc_run.log`, `ld_compare.log`, `ld_replicate.log`, `cache/ld_adc_final_run.pt` | adapted learned decoder |
| `finetune/imp/` | `REPORT.md`, `results.json`, `eval_main.log`, `eval_fresh.log`, `final_tables.log`, `m4_physics.log`, `fig_M1.png`...`fig_M8.png`, `fig_train.png`, `cache/imp_ft.pt` | imperfection fine-tuning; M4 physics control (its M3 paragraph credits the fine-tuned net; the gain belongs to v2b, see 6.3) |
| `finetune/adversarial/` | `VERDICT.md`, `s1_audit.log`, `s3_adc_test.log`, `s4_imp.log`, `s4b_m3pair.log`, `s5_official.log`, `s6_latency.log`, `s7_restarts.log`, `s8_physics_ml.log`, `s9_hybrid_rep.log`, `mlq.py`, `advc.py` | quantiser ML, PN-ML, POS-MAP, hybrid |
| `lit/` | `baselines_survey.md`, `when_dl_helps.md`, `keyword_scan.log` | verified literature |
| (root) | `RESULTS_SUMMARY.md` | earlier consolidated summary; its ADC reading list (Section 6 there) is not from `lit/` and is not verified |
| `unfolded_v2/` | `diag/`, `lit/` (logs only) | unfinished stage, no report; nothing from it is used here |

**Large caches** (regenerable from the fixed seeds): session scratchpad `C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/` (`nn_decoder/`, `nn_decoder_adv/`, `finetune_adc/`, `finetune_imp/`, `finetune_adv/`, `nn_decoder_adapt/`, `iabc_v2_adv/`), v2b training data in `.../scratchpad/v2b_data/`; a copy of most of these is in `C:/Users/Milon/iabr_deep_research_cache/`. Checkpoints that must not be overwritten: `IABR_v2_extracted/IABR_v2/outputs/checkpoints/IABRv2/final.pt`, `paper/evidence/learned_decoder/lurd.pt`.

**Seeds used across stages** (bootstrap always seed 0):

| use | seeds |
|---|---|
| stress test / SC tuning / stress replication | 5000, 5015 / 9900, 9915 / 9000, 9015 (M7 fresh 9600, 9615) |
| IABC sweep / IABC adversarial | 7700-7707 / 8800+k, 8830-8833, 8850-8853 |
| IABC-v2 test / IABC-v2 adversarial | 8900+k / 9300-9307, 9330-9332, 9340-9347, 9350-9352, 9360-9370 |
| ADC fresh / imperfection fresh | 8300, 8315 / 8400, 8415 |
| fine-tuning adversarial | 8700, 8715 (ADC test), 8730, 8745 (kappa tuning), 8760, 8775, 8780-8787, 9415, 9425, 9435 (smoke), 9515 |
| learned-decoder leakage control | 880001 |
| training streams | [31337, s], [61000000, s], [62000000, k], [63000000, k], [64000000, s], [71000000, s]; learned decoder rng 55 |

This report itself used no new seeds and ran no experiment.
