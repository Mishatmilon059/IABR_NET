# Adversarial check of the IABC claims

Folder: `D:/ai_ml_project/paper/evidence/deep_research/iabc/adversarial/`. No file outside this folder was changed. `final.pt` was only read.

| script | log | what it does |
|---|---|---|
| `a1_replicate.py` | `a1_replicate.log`, `a1_replicate_part2.log` (adds cells g1, g2) | Fresh-seed sweep and recompute of the official bank. Per-scene results are in `cache/a1__*.npz`. |
| `a2_analyze.py` | `a2_analyze.log` (`a2_analyze_attempt1.log` is the same run without g1/g2) and `a2_results.json` | Tables and paired bootstraps. |
| `a3_shrink.py` | `a3_shrink.log` (`a3_shrink_attempt1.log` is the same run without the CI line) | Shrunk classical ALS and self-cal, with lam picked on separate tuning seeds. |
| `a4_clean_added.py` | `a4_clean_added.log` | Error that IABC adds on clean scenes. |

**Seeds.** All are new and none overlaps 5000/5015, 7700–7707 or 9900/9915:
- Sweep scenes: 8800+k, where k is the SNR index (−10..25 dB in 5 dB steps), 200 scenes per SNR, L=3.
- Noise: [seed,1]. Impairments: [seed,3].
- L-robustness runs: 8830–8833, 300 scenes per SNR.
- Shrinkage tuning: 8850/8851. Shrinkage test: 8852/8853, 400 scenes each.
- Bootstrap: 2000 resamples within SNR groups, seed 0.

**Independence.**
- The identifiable target and residual are recomputed with my own least-squares projection. I did not use `impairment_targets` or `ident_err`. The two agree with the simulator target to within 1.5e-8.
- Heat and dual are decoded with the notebook `decode()` directly, not reassembled.
- `selfcal_ret` returns the same angles as the notebook `selfcal` (max |dU| = 0).
- Fresh `Hc` equals `Ht*exp(-imp_lg - j imp_ph)` to within 3e-7.
- On the official bank, the CNN was re-run on Y rebuilt from cached `Ht`. Fresh `Hc` matches the cached `Hc` to within 4.8e-7. NOMP was re-run on both inputs.

**Extra classical baseline.** "SC snrgate" is self-cal with lam 1, tau 0 and 4 iterations, applied only when the nominal SNR is ≥ 10 dB. The 10 dB threshold was fixed in advance from the 7700-seed sweep. It assumes the receiver knows its noise power.

## 1. Script audit (iabc/)

| item | verdict |
|---|---|
| Sign and conjugation of the correction | **OK.** In s1, the front() sign leaves a residual of 1e-7°, the opposite sign gives 2× the error, and the IABC output equals the correction front() applies. |
| Identifiable projection applied the same way to target, IABC, SC and oracle | **OK.** `ident_err` re-projects every residual, and my independent projection agrees. |
| dB conversion (20/ln10 × natural-log gain) | **OK.** |
| "Before" uses only the identifiable part | **OK.** |
| Same scenes for every method | **OK.** All methods are paired on the same angles, noise and uniform draws. |
| Oracle ALS | **Correct as implemented, but mislabelled.** It is an unshrunk least-squares fit at the true angles, not an upper bound. At 0 dB, a shrunk version (lam 0.5) removes +38% of the gain MSE where lam 1 gives −40% (a3). "Gated oracle" is therefore not a bound either. |
| Fair self-cal | **Partly.** Tuned SC (tau 2) gates on the model-fit SNR. That gate stays shut under large impairments: at SNR ≥ 10 dB it removes only 30% at 30°/6 dB and 52% at 45°, while SC snrgate removes 92–97%. Every "beats tuned SC" claim at large severities must also be checked against SC snrgate (done below). |
| Mixed metrics in claim i.1 | **Inconsistent.** "49.7% of MSE" is paired with "1.66 → 1.10 dB RMS". 1.10 is the mean of per-scene RMS. The RMS that matches the MSE figure is 1.68 → 1.18 dB. |

## 2. Replicated headline numbers (fresh seeds)

| quantity | original | replicated [95% CI] |
|---|---|---|
| Gain MSE removed by IABC, g3, all SNR | 49.7% | 50.5% [49.7, 51.3] |
| Same, SNR ≥ 10 / 15 dB / 0 dB | 76.9 / 77 / 27% | 77.5 / 78.8 [76.7, 80.9] / 29.5 [26.9, 32.0]% |
| Same, g6, at 15 dB / 0 dB | 50 / 35% | 51.2 [50.2, 52.1] / 36.1 [34.4, 37.6]% |
| Phase MSE removed by IABC, ph8, all SNR / 15 dB / 0 dB | 10.0 / 17 / 3% | 10.4 [9.9, 10.8] / 18.5 / 2.6% |
| Phase MSE removed by IABC, ph45, all SNR | 1.0% | 1.0% [1.0, 1.1] |
| Residual MSE, IABC − SC tuned, g3 15 dB / 0 dB (dB²) | — | +0.469 [+0.415, +0.527] / −0.686 [−0.773, −0.595] |
| Pd, IABC+NOMP − NOMP, g6 / c30_6 | +0.022 / +0.022 | +0.024 [+0.018, +0.030] / +0.028 [+0.022, +0.033] |
| Pd, IABC+NOMP − NOMP, clean (fresh) | −0.009 | −0.013 [−0.016, −0.009] |
| Pd, IABC+NOMP − NOMP, official bank (8000) | −0.0082 [−0.0096, −0.0069] | −0.0082 [−0.010, −0.007]; NOMP 0.7581, IABC+NOMP 0.7499 (recomputed) |

## 3. Verdict per claim

**(i.1) "Inside its training range, IABC removes about half of the identifiable per-antenna gain error." — NEEDS REWORDING.**
- The ~50% figure holds only at the edge of the training range (3 dB).
- At 1 dB IABC removes 8.9% [6.5, 11.4]; at 2 dB, 41.8% [40.7, 42.8].
- The figure depends on L: 58% with L=1 and 45% with L=6, at 3 dB.
- "Half" refers to MSE. In RMS terms the drop is 1.68 → 1.18 dB, which is 30%.
- Tuned SC removes more over all SNRs (53.9% at g3) and much more at SNR ≥ 10 dB (94.4% against 77.5%).

Corrected sentence: *"Inside its training range IABC removes 9–50% of the mean-squared identifiable gain error, increasing with the error size: 50% at 3 dB, i.e. RMS 1.68 → 1.18 dB, and 77% at SNR ≥ 10 dB. Tuned classical self-calibration removes 94% at SNR ≥ 10 dB."*

**(i.2) "At 0 dB … where oracle ALS makes the gain error worse (−32%), IABC still removes 27%." — NEEDS REWORDING (true but misleading).**
- Replicated: IABC 29.5%, unshrunk oracle −35.9%, tuned SC 5.3%.
- The "oracle" is an unshrunk estimator, so it is not a ceiling.
- A deployable self-cal from NOMP angles with lam 0.5 (picked on separate seeds) removes 38.7%. That is +11.1 points [+8.2, +13.7] more than IABC (a3).
- That lam-0.5 self-cal also adds 6.7° of phase error in this gain-only cell. Applying only its gain part would avoid that. This last point follows from the logged numbers; it is not a separate run.

Corrected sentence: *"At 0 dB SNR IABC removes 28–30% of the gain MSE (3 dB cell), where unshrunk self-calibration adds error and the tuned gate mostly stays shut (5%). A shrunk classical self-calibration (lam 0.5) removes 39%."*

**(i.3) "As a pre-correction before tuned self-calibration, IABC raises Pd when gain errors are at least 3 dB." — NEEDS REWORDING.**
- g3 alone: +0.003 [−0.000, +0.007], not significant (the original also had it not significant).
- c8_3: +0.005 [+0.001, +0.009] against tuned SC, but +0.003 [−0.001, +0.007] against SC snrgate.
- g6: +0.015 [+0.010, +0.021]; against SC snrgate, +0.014 [+0.009, +0.020].
- c30_6: +0.034 [+0.027, +0.040]; against SC snrgate, +0.021 [+0.016, +0.027].
- Clean: −0.003 [−0.005, −0.001], significant.
- Per SNR, the gain comes from SNR ≥ 10 dB (plus 0 dB at g6).
- The 4 dB figure was not re-measured.

Corrected sentence: *"Placed before classical self-calibration, IABC raises Pd only when per-antenna gain errors reach about 4–6 dB, which is beyond its training range: +0.015 at 6 dB and +0.02–0.03 at 30°/6 dB, mostly at SNR ≥ 10 dB. It costs 0.003 on unimpaired scenes."*

**(i.4) "For gain errors of 4–6 dB, IABR-Net heat beats NOMP-only … beats NOMP+SC tuned only at g6, c20_4, c30_6." — NEEDS REWORDING.**
- Against NOMP: g6 +0.034 [+0.027, +0.040]; at 15 dB, +0.072 [+0.051, +0.093]. The original +0.085 lies inside this CI.
- At g6, heat is significantly worse than NOMP at −10 and −5 dB (−0.017 and −0.019).
- Against tuned SC: g6 +0.013*, c20_4 +0.010*, c30_6 +0.011*. These replicate.
- Against SC snrgate, the c30_6 advantage disappears: −0.002 [−0.009, +0.007]. g6 (+0.012*) and c20_4 (+0.011*) still hold.
- At ph45, heat is worse than NOMP: −0.013*.

Corrected sentence: *"For 4–6 dB gain errors IABR-Net heat beats NOMP-only (+0.034 at 6 dB, up to +0.07–0.11 at 15–25 dB SNR) and beats the best classical self-calibration we tested by about 0.012. It is worse below 0 dB SNR and for large phase errors, and its advantage at 30°/6 dB disappears against SNR-gated self-calibration."*

**(ii.1) "S1 'removes most of the impairment' is false for phase" — CONFIRMED.**
- Phase MSE removed: 10.4% at 8°, 18.0% at SNR ≥ 10 dB, and 1.0–6.1% beyond the training range.

Sub-sentence "Self-cal removes 70–95% at SNR ≥ 10 dB" — **NEEDS REWORDING.** Tuned SC removes only 52.5% at 45° and 29.8% at 30°/6 dB. Corrected: *"…tuned self-cal removes 70–95% at SNR ≥ 10 dB for phase errors up to 30°. Where its gate stays shut (45°, 30°/6 dB), an SNR-gated self-cal removes 92–97%."*

**(ii.2) "'IABC works better than classical self-cal' is false at SNR ≥ 10 dB in every cell" — REFUTED as worded.**
- 30°/6 dB, gain: IABC removes 44.9% against 29.9% for tuned SC. Residual MSE difference −1.69 [−2.04, −1.35] dB².
- L=1, 3 dB, 15 dB: 88.6% against 86.1%, difference −0.070 [−0.118, −0.030] dB².
- The Pd half of the claim replicates. IABC+NOMP is significantly below NOMP+SC tuned at clean, ph8, g3 and ph45, and above it only at c30_6 (+0.018 [+0.011, +0.024]).

Corrected sentence: *"At SNR ≥ 10 dB classical self-calibration removes more identifiable error than IABC in every cell we tested, except the gain error at 30°/6 dB, where the tuned gate stays shut (IABC still loses to SNR-gated self-cal: 45% vs 95%), and single-path scenes at 3 dB (89% vs 86%)."*

**(ii.3) "The network is needed" is false on the official ideal-hardware bank — CONFIRMED.**
- Recomputed independently: −0.0082 [−0.010, −0.007].
- Fresh clean L=3 scenes: IABC+NOMP −0.013*, heat −0.011*, dual −0.006 [−0.009, −0.002] against NOMP.

**(ii.4) "IABC never hurts" is false — CONFIRMED.**
- Clean: −0.013 [−0.016, −0.009]. The harm is concentrated at 10–25 dB SNR (−0.015 to −0.035 per SNR).
- On clean scenes IABC adds 0.44° / 0.21 dB, as mean per-scene RMS (the original's 0.44° / 0.19 dB metric).
- The bank-level numbers (27/55 banks, close spacing, tail SNR) were not re-measured. The official bank row was reproduced exactly.

**Recommendation sentence "the oracle can remove 95–99% at SNR ≥ 10 dB" — NEEDS REWORDING.**
- That range holds only for errors of at least 3 dB or 20°.
- The oracle removes 70% at 8° and 59% at 1 dB.

Also, a redesigned IABC should have to beat both SNR-gated and shrunk classical self-calibration, not only tuned SC.

## 4. Things that cut against the report's framing (null/negative)
- On clean scenes, tuned SC adds more per-antenna error than IABC: 2.0–4.7° at 0–15 dB, against 0.3–0.7° for IABC (`a4_clean_added.log`). Yet tuned SC costs less Pd than IABC (−0.003 against −0.013 relative to NOMP). Per-antenna error is a poor proxy for Pd here.
- IABC's phase output at 0 dB (+2.6%) beats every shrunk classical phase estimate (−16% or worse). The honest phase baseline at 0 dB is "no correction" (0%).
