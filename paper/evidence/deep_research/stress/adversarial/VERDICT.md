# Adversarial check of the stress tests (M1 to M8)

Folder `paper/evidence/deep_research/stress/adversarial/`. I read every script in `stress/` and wrote three scripts of my own. Each one's stdout is saved in the `.log` next to it:

- `xcheck.py`: an independent forward model (`adv_common.py`), written from the spec, compared with `stress_sim.generate` on the same random draws.
- `replicate.py`: fresh-seed replication of the headline cells. It uses my own generator and draw scheme, N = 1000 scenes per bank (2x the original), base seeds 9000 / 9015, and paired bootstrap (2000 resamples, seed 0).
- `m7_check.py`: an independent recomputation of M7 on the cached banks, plus a fresh bank (seeds 9600 / 9615).

No existing file was modified and nothing was committed. Large arrays are in `cache/`. "Fresh" below means the new bank; "orig" means `stress/run_stress.log`.

## 1. Simulator: no implementation bug found

My forward model was written independently: `scipy.linalg.toeplitz` coupling, `hypot` near field, angle-rounding phase shifters, and an offset-index ADC. On the same draws it matches `stress_sim.generate` to a relative error of ≤ 1.4e-13 for:
- clean
- M1 ρ 0.1 / 0.3
- M2 σ 0.05 / 0.2
- M4 b = 4, 3, 2, 1
- M5 b = 4, 3, 2, 1
- M6 10° / 30°
- M8 2 m / 0.5 m

`to_ant` matches to 7e-16 (`xcheck.log`).

The spec items are implemented as stated:
- **Coupling (M1):** `C_r H C_t^T`. `C_t` is complex symmetric, so the transpose is a no-op and the model is reciprocal.
- **Phase-shifter quantisation (M4):** applied to both W and F, with ties rounded up.
- **ADC (M5):** per-scene peak AGC.
- **Phase noise (M6):** iid on each of the 256 measurements.
- **Position errors (M2):** enter through `cos(angle)·x_n`, so they are angle dependent.
- **Near field (M8):** far-field limit is correct.
- **Pipeline:**
  - The same Y feeds all methods, and L = 3 is passed to every method.
  - Only `oracle-seeded` sees the ground truth, and it is labelled as such.
  - L = 3 banks contain no NaN angles.
  - The evaluator is the original `prepare_for_metric` / `get_ang_difference`.

Modelling notes (these are choices, not bugs):
- Using a Hermitian instead of a symmetric coupling Toeplitz would change Y by 53% at ρ = 0.3. The symmetric (reciprocal) choice is the physical one, but RESULTS.md should state it.
- **Unlogged number:** "peak/RMS ≈ 12" (`stress_sim.py:137`, `report.py:36`, `RESULTS.md:8`) is not printed by any logged script. Measured here, the median is **12.19 at 15 dB but 8.47 at 0 dB**, and 3-RMS AGC clips 1.8% / 1.2% of the real components (`xcheck.log`). The claim holds at 15 dB only.

## 2. Fairness: one material issue, the self-calibration gate (affects the "beats NOMP+selfcal" claims)

`SC = (1.0, 5.0)` (`new_results/common.py:19`) gates self-calibration on model-to-residual energy > 5 (`notebook_src.py:616`). Under large impairments this gate is closed on the raw channel but open on the CNN-corrected channel. The fraction of scenes where the gate opens (`replicate.log`):

| bank | Hant (NOMP+selfcal) | Hc (network methods) |
|---|---|---|
| gain 6 dB, 15 dB | 0.02 | 0.80 |
| phase 30° + gain 4 dB, 15 dB | 0.00 | 0.13 |
| every bank at 0 dB, including clean | 0.00 | 0.00 |

So "NOMP+selfcal" is effectively plain NOMP in exactly the cells where the networks "win", and at 0 dB self-calibration is a no-op for every method.

I added two variants:
- **tau 0 (gate always open):** a priori, no tuning. It helps at 15 dB but hurts badly at 0 dB.
- **Tuned selfcal (λ 1, τ 2, 4 iterations):** tuned only on separate tuning banks (seeds 9900 / 9915, M3 gain 6 and phase 30 + gain 4). It scores 0.7150 on the tuning banks against 0.6945 for the default.

Other fairness checks passed:
- No method sees ground truth except oracle.
- M7 constants (t = 0.45, κ_b = 13) were tuned on separate banks; I re-used them frozen on fresh banks.
- NOMP gets the same L as the networks.

**Attribution:** `IABR-v2 dual` gives the same Pd as "NOMP on Hc" (greedy NOMP on the CNN-corrected channel) to within 0.001 in every fresh cell. The dual method's wins therefore come from the CNN's correction stage, not from heat-map seeding.

## 3. Verdicts on the headline claims (fresh seed, N = 1000, paired 95% CI)

### Network wins over both NOMP and NOMP+selfcal

| claim (orig) | fresh vs NOMP | fresh vs NOMP+selfcal (default) | fresh vs tuned selfcal | verdict |
|---|---|---|---|---|
| gain 6 dB, 15 dB, heat +0.071 / +0.069 | +0.068 [+0.059, +0.077] | +0.067 [+0.057, +0.076] | **+0.028 [+0.020, +0.036]** | **CONFIRMED but overstated.** It reproduces exactly against the default baseline. Against a working self-calibration the win is about 40% of what was claimed. |
| gain 6 dB, 15 dB, dual +0.064 / +0.062 | +0.059 [+0.050, +0.068] | +0.058 [+0.049, +0.066] | **+0.019 [+0.012, +0.027]** | **CONFIRMED but overstated** (same reason). |
| phase 30° + gain 4 dB, 15 dB, heat +0.020 / +0.021 | +0.027 [+0.018, +0.037] | +0.027 [+0.018, +0.036] | **−0.004 [−0.014, +0.007]** | **REFUTED** as a win over self-calibration. It reproduces only against the gated-off baseline; tau 0 gives −0.004 as well. |
| phase 30° + gain 4 dB, 15 dB, dual +0.024 / +0.025 | +0.025 [+0.017, +0.033] | +0.024 [+0.017, +0.033] | **−0.006 [−0.015, +0.002]** | **REFUTED** (same reason). |
| phase 30° + gain 4 dB, 0 dB, dual +0.009 / +0.009 | +0.008 [+0.002, +0.013] | +0.007 [+0.001, +0.013] | +0.007 [+0.001, +0.013] | **CONFIRMED** (tiny; comes from Hc; self-calibration is inactive at 0 dB). |
| 2-bit phase shifters, 15 dB, dual +0.015 / +0.012 | +0.008 [+0.002, +0.015] | +0.010 [+0.005, +0.016] | +0.012 [+0.005, +0.020] | **CONFIRMED** (smaller; the original +0.015 is at the edge of the fresh CI). |
| 2-bit phase shifters, 0 dB, dual +0.006 / +0.006 | +0.006 [+0.002, +0.009] | +0.006 [+0.002, +0.009] | +0.006 [+0.002, +0.009] | **CONFIRMED.** |

New in the fresh run: gain 6 dB at 0 dB, dual +0.010 [+0.003, +0.017] vs NOMP. The original was +0.005 and not significant. This does not change any verdict.

### Networks beat NOMP but not NOMP+selfcal

- **Coupling ρ = 0.3, 15 dB: REFUTED (did not reproduce).**
  - IABR-Net heat − NOMP: +0.003 [−0.003, +0.010] (orig +0.017 [+0.007, +0.028]).
  - IABR-v2 dual − NOMP: +0.000 [−0.006, +0.006] (orig +0.013).
- **Phase 20°, 15 dB: CONFIRMED.**
  - heat − NOMP: +0.018 [+0.011, +0.026]; dual − NOMP: +0.015 [+0.009, +0.021].
  - Neither beats self-calibration (heat − selfcal +0.003 [−0.001, +0.007]).
- Phase 30° at 15 dB (phase only): not re-run.

### "No network win in M2, M5, M5c, M6, M8": CONFIRMED for the cells re-run (M2 σ 0.05 / 0.2, M5 b 4 / 1, M6 30°, M8 0.5 m; M5c was not re-run)

No cell beats both NOMP and NOMP+selfcal.
- **Phase noise 30° at 15 dB:** dual − selfcal is +0.009 [+0.003, +0.015], but only because the default self-calibration *hurts* there (−0.010 vs NOMP). Dual − NOMP is −0.001.

### Where the heat-seeded network collapses: CONFIRMED

- 2-bit phase shifters, 15 dB, heat − NOMP: −0.091 [−0.105, −0.077] (orig −0.104).
- Near field 0.5 m, 15 dB, heat − NOMP: −0.061 [−0.075, −0.049] (orig −0.071).

### NOMP degradations vs clean (paired): CONFIRMED in sign and size

The fresh clean NOMP is 0.938 at 15 dB against 0.929 in the original, so fresh losses at 15 dB run about 0.01 to 0.02 larger. In a few cells the original point estimate therefore falls just outside the fresh CI.

| cell | fresh | orig |
|---|---|---|
| phase 45°, 15 dB | −0.340 [−0.355, −0.324] | −0.324 |
| phase 45°, 0 dB | −0.196 | −0.207 |
| position error σ 0.2, 15 dB | −0.325 | −0.302 |
| position error σ 0.2, 0 dB | −0.182 | −0.171 |
| 1-bit phase shifters, 15 dB | −0.653 | −0.634 |
| 1-bit phase shifters, 0 dB | −0.497 | −0.499 |
| 1-bit ADC, 15 dB | −0.756 | −0.750 |
| 1-bit ADC, 0 dB | −0.537 | −0.521 |
| 4-bit ADC, 15 dB | −0.167 | −0.156 |
| 2-bit phase shifters, 15 dB | −0.194 | −0.174 |
| 2-bit phase shifters, 0 dB | −0.115 | −0.115 |
| phase 20°, 15 dB | −0.121 | −0.101 |
| phase 20°, 0 dB | −0.044 | −0.047 |
| phase noise 30°, 15 dB | −0.173 | −0.172 |
| phase noise 30°, 0 dB | −0.097 | −0.109 |
| phase 30° + gain 4 dB, 15 dB | −0.235 | −0.228 |
| gain 6 dB, 15 dB | −0.115 | −0.116 |

The "> 0.05 loss" thresholds hold, but three are borderline because their CIs straddle −0.05:
- coupling ρ 0.3, 15 dB: −0.057 [−0.066, −0.048]
- near field 0.5 m, 15 dB: −0.058 [−0.067, −0.050]
- position error σ 0.05, 15 dB: −0.066 [−0.074, −0.058] (this one clears −0.05)

The breakpoints at the other SNR also hold:
- coupling ρ 0.3 at 0 dB: −0.029 (stays under 0.05)
- near field 0.5 m at 0 dB: −0.024 (stays under 0.05)
- position error σ 0.05 at 0 dB: −0.012 (stays under 0.05)
- 4-bit ADC at 0 dB: −0.026

## 4. Unknown number of paths (M7): independent recomputation

**Cached banks.** My code recomputes L̂ from the stored T / E arrays for rules (a), (a'), blind CFAR, MDL and AIC. In all 10 cases (5 rules × 2 SNRs) the result is identical to the stored L̂. P(L̂ = L), recall and precision equal `m7_order.log` exactly. AIC returns L̂ = 10 for all 1000 scenes at both SNRs.

**Threshold note: CONFIRMED.** The realised false-alarm rate of κ = 10.145 is 0.0732 on the 4x oversampled grid and 0.0112 on the 16x16 grid. Re-deriving κ_MC with a fresh seed gives 12.27 (orig 12.32).

**Fresh banks.** 1000 scenes per SNR, constants frozen, my own maximum-cardinality matching (both angles within 1°).

| rule | 15 dB P(L̂=L) / recall / precision | 0 dB P(L̂=L) / recall / precision | orig 15 dB / 0 dB P(L̂=L) |
|---|---|---|---|
| (a) oracle CFAR | 0.880 / 0.836 / 0.834 | 0.478 / 0.448 / 0.541 | 0.890 / 0.491 |
| (a') MC κ | 0.931 / 0.835 / 0.845 | 0.443 / 0.442 / 0.569 | 0.941 / 0.442 |
| blind CFAR κ_b 13 | 0.924 / 0.835 / 0.844 | 0.448 / 0.442 / 0.561 | 0.933 / 0.457 |
| MDL | 0.910 | 0.467 | 0.922 / 0.463 |
| AIC | 0.000 (always L̂ = 10) | 0.000 | 0.000 / 0.000 |
| heat count t 0.45 | 0.857 | 0.470 | 0.868 / 0.476 |
| NOMP, known L (recall) | 0.835 | 0.458 | 0.852 / 0.444 |

Verdicts:
- **(a) gets L right 0.890 / 0.491 of the time, with recall close to known-L NOMP: CONFIRMED** (0.880 / 0.478; recall 0.836 vs 0.835 at 15 dB and 0.448 vs 0.458 at 0 dB).
- **The blind threshold is the best blind rule: CONFIRMED** at 15 dB (0.924, against 0.910 for MDL). At 0 dB, MDL (0.467) edges out the blind threshold (0.448).
- **AIC never stops: CONFIRMED.**
- **Heat count is not better than (a) at 15 dB: CONFIRMED.** −0.023 [−0.046, +0.000]. It is worse than the blind rule: −0.067 [−0.087, −0.048].
- **Heat count beats (a') at 0 dB: CONFIRMED.** +0.027 [+0.007, +0.048].
- **Heat count does not beat the blind rule at 0 dB: UNCERTAIN.** The fresh result is +0.022 [+0.003, +0.042], now significant; the original was +0.019 [−0.001, +0.040]. Both runs are borderline.
- **Heat count does not beat (a) at 0 dB: CONFIRMED.** −0.008 [−0.027, +0.011].

## 5. What this means for the fine-tuning recommendation

1. **Gain errors beyond 3 dB at high SNR remain the one robust network win,** but the honest size is about +0.02 to +0.03 against a self-calibration that is actually switched on, not +0.07.
2. **The phase 30° + gain 4 dB win at 15 dB disappears once self-calibration is enabled.** The "network beats NOMP+selfcal" table should either use a tuned or ungated self-calibration baseline, or state that the τ = 5 gate is closed for NOMP (2% / 0% of scenes) but open for the network (80% / 13%).
3. **The coupling ρ 0.3 win over NOMP did not reproduce.**
4. **Self-calibration never runs at 0 dB under τ = 5,** for any method, so every 0 dB "selfcal" number is plain NOMP.
5. **The default self-calibration lowers Pd under phase noise** (30° at 15 dB: −0.010 vs NOMP).
6. **The dual method's gains come from the CNN correction stage.** Hc plus plain NOMP gives the same Pd, so fine-tuning effort belongs in the correction stage, not the heat map.
