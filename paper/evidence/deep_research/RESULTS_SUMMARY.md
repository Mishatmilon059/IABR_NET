# IABR-Net deep research: consolidated results (saved 2026-10-15)

This file collects every **verified** result of the deep-research runs of 2026-10-15 in one place, so it can be reused later (paper writing, a new Claude session, another laptop). Each number comes from a logged script in the folders listed in section 10. "Adversarially checked" means a separate agent re-ran the claim on fresh random seeds and tried to refute it.

**Status (updated 2026-10-16 00:30):** the main research run is COMPLETE (11/11 stages). Final synthesis with ranked journal framings A/B/C: `REPORT.md`; final adversarial verdict: `finetune/adversarial/VERDICT.md` (section 11 below). Still running: unfolded-decoder v2 (`unfolded_v2/`).

Metric conventions: paper-style Pd = fraction of angle components with |error| <= 1 deg (scenes with fewer than L detections dropped); strict Pd counts those scenes as misses. 95% CIs are paired bootstrap, 2000 resamples within SNR groups, seed 0. `*` = CI excludes 0.

---

## 1. Is NOMP alone enough? (verify/, adversarially consistent)

An independent from-scratch NOMP matches the cached NOMP-only predictions scene by scene on all 10 checked banks (max difference 5.6e-14 rad); an independent metric matches `common.py` exactly.

| Official bank (8000 scenes, ideal hardware, L=3, SNR -10..25) | Pd paper | Pd strict | RMSE (deg) |
|---|---|---|---|
| **NOMP only (no network)** | **0.7581** | 0.7581 | 0.3277 |
| NOMP + self-calibration | 0.7576 | 0.7576 | 0.3288 |
| IABR-v2 dual (network + NOMP) | 0.7551 | 0.7551 | 0.3337 |
| IABR-Net heat-seeded (classical decoder) | 0.7503 | 0.7503 | 0.3329 |
| IABR-Net all-neural (CNN + learned decoder) | 0.7508 | 0.7508 | 0.3324 |
| U-Net (Lloria et al., TVT 2026) | 0.7366 | 0.6940 | 0.3611 |
| ResNet (Lloria et al., validated port) | 0.7103 | 0.6728 | 0.3762 |
| Network output only | 0.4507 | 0.4507 | 0.5660 |

- Heat-seeded - NOMP: -0.0078 [-0.0095, -0.0061]*; dual - NOMP: -0.0030 [-0.0043, -0.0016]*; NOMP - U-Net: +0.0215 [+0.0181, +0.0248]*.
- Oracle-seeded decoder (true angles as start) 0.8261, but only +0.0051 of that gap is NOMP search loss; the rest is the objective itself at low SNR (a better start cannot fix it).
- **Conclusion:** on ideal hardware the network does not beat NOMP. "The network is needed" cannot be claimed for the official bank.

## 2. Where NOMP breaks (stress/, stress/adversarial/)

NOMP loss vs clean hardware at 15 dB (same scenes): position error sigma=0.05 -0.058, sigma=0.2 -0.30; phase 20 deg -0.10, 45 deg -0.32; gain 6 dB -0.12; 2-bit phase shifters -0.17, 1-bit -0.63; ADC 4-bit -0.16, 1-bit -0.75; phase noise 20 deg -0.08, 30 deg -0.17; coupling rho=0.3 -0.06 (borderline); near field 0.5 m -0.06. Fair classical baseline = NOMP + **tuned** self-calibration (lam 1, tau 2, 4 iters); the default gate (tau 5) shuts off under large impairments. The original network beats it only at gain 6 dB / 15 dB (+0.028 heat); the dual method's gain comes from the IABC correction, not the heatmap seeds.

## 3. Old IABC (iabc/, iabc/adversarial/) - verdict: drop / replace

- Removes ~10% of the identifiable phase error (MSE) at 8 deg, 1-6% beyond its training range; ~50% of the gain error at 3 dB (RMS 1.68 -> 1.18 dB); 77% at SNR >= 10 dB. Tuned classical self-cal removes 70-95% at SNR >= 10 dB.
- Hurts ideal hardware: IABC+NOMP - NOMP on the official bank -0.0082 [-0.0096, -0.0069]*; worse in 27 of 55 cached banks, better in 1 (gain_g4_snr15 +0.015).
- Root cause (diagnosed): its phase input is only valid for one path at high SNR; with L >= 2 it is garbage, so the 458-parameter MLP learned to ignore it.

## 4. IABC-v2 (iabc_v2/, iabc_v2/adversarial/) - verdict: big estimation improvement, small detection effect

- Architecture: v2a = 597k-parameter transformer over 32 antenna tokens (rows/columns of Ht and of the covariance matrices + old features); v2b = v2a -> NOMP -> classical self-cal -> 257k-parameter learned refiner (854k total). Checkpoints `iabc_v2/cache/v2a_best.pt`, `v2b_best.pt`.
- Cramer-Rao bound (L=3, RMS over 32 antennas), phase: 38 / 12.7 / 4.2 / 0.74 deg and gain 6.1 / 2.0 / 0.67 / 0.12 dB at SNR -10 / 0 / 10 / 25 dB (verified by finite differences and a Monte-Carlo ML check). Below ~5 dB SNR no corrector can remove 8 deg / 3 dB errors.

| % of identifiable MSE removed, SNR >= 5 dB | old IABC | IABC-v2b | tuned self-cal | classical EB-ML (adversarial) | approx. bound |
|---|---|---|---|---|---|
| 8 deg phase | 15 | 67-68 | 50 | 59 | 74 |
| 20 deg phase | 5 | 91 | 85 | 89 | 92 |
| 45 deg phase | 1 | 97 | 43 | 95 | 98 |
| 3 dB gain | 73 | 90 | 84 | 89 | 91 |
| 6 dB gain | 50 | 96.5 | 80 | 96 | 97 |
| 30 deg + 6 dB | 8 / 44 | 93 / 96 | 26 / 25 | 92.5 / 95.6 | 96 / 97 |

- Downstream Pd, (v2b + NOMP + tuned SC) - (NOMP + tuned SC): 45 deg +0.057*, 30 deg/6 dB +0.049*, 6 dB +0.008 to +0.010*, 20 deg +0.004 to +0.006*; **8 deg/3 dB +0.003 (not significant)**. In IABR-Net (trunk frozen) +0.074* at 45 deg.
- Official bank: v2b + NOMP -0.001 [-0.002, +0.001] (neutral); full pipeline with tuned SC -0.002*.
- Adversarial: a classical estimator (joint ML + shrinkage + blind-SNR gate, "EB-ML-g5") ties or beats the v2b pipeline in 6 of 7 impaired cells; v2b wins clearly only at 45 deg and in speed (~6.5 ms vs ~37 ms per scene).
- **Realistic small errors (<= 8 deg, <= 3 dB):** NOMP Pd ideal 0.751, 5 deg 0.745, 8 deg 0.734, 3 dB 0.731; a PERFECT corrector at 8 deg only gives 0.739 -> 0.743 (+0.004). So better calibration cannot raise detection much in the realistic range.
- Allowed wording: "IABC-v2 removes 90-97% of the correctable impairment (MSE) at SNR >= 5 dB, close to the Cramer-Rao bound, about 6x faster than the strongest classical ML self-calibration." Not allowed: "IABC is essential" / "beats the best classical method in general".

## 5. All-neural pipeline vs published networks (nn_decoder/, nn_decoder/adversarial/)

- Pipeline: CNN (IABC + heatmap) -> heatmap peaks -> learned unfolded decoder (lurd.pt, 4,803 params: one shared MLP 7->64->64->2 + learned self-cal strength), no FFT search, no explicit Newton formula. Must be described as "unfolded from NOMP": heatmap seeding, LS gains, residual subtraction and the self-cal gate are fixed classical components; the MLP step correlates 0.91-0.92 with the Newton step.
- Official bank: 0.7508; vs U-Net +0.0142 [+0.0108, +0.0176]* (strict +0.0568*); vs ResNet +0.0404 [+0.0369, +0.0439]*; vs NOMP -0.0073 [-0.0091, -0.0056]*; vs classical-decoder IABR-Net +0.0005 (no difference). Held-out 6000 scenes give the same conclusions.
- Per SNR vs U-Net: significant at -10, -5, 15, 20, 25 dB; not at 0, 5, 10 dB.
- Latency (1 thread, batch 1): NOMP 4.8-6.9 ms; CNN + classical decoder 12.3 ms; CNN + learned decoder ~98-101 ms (unrolled torch implementation).

## 6. Low-resolution ADC (finetune/adc/, nn_decoder_adapt/, finetune/adversarial/) - REVISED by the adversarial check

The first ADC stage found that the ADC-fine-tuned CNN read out directly beat every classical method it tried (+0.13 to +0.18 Pd at 1-2 bits / 15 dB vs quantisation-consistent NOMP). The final adversarial check added a stronger classical baseline: maximum-likelihood refinement under the exact quantiser likelihood (probability that each real/imag value falls in its observed ADC cell; known noise variance), with 13 starts + EM. Fresh seeds, 1000 scenes per cell (`finetune/adversarial/s3_adc_test.log`):

| cell | NOMP | consistent NOMP | network read-out alone | classical ML (all starts) | hybrid: ML from network seeds | ML from truth (reference) |
|---|---|---|---|---|---|---|
| 1-bit, 15 dB | 0.191 | 0.237 | 0.419 | 0.540 | **0.572** (+0.032 [+0.022, +0.042]* vs ML) | 0.626 |
| 2-bit, 15 dB | 0.289 | 0.451 | 0.593 | 0.682 | 0.687 (+0.005, n.s.) | 0.719 |
| 1-bit, 0 dB | 0.163 | 0.226 | 0.239 | 0.275 | 0.274 (-0.001, n.s.) | 0.439 |
| 2-bit, 0 dB | 0.373 | 0.422 | 0.438 | 0.476 | 0.471 (-0.004, n.s.) | 0.540 |

Latency, ms per scene (`s6_latency.json`): NOMP 5.8; network read-out 6.2; consistent NOMP 76.9; ML from one start 93.8; ML all starts 482.4. Hybrid = network + one ML refinement (about 100 ms) = about 5x cheaper than classical multi-start ML.

- REFUTED: "the network alone beats every classical method" - quantisation-aware ML is +0.036 to +0.12 better.
- CONFIRMED: the hybrid (network seeds + quantisation-aware ML) matches classical multi-start ML in every cell at about 1/5 of its cost, and beats it at 1-bit / 15 dB (+0.032*).
- The network read-out alone reaches 78-92% of ML's Pd at about 1/80 of its cost and needs no noise-level knowledge; ML loses 0.04-0.07 when its assumed SNR is wrong by 5 dB (e.g. 2-bit / 15 dB: 0.682 -> 0.606-0.615).
- At 3-4 bits the classical decoder is better (from the ADC stage) - use a bit-aware switch.
- Caveats: one quantiser model (peak AGC, uniform); true L given.
- Low-resolution ADCs (1-3 bits) are a realistic, widely studied mmWave power-saving design. Reading list: SignalNet (Dreifuerst & Heath, IEEE TSP 2022), Mo-Schniter-Heath few-bit mmWave channel estimation (arXiv:1610.02735), Li et al. one-bit massive MIMO (IEEE TSP 2017), Walden ADC survey (IEEE JSAC 1999), Barthelme & Utschick (IEEE TSP 2021).

## 7. Severe-impairment fine-tuning (finetune/imp/REPORT.md; final adversarial check pending)

Trained 3300 steps (61.7 min) on phase <= 45 deg, gain <= 6 dB, position errors, phase noise, coupling. Only fresh-seed replications are claimed (wins were searched post hoc among 11 learned variants):

| cell (fresh seeds, 1000 scenes) | NOMP | fine-tuned net seeds on raw Ht + SC | diff vs best classical | oracle-seeded |
|---|---|---|---|---|
| position error sigma 0.2, 15 dB | 0.627 | 0.662 | +0.035 [+0.028, +0.043]* | 0.690 |
| phase noise 30 deg, 15 dB | 0.761 | 0.807 | +0.046 [+0.035, +0.057]* | 0.839 |
| phase noise 20 deg, 15 dB | 0.861 | 0.877 | +0.016 [+0.009, +0.025]* | 0.902 |
| phase noise 30 deg, 0 dB | - | - | +0.016 [+0.009, +0.022]* | - |
| near field 0.5 m, 15 dB (out of distribution) | - | - | +0.008* | - |

- The gain comes from the learned seeds (better starting points), not from the learned correction; mild impairments lose about 0.01-0.02*; the winning heat-only path costs -0.006* on the official bank (no impairment-aware switch yet).
- Large phase/gain (M3): replicated wins (+0.066* at 45 deg / 0 dB, +0.025* at 30 deg / 0 dB) all come from the IABC-v2b corrector, not from fine-tuning. Coupling (M1): no win.
- Low-resolution phase shifters (M4): NETWORK NOT NEEDED - decoding with the receiver's known quantised codebook closes 93-100% of the gap.

## 8. Literature (lit/)

- `lit/baselines_survey.md`: of 20+ learned-method papers checked, only two (Han et al. JSAC 2020; Li et al. TWC 2025) run NOMP, and both keep Newton/NOMP refinement inside their pipeline. Lloria et al. compared only TSDCE/DFT-CEA (baseline lineage); several papers call NOMP slow although the NOMP paper reports it faster than AST/Lasso/MUSIC. No paper compares a pure network against NOMP for analog-mmWave joint AoA/AoD.
- `lit/when_dl_helps.md`: DL beats model-based methods under unmodelled hardware effects, nonlinear/quantised measurements, unknown model order, few snapshots; gains vanish against matched gridless/ML estimators. 33 verified references.

## 9. What can and cannot be claimed (current state)

Can: the pipeline beats the published U-Net (+0.014, strict +0.057) and ResNet (+0.040) on their benchmark; NOMP-only is a stronger baseline than those networks (credit Mamandipoor et al. 2016); IABC-v2 calibrates close to the CRB; with 1-2-bit ADCs the hybrid (network seeds + quantisation-aware ML) matches classical multi-start ML at about 1/5 of its cost and beats it at 1-bit / 15 dB; the network read-out alone is about 80x faster than ML at 78-92% of its accuracy.
Cannot: that the network beats NOMP on ideal hardware; that IABC (old or new) is essential; that the learned decoder is "fully neural" without saying it is unfolded from NOMP. Hiding NOMP would fail review.

## 10. File index

All under `D:/ai_ml_project/paper/evidence/deep_research/`:
- `verify/` REPORT.md, verify.json, tables.md (section 1)
- `stress/` RESULTS.md, results.json, figures, stress_sim.py; `stress/adversarial/VERDICT.md` (section 2)
- `iabc/` tables.md, s5_summary.log, results.json, figures; `iabc/adversarial/VERDICT.md` (section 3)
- `iabc_v2/` s_summary.log, results.json, criteria.json, crb.json, figures, cache/v2a_best.pt, v2b_best.pt; `iabc_v2/adversarial/VERDICT.md` (section 4)
- `nn_decoder/` REPORT.md, results.json, tables.md, fig_pd_vs_snr.png; `nn_decoder/adversarial/VERDICT.md` (section 5)
- `finetune/adc/` REPORT.md, results.json, fig_adc.png, cache/adc_ft.pt; `nn_decoder_adapt/` REPORT.md (section 6)
- `finetune/imp/` results.json, eval logs, figures, cache/imp_ft.pt (section 7, report pending)
- `lit/` baselines_survey.md, when_dl_helps.md (section 8)
- Large prediction caches and generated training data (regenerable from fixed seeds): permanent copy at `C:/Users/Milon/iabr_deep_research_cache/` (dr_cache/, v2b_data/); the original session scratchpad copy is temporary.
- Workflow scripts (resumable): `D:/ai_ml_project/paper/workflows/` (deep-research-resume-sequential-wf_57f76d2f-fc7.js, run id wf_57f76d2f-fc7); keep-awake helper `keepawake.py`.

## 11. Final adversarial verdict on the fine-tuning claims (finetune/adversarial/VERDICT.md, 2026-10-15 ~23:55)

- ADC: "network alone beats every classical method" REFUTED (quantiser ML better by +0.089*, +0.121*, +0.038* at 2-bit/15, 1-bit/15, 2-bit/0 dB; holds even with a 5 dB noise-level mismatch). Fine-tuning through the decoder causes no regression (CONFIRMED). The adapted learned decoder does not beat quantiser ML (REFUTED).
- NEW, CONFIRMED (H1): the network output used as a starting point for a physics-model ML gives the best Pd. ADC 1-bit / 15 dB: +0.032* (seed 8715) and +0.028* / +0.036* (replicated on seed 9515) over multi-start quantiser ML; 2-bit / 15 dB +0.011*. Phase noise: network seeds + phase-noise-aware ML (PN-ML) beat PN-ML alone by +0.012* (single bank).
- Phase noise (M6) "learned seeds win": REFUTED against PN-ML (network alone -0.011* to -0.026* vs PN-ML), although it still beats NOMP / simple classical methods.
- Position error (M2, sigma 0.2, 15 dB): CONFIRMED against the classical methods built here: +0.044* vs NOMP, +0.055* vs a position-aware MAP (POS-MAP); POS-MAP started from the network seeds is +0.038* better than POS-MAP's own multi-start.
- Large phase/gain (M3): the wins belong to IABC-v2b, not to fine-tuning. Fine-tuned pipelines cost 0.004-0.008 Pd on ideal hardware (CONFIRMED).
- Overall pattern for the paper: where hardware breaks NOMP, the best system is "network finds the starting points, an impairment-aware physics model refines" - better than either part alone.


## 12. ADC 1-4 bit test with unknown SNR (evidence: paper/evidence/adc_bits_test/, run 2026-11-10)

Fresh scenes (seeds 14000/14010/14020 for 0/10/20 dB, 150 scenes per cell, 450 per bit depth), L=3, peak-AGC uniform ADC, ideal hardware, same scenes for all methods. Pooled over SNR (the methods do not know each scene's SNR). Strict Pd equals paper-style Pd. CNN = your ADC-fine-tuned CNN; hybrid = quantiser ML started from the CNN seeds.

| method (pooled Pd) | 1 bit | 2 bits | 3 bits | 4 bits |
|---|---|---|---|---|
| NOMP | 0.192 | 0.329 | 0.566 | 0.753 |
| consistent NOMP | 0.235 | 0.438 | 0.656 | 0.780 |
| CNN read-out (your network alone) | 0.343 | 0.528 | 0.588 | 0.556 |
| quantiser ML, noise level known | 0.438 | 0.617 | 0.743 | 0.819 |
| quantiser ML, noise estimated from data | 0.410 | 0.570 | 0.730 | 0.809 |
| quantiser ML, fixed guess 10 dB | 0.389 | 0.508 | 0.659 | 0.775 |
| hybrid, noise level known | 0.456 | 0.615 | 0.739 | 0.810 |
| hybrid, noise estimated | 0.428 | 0.576 | 0.725 | 0.803 |

- CNN read-out minus NOMP: +0.151*, +0.199*, +0.022 (n.s.), -0.196* at 1 / 2 / 3 / 4 bits. CNN minus quantiser ML (known noise): -0.095*, -0.089*, -0.154*, -0.262*. Against ML with a fixed 10 dB guess: -0.046*, +0.020 (n.s.), -0.071*, -0.219*.
- Hybrid minus quantiser ML with the same noise assumption: +0.018* at 1 bit, -0.002 / -0.003 (n.s.) at 2 / 3 bits, -0.009* at 4 bits. The hybrid is the best method only at 1 bit; quantiser ML with known noise is best at 2-4 bits.
- Pd lost by quantiser ML when the noise level is wrong (+-5 dB): 0.035-0.036 / 0.066 / 0.046-0.055 / 0.029-0.030; fixed 10 dB: 0.049 / 0.109 / 0.084 / 0.044; data-estimated noise: 0.028 / 0.046 / 0.013 / 0.009.
- Decomposition of the hybrid-minus-NOMP gap (1 / 2 / 3 / 4 bits): quantiser-aware step alone (ML minus NOMP) +0.246 / +0.288 / +0.177 / +0.066; extra from the network (hybrid minus ML) +0.018 / -0.002 / -0.003 / -0.009. Most of the gain over NOMP is prior art (the exact cell likelihood: Mo, Schniter & Heath 2018; GNOMP), not the network.
- Latency ms/scene (1 thread): NOMP about 2.4 (the 3-bit value 10.4 is timing noise), CNN 4-6, consistent NOMP 30-49, hybrid 18-37, quantiser ML (known noise) 256-450.
- Caveats: the wrong-noise and estimated-noise ML variants use 13 starts without the EM start; known-noise ML uses 13 starts plus EM; ADC-only (no per-antenna hardware errors); true L given; one quantiser model.

## 13. Code audit and NOMP-limitations study (evidence: paper/evidence/audit_2026-11/, 2026-11-10)

- CODE_AUDIT.md: no bug in physics, labels, metric or decoder. Network-only 0.4507 is a resolution problem (no sub-cell supervision; parabola fitted to probabilities). Free fix: parabola on log-probability gives 0.5421. Four design weaknesses: heat target without sub-cell information, 3x3 peak picking misses close paths (33.6% vs NOMP 1.9%), IABC costs 0.003 on ideal hardware, training data matches the test condition in only 1.9% of samples.
- NOMP_LIMITATIONS_AND_DECODER_PLAN.md: realistic NOMP limitations ranked; target scenarios (few-bit ADC with unknown noise; reduced-overhead sweeps; phase-incoherent sweeps); recommended decoder (proposal network + unfolded NOMP + model-based verification).

## 14. Realistic hardware error ranges (sources)

Axiro F5288 short-form datasheet Rev. 2.00 (Dec. 25, 2025), p. 1: 1 / 1.4 deg typical TX / RX RMS phase error, 0.14 / 0.09 dB typical TX / RX RMS gain step error (typical, not worst case; gain figure is a step error). Bhatta, Park, Baek, Kim, Sensors 23(13):6124, 2023, doi 10.3390/s23136124: 8-channel 28 GHz CMOS beamformer, RMS phase 3.2 deg, amplitude 0.6 dB (Table 1; body text gives <3 deg / <2 dB for phase states, so the paper is internally inconsistent). Seo, Lee, Lee, Shin, Sensors 24(4):1087, 2024, doi 10.3390/s24041087: 28 GHz GaN 6-bit phase shifter, RMS phase 8.56 deg before and 1.08 deg after calibration; RMS amplitude 2.9 dB before and 3.3 dB after. ADC resolution of commercial 5G equipment was NOT verified (no source opened); research suggests 3-4 bits can suffice at mmWave receivers (Liu et al., IEEE Access 2019 survey; search snippets only).
