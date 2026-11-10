# NOMP limitations under realistic conditions and a decoder plan (literature study, 2026-11-10)

No code was run for this study. "Project" = verified numbers from the notes under `paper/evidence`. Wording of quotes must be re-checked against the PDFs before use. Realistic means datasheet-level hardware (Axiro F5288 short-form datasheet, Rev. 2.00: about 1 / 1.4 deg typical TX / RX RMS phase error, 0.14 / 0.09 dB typical gain-step error) and uncalibrated research parts at most about 3-8.5 deg and 0.6-3 dB (Bhatta et al., Sensors 2023, doi 10.3390/s23136124; Seo et al., Sensors 2024, doi 10.3390/s24041087).

## 1. NOMP's own stated assumptions and limitations

- Separation: "Without a minimum frequency separation condition, the estimation problem can be hopelessly ill-posed." (Sec. II-C; re-check wording).
- Noise level: the stop threshold depends on sigma^2 (Sec. III-A); no estimator is specified.
- Detection theory assumes earlier tones were found correctly; miss analysis models noise only.
- Convergence theory is called pessimistic; the project found the bound vacuous at oversampling 4.
- Compressive measurements (Sec. VII): M = 64 of N = 256 random measurements, K = 13 near the CRB, K = 16 large errors.
- Scope: 1-D, one snapshot, white Gaussian unquantised noise, known measurement matrix; no ADC, phase drift, hardware or wideband model.
- Cost: cyclic refinement dominates, O(Rc Rs K^2 N).

Follow-ups: NOMP-CFAR (Xu et al., IEEE TAES 2023; about 1 dB cost for unknown noise), MNOMP (arXiv 1802.01266), enhanced MNOMP (arXiv 2607.17308, abstract only: grid-restricted detection), GNOMP (Zhu et al., IEEE TAES 2026: few-bit losses 4.8 / 2.3 / 0.9 dB at 1 / 2 / 3 bits in one scenario), Han et al. JSAC 2020 (NOMP over 5 minutes vs YOLO + Newton under 2 s at M = N = 128, L = 10; accuracy "nearly the same"), Li et al. TWC 2025 (near-field NOMP needs an exhaustive polar search; keypoint network about 5 dB NMSE better only after a small NOMP refiner), Marzi / Rasekh / Madhow compressive-beacon work.

## 2. Ranked realistic limitation scenarios

| # | Scenario | Why NOMP degrades | Evidence | Classical fix | Learned advantage in the literature | Expected gain | Difficulty |
|---|---|---|---|---|---|---|---|
| 1 | 1-2-bit ADC with unknown noise level | Gaussian likelihood wrong; noise level not identifiable at 1 bit | Project: 1-bit/15 dB NOMP 0.191, quantiser ML 0.540, network seeds + ML 0.572 (+0.032*, replicated) | Strong: GNOMP, multi-start quantiser ML, EM-VAMP (GNOMP-2D and EM-VAMP not yet run) | Unfolded 1-bit nets match their parents (Hu 2025); LoRD-Net's gap is partly a weak baseline | +0.03 Pd in one cell, ties elsewhere; about 5x cheaper than multi-start ML | Medium |
| 2 | Reduced overhead (16-64 of 256 beam pairs) | No FFT periodogram; high sidelobes; lost beamforming gain | NOMP Sec. VII (K = 16 fails at M/N = 1/4, 1-D); no project data | Medium-strong: compressive NOMP (Marzi), hierarchical search (Alkhateeb 2014), posterior matching (Chiu 2019) | Yes but never against NOMP: Sohrabi 2021, Heng et al., Abdallah 2021 | Unknown | Medium-high |
| 3 | Phase incoherence across the sequential sweep (CFO, phase noise, Doppler) | Atoms assume one constant phase per path | Project: NOMP -0.08 (20 deg), -0.17 (30 deg) at 15 dB; network seeds +0.016 / +0.046 over NOMP, but a phase-noise-aware ML beats the network alone by 0.011-0.026 | CFO as an extra parameter (Swift-Link), phase-aware ML, power-only phase retrieval + NOMP | None | +0.01-0.05 over NOMP, about +0.01 over the matched classical ML | Medium |
| 4 | Clustered / diffuse multipath | Point-path model wrong; model order ill-defined | Project, two paths at 15 dB: NOMP Pd 0.995 (10 deg apart) ... 0.50 (1 deg); every network worse. 3GPP TR 38.901: 12-20 clusters, 20 rays each (partial source) | Partial | None vs NOMP | Unknown | High |
| 5 | Unknown L, weak paths, unknown noise | Threshold needs sigma^2; grid-restricted test | Project M7: P(correct L) blind CFAR 0.933, MDL 0.922, CNN count 0.868 at 15 dB | Strong: NOMP-CFAR, MDL | DeepFreq etc. beat AIC/MDL, none tested against CFAR | about zero at high SNR | Low |
| 6 | Latency at 64-256 antennas | Cost grows with L, grid size, dimensions | Han 2020; project at 16x16: NOMP 5-7 ms, network pipelines 12-100 ms | Coarse-to-fine grids | Yes (speed only) | Speed only | Medium |
| 7 | Low SNR on ideal hardware | Threshold effect of the ML objective | Project: oracle +0.027, legal rules +0.0007 | none needed | none without the benchmark prior | about zero | not a target |
| 8 | Datasheet-level hardware errors; quantised phase shifters | Small dictionary mismatch | Project: 0.751 -> 0.745 (5 deg), 0.734 (8 deg); perfect corrector +0.004; known quantised codebook closes 93-100% | Complete | not needed | zero | not a target |
| 9 | Beam squint, near field at 16 elements | Extra parameter per path | Squint at most 6% of a DFT bin (400 MHz at 28 GHz, calculation); NOMP loses 0.06 only at 0.5 m | Add the parameter | Li 2025 only at N = 1024 | zero at this size | not a target |

## 3. Most defensible target scenarios

A. Few-bit ADC (1-2 bit) with unknown noise level and unknown L: the only setting with a verified, replicated project win. Baselines needed: GNOMP extended to 2-D (CFAR and known-L), multi-start quantiser ML with a +-5 dB mismatch arm, EM-VAMP-seeded ML, SBRI-X (1 bit), NOMP, NOMP-CFAR. Decisive control: learned seeds vs EM-VAMP seeds.
B. Reduced-overhead sweep (16-64 of 256 pairs), multi-path, unknown L: no learned method has been compared with compressive NOMP. Run a truth-seeded headroom check first; if the oracle gap is small, drop it.
C. Phase-incoherent sweep (phase constant within a burst, random across bursts, plus residual CFO), possibly combined with B; expect small margins over a matched classical model.
Clustered channels and unknown L: robustness section only (project data show NOMP beats networks on close pairs and blind CFAR beats the CNN count).

## 4. Recommended decoder: proposal network + unfolded NOMP + model-based verification

1. Input (model-based): back-project the measurements through the known codebook onto a 64x64 (u, v) map so one network serves any number of measurements; channels: real, imaginary, magnitude, noise estimate, bit depth, sweep mask.
2. Proposal network (learned): small CNN giving a centre heatmap, a sub-bin offset, a confidence per candidate (CenterNet / DeepFreq style).
3. Unfolded NOMP refinement (model-based, few learned scalars): cyclic Newton steps on the true likelihood (exact cell likelihood for quantised data). Do NOT learn the step: the project's learned step correlated 0.91-0.92 with Newton. Learn only per-layer damping and an effective noise level via a small hypernetwork (HyperLISTA-style).
4. Add / prune / stop: residual test (NOMP-CFAR or GNOMP Rao test) plus a learned confidence score.
5. Output and loss: set of (u, v, gain, confidence); Hungarian-matched set loss + heatmap and offset losses + a smoothed hit-rate term; bias control (Diskin et al. 2023).
6. Training data: L from 0 to 8, separations down to below a beamwidth, path powers up to 25 dB apart, clustered channels, SNR -10..25 dB, 16-256 measurements, 1-3 bits and unquantised, realistic hardware errors only.
7. Mandatory controls: same refinement from non-learned seeds (FFT peaks, EM-VAMP); oracle-selection ceiling; off-prior test banks.

## 5. Verified references (V = text read; A = abstract/metadata only)

NOMP (Mamandipoor, Ramasamy, Madhow, IEEE TSP 64(19), 2016, arXiv 1509.01942, V). NOMP-CFAR (Xu et al., arXiv 2210.10302, V in notes). GNOMP (Zhu et al., IEEE TAES 62, 2026, arXiv 2307.00491, V in notes). MNOMP (Zhu, Han, Blum, Xu, arXiv 1802.01266, V; venue unverified). Enhanced MNOMP (arXiv 2607.17308, A). Marzi, Ramasamy, Madhow (IEEE JSTSP 2016, arXiv 1506.05367, V). Rasekh, Madhow (arXiv 1801.06608, V). Swift-Link (Myers, Mezghani, Heath, IEEE TSP, arXiv 1806.02005, V). Han et al. (IEEE JSAC 38(9), 2020, arXiv 2002.09858, V). Li et al. (IEEE TWC 24(7), 2025, arXiv 2501.11844, V). Sohrabi, Chen, Yu (arXiv 2012.13607, V). Heng, Mo, Andrews (arXiv 2107.13121, A). Alkhateeb et al. (arXiv 1401.7426, A). Chiu, Ronquillo, Javidi (IEEE JSAC, A). Abdallah et al. (IEEE TWC 2021, arXiv 2102.10847, V in notes). 3GPP TR 38.901 (partial, via a mirror). Akdeniz et al. (arXiv 1312.4921, V). Samimi, Rappaport (arXiv 1510.03081, V). Giordani et al. (arXiv 1804.01908, V). Wang et al. (arXiv 1708.07605, A). Cui, Dai (arXiv 2108.07581, A). DETR (arXiv 2005.12872, A). Objects as Points (arXiv 1904.07850, A). cResFreq (IEEE TGRS 60, 2022, A). Mo, Schniter, Heath (IEEE TSP 66(5), 2018, V in notes). Hu, Sun, Zhang (IEEE JSTSP 2025, arXiv 2502.17473, V in notes). Others verified in the project notes: DeepFreq, NeuPLSE, mpNet, ChainNet, Barthelme & Utschick, Chen et al. (Learning to Stop), HyperLISTA, HyperMIMO, Diskin et al., Bialer et al., LoRD-Net, LDAMP, DPIR, Multipath Matching Pursuit (full entries in `deep_research/unfolded_v2/lit/unfolding_survey.md`).

Not established: no paper compares a learned decoder with compressive NOMP or NOMP-CFAR for multi-path AoA/AoD; no paper unfolds NOMP itself; scenarios 2 and 4 have no project measurement, so their gains are open questions.
