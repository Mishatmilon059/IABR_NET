# Positioning: where our low-resolution-ADC angle estimator stands

Prepared 2026-10-16. Sources: the nine study notes in `D:/ai_ml_project/paper/evidence/adc_program/papers/` and the companion `TAKEAWAYS.md` in this folder. Numbers are as recorded in those notes (our numbers from `RESULTS_SUMMARY.md` via the notes). Nothing was run for this document. Status tags follow `TAKEAWAYS.md`: FULL, PARTIAL, ABSTRACT, FIG (plot only), BIB-ONLY.

Problem being positioned: joint AoA/AoD estimation for an analog mmWave MIMO with 16 x 16 ULAs and 16 x 16 DFT codebooks. Beam-domain measurement Y = W^H H F + Z (16 x 16 complex, one snapshot), L = 3 paths, b-bit mid-rise ADC on Re and Im, per-scene peak AGC, SNR -10 to 25 dB. Metric: paper-style Pd (1 degree per angle after Hungarian matching; mean over SNR groups), strict Pd, RMSE.

---

## 1. Comparison matrix

Columns: Work | Task | Data model | Bits | Snapshots | Known noise / L | Method family | Learned? | Reported accuracy (status) | Cost (as reported) | Code.

| Work | Task | Data model | Bits | Snapshots | Known noise / L | Method family | Learned? | Reported accuracy (status) | Cost (as reported) | Code |
|---|---|---|---|---|---|---|---|---|---|---|
| NOMP, Mamandipoor 2016 (FULL) | 1-D line spectrum | unquantised y = sum g x(w) + z | ideal | 1 | sigma for CFAR; L via CFAR/BIC | greedy FFT + Newton + cyclic refinement | no | ours, 2-D: Pd 0.7581 ideal bank; paper: figures only | ours 5.8 ms/scene (1 thread); paper 6.92 s / 300 runs (N=256) | none; ours is an independent copy |
| NOMP-CFAR, Xu 2023 (FULL) | 1-D, unknown K and sigma | unquantised | ideal | 1 (MMV extension) | sigma unknown; K_max = 2K | NOMP + cell-averaging CFAR, prune/add | no | P_D 0.75 at 14 dB (NOMP) vs 15 dB (NOMP-CFAR): about 1 dB cost (paper) | not stated in note | not stated |
| GNOMP, Zhu 2026 (FULL, published) | 1-D few-bit line spectrum | Q_C(sum a x + eps), uniform, open outer cells | 1, 2, 3, inf (sim) | 1 | sigma known (unknown variant not specified); K via Rao + CFAR | exact cell likelihood, Rao test, Newton, cyclic | no | 1-bit scenario loss 4.8/2.3/0.9 dB (one scenario; theory 4.8/2.3/1); ANM better below 17 dB at 1 bit (paper Fig. 14) | 0.11-0.15 s/scene real data (N=256, range only); 0.056 s vs ANM 13.78 s (N=128 sim) | none found (absence UNVERIFIED) |
| GAMP, Rangan 2011/2012 (FULL arXiv v2) | generic sparse estimation | y = Q(A x + w), separable output | any | 1 | prior and output channel known | AMP with scalar denoisers + Onsager term | no | sim: nonlinear model gains >11 dB over linearised | 10-20 iterations; O(mn) per iteration | none (generic) |
| EM-GM-GAMP / EM-VAMP, Mo, Schniter, Heath 2018 (FULL arXiv v4) | broadband mmWave channel (64x64, 16 delay taps) | quantised CS, Zadoff-Chu pilot (FFT operators) | 1-4 (+ inf) | Np = 1024-5120 training symbols | sigma known; L not needed | EM + GAMP/VAMP | EM learns prior only | NMSE only: 1-bit penalty about 2.5 dB at 0 dB and about 10 dB at 15 dB (plot readings) | MATLAB; VAMP about 3 iterations, GAMP about 8, QIHT about 35 | not stated in note (gampmatlab not inspected) |
| BLMMSE, Li et al. 2017 (FULL arXiv v3) | massive MIMO channel (Rayleigh) | y = Q(Phi h + n) | 1 | tau pilots | SNR known; K known | Bussgang gain + LMMSE | no | error floor -4.40 dB (eq. 18); Fig. 2 about -7.1 dB (plot); ours: BLMMSE+NOMP worse than NOMP in all 4 cells | not stated | none |
| Capacity, Mo and Heath 2015 (FULL) | capacity, not estimation | r = sgn(Hx + n), perfect CSI | 1 | n/a | perfect CSI | capacity bounds | no | 1.96 dB low-SNR loss (eq. 10) | n/a | none |
| VALSE-EP, Zhu, Zhang, Meng 2021 (FULL arXiv draft) | 1-D line spectrum | Q on Re/Im; EP pseudo-measurements | 1-3 | 1 | noise learned by EM; K learned by greedy evidence | variational Bayes + EP | learned hyperparameters only | plots only; "near CRB" claimed; no NOMP/ML comparison | O(N K^3) claimed; no timing | MATLAB repo (not read) |
| SBRI-Net / SBRI-X-Net, Hu, Sun, Zhang 2025 (FULL) | 1-D DOA, sparse arrays (18/10 el.), 2-3 sources | csgn (sign of Re, Im), zero threshold | 1 only | 1 | noise unknown (amplitudes in noise units); K <= 3 | MM/IRLS sparse Bayes, unrolled CNN + FC | yes (CNN for inverse; FC for gaps; slope b) | hit rate on-grid 0 dB: 0.70 vs SBRI-X 0.71 (figure); 3 targets, 0 dB, 10 el.: 0.56 vs 0.245 (figure) | FLOP formulas only; no timing | none |
| SBRI-Net precursor, Hu, Sun, Zhang 2024 (FULL text) | 1-D off-grid DOA | sign of Re, Im | 1 | 1 | not stated | unrolled sparse Bayes, 2 blocks | yes | FIG only; baselines OGIR, CNN-DNN | not stated | none |
| TransMUSIC, Ji, Mao, Xi, Chen (FULL text; ICASSP record UNVERIFIED) | 1-D DOA, ULA M = 8 | 1-bit complex sign / sqrt(2) | 1 | L = 200 | source count learned | Transformer + MUSIC + learned peak finder | yes | FIG only; claims beating unquantised MUSIC | not measured | github Transformer_Music (linked) |
| Yeganegi et al. 2024 (FULL text, partly garbled) | 1-D DOA | one-bit, uniform dither | 1 + dither | several (count UNVERIFIED) | dither | LISTA on dithered covariance | yes | FIG only; baseline MUSIC | not stated | github (linked) |
| Fesl, Turan, Bock, Utschick 2023 (FULL text) | channel vector, N = 64 | Q_B on Re/Im, P = 1 pilot | 1-3 | 1 | SNR known | conditional Bussgang LMMSE with latent GMM/MFA/VAE | yes (prior) | FIG only; text: beats BLS, Buss-Scov, EM-GM-GAMP | not measured | not stated |
| Fesl, Banna, Utschick 2024 (ABSTRACT) | channel | one-bit EM | 1 | ? | ? | GMM prior picks component, then one-bit EM | yes | no numbers read | ? | ? |
| FBM-CENet, Nguyen, Nguyen, Swindlehurst 2021 (PARTIAL) | channel and detection, K users | b-bit Re/Im, N0 known | 1-3 | pilot 5K | N0 known | unrolled gradient ascent on sigmoid-ML, learned pilot | yes | qualitative only | not stated | not stated |
| Doshi and Andrews 2022 (FULL text) | one-bit mmWave channel, CDL-A..E | one-bit pilots, phase-shift quantisation | 1 | one pilot block | genie scale for NMSE | WGAN-GP latent optimisation (QGCE) | yes | WGAN-GP beats BG-GAMP by 0.5-5 dB (text); BG-GAMP sparsity genie-tuned | not stated | not stated |
| Park, Kim, Choi 2026 (ABSTRACT) | beamspace denoiser, low-res ADC | composite noise model | low-res | ? | ? | Bayesian binary test, closed-form threshold (non-learned) | no | abstract only | FPGA (abstract) | none stated |
| SignalNet, Dreifuerst and Heath 2022 (FULL arXiv v2) | 1-D sinusoid decomposition, N = 64 | Q_b on Re/Im, unit-norm AGC | 1, 3 | 1 | SNR not defined; m learned | CNN detector + successive CNN estimator | yes | digitised: 3-bit Chamfer worse than classical at -20/-15 dB; frequency threshold (Eq. 45) appears wrong | <1 ms on GPU (claimed, not measured) | none for SignalNet |
| LoRD-Net, Khobahi et al. 2021 (FULL arXiv v1) | BPSK detection, 128 x 16, channel unknown | sign of y - b, real-valued | 1 only | 1 vector per detection; B pilots | C = I known; channel learned from pilots | unfolded relaxed one-bit ML + PSD preconditioners, surrogate H | yes | digitised: nML 1.04e-2 vs LoRD-Net 6.07e-4 at 8 dB (paper); our faithful nML 7.48e-4 | 30 layers vs 700 nML iterations; no timing | none usable (README only) |
| Algorithm unrolling review, Monga et al. 2021 (FULL review) | none (review) | n/a | n/a | n/a | n/a | unrolling recipe | n/a | second-hand numbers only | n/a | n/a |
| Abstract-only group: Sheng 2021; Zhou 2024; Karakoca 2023; He 2020 (ABSTRACT) | DOA / channel | low-bit or one-bit | 1 or low | ? | ? | ADCnet+AtomicNet; diffusion; cGAN; LDGEC | yes | no numbers read | "10x latency" claim UNVERIFIED | none stated |
| **Our previous hybrid (CNN seeds + quantiser ML), project** | 2-D AoA/AoD, 16 x 16 beamspace, L = 3 | Y = W^H H F + Z, b-bit mid-rise, peak AGC | 1, 2 | 1 | sigma known for ML; L = 3 given to all | learned heat-map seeds, then exact cell likelihood with LM and EM | yes (seeds only) | 1-bit 15 dB: 0.572 vs ML 0.540 (+0.032, CI excludes 0; replicated +0.028, +0.036); 2-bit 15 dB: 0.687 vs 0.682 (tie); 0 dB: tie (1-bit 0.274 vs 0.275; 2-bit 0.471 vs 0.476) | ML 1 start 93.8 ms; ML all starts 482.4 ms; CNN 6.2 ms (1 thread). Hybrid latency row not in the notes | ours (adc_program/finetune) |
| **Our CNN read-out alone, project** | same | same | 1, 2 | 1 | sigma-free | CNN, one pass | yes | 1-bit 15 dB 0.419; 2-bit 15 dB 0.593; 1-bit 0 dB 0.239; 2-bit 0 dB 0.438. 78-92% of ML Pd | 6.2 ms (1 thread) | ours |
| **Our unfolded-v2 decoder (previous attempt), project** | same | same | 1, 2 | 1 | known sigma | unfolded NOMP / quantiser LM with learned step, hypernet, fusion, selector, stop head | yes | validation null: front end 0.6827 to 0.6854; S2B 0.6864 to 0.6788; S2A best 0.6851 < 0.6864. S3 and test never run | 98-101 ms vs NOMP 4.8-6.9 ms | ours (not published) |
| **Our planned network (not built)** | same | same | 1, 2 | 1 | known sigma at first; sigma-mismatch arm | LoRD-Net-style ablation: classical LM with learned surrogate sigma_eff and selection score, PSD preconditioners, staged training from classical init | yes (few scalars) | none yet | target not yet set; latency must be measured on 1 thread | ours (to build) |

Reading the matrix. Three facts stand out. (1) Every learned few-bit method above is 1-D, or channel-vector, or NMSE-only, except our own work. (2) Only the Hu 2025 one-bit line and our hybrid compare against a sparse, quantisation-aware baseline on a sparse problem. Hu 2025 compares with its own probit parent (SBRI) and with one-bit sparse methods (SLIM, LIKES, SPICE, OGIR, CNN-DNN), but not with NOMP or with multi-start exact-cell ML. (3) Almost every literature accuracy number is a plot reading; the only text-quoted numbers are the Doshi and Andrews dB gains, the GNOMP loss scenario, the Mo et al. NMSE trends, and the Hu et al. hit-rate readings.

---

## 2. The gap our work can fill (stated cautiously)

**What the surveyed literature does not contain (verified in the notes; the survey covers 9 full notes plus the 12-work sweep).**
- No surveyed work estimates a joint (AoA, AoD) pair from a single 16 x 16 low-resolution beam sweep with peak AGC, L known, at 1 or 2 bits.
- No learned few-bit estimator in the set is compared against NOMP or against exact-likelihood quantiser ML on a sparse problem. The sweep note's conclusion says the same; it checked full texts for most entries and abstracts for the rest.
- The few-bit line-spectrum papers (GNOMP, VALSE-EP, SignalNet) are 1-D. GNOMP compares against ANM and Bayesian methods in 1-D; VALSE-EP compares only against VALSE; SignalNet has no ML or NOMP baseline and states no other quantisation-aware sinusoid estimator exists in its comparisons.
- The few-bit mmWave estimators (Mo et al. 2018, Fesl 2023, Nguyen 2021, Doshi and Andrews 2022) measure NMSE or rate, not angle accuracy, and use fully digital or channel-vector settings.

**Gap statement (what we can defensibly claim to fill).**
A matched-information, same-metric comparison in the 2-D beam domain at 1 and 2 bits, in which the strongest quantisation-aware classical methods are run on the same test banks, and in which a learned component is tested against a training-free seed. Specifically:
1. The quantiser-ML baseline with multi-start LM and EM, the GNOMP-2D extension (CFAR and known-L modes), the EM-VAMP-seeded ML (Mo et al. adapted), and the SBRI-X loop on our 2-D dictionary (1-bit only) are all run under the same cells, metrics and knowledge.
2. Learned seeds (our CNN) are compared with training-free seeds (EM-VAMP) on the same banks. This is the decisive control. Without it, the learned part is not shown to be needed.

**What our verified evidence already supports (narrow).**
- Network seeds followed by an unchanged exact-cell refinement beat multi-start quantiser ML only at 1 bit, 15 dB: 0.572 vs 0.540 (+0.032, CI excludes 0, replicated +0.028 and +0.036). They tie at 1-bit 0 dB, and at 2 bits at both SNRs.
- A sigma-free CNN read-out reaches 78-92% of multi-start ML Pd at about 1/80 of its latency (6.2 ms vs 482.4 ms, 1 thread).
- The learned refinement components did not improve validation over their classical initialisation (unfolded-v2). This is a negative result and must be reported as one.
- The exact cell likelihood is prior art (GAMP framework, Mo et al. 2018, GNOMP). Our novelty can only be the seeding, the cost trade-off and the sigma-free read-out, not the likelihood.

**What we must not claim until the runs exist.**
- That we beat GNOMP (2-D or 1-D), EM-VAMP, SBRI-X, NOMP-CFAR or VALSE-EP.
- That a network is needed for seeding (until EM-VAMP-seeded ML is compared on the same banks).
- Any unknown-L result under quantisation (no such result exists; only unquantised M7 results do).
- Any result at 3-4 bits, any out-of-range SNR result below -10 dB or above 25 dB, or anything about hardware.
- That the hybrid is cheaper than multi-start ML by a stated factor: the notes give "about 1/5" in two places, but the latency table has no hybrid row. Check RESULTS_SUMMARY before quoting.

---

## 3. Mandatory baselines (priority, status, feasibility on 8 logical CPUs and about 0.8 GB free RAM)

Priority definitions. P1: needed before any claim of beating the state of the art. P2: needed if we cite the work as the closest comparator. P3: cite, or report as an ablation only.

| # | Baseline | Priority | Status | Feasibility on our CPU | Note |
|---|---|---|---|---|---|
| 1 | Quantiser ML, multi-start (13 starts + EM), known sigma and L; plus a +-5 dB sigma-mismatch arm | P1 | Run (fresh seeds, 1 bit and 2 bit at 0 and 15 dB) | Have it: 93.8 ms (1 start), 482.4 ms (all starts) per scene, 1 thread | The strongest classical reference; the mismatch arm is needed because ML loses 0.04-0.07 Pd at 5 dB off. |
| 2 | NOMP, 2-D, gamma = 4, R_c = 3 (ideal and quantised input) | P1 | Run (ideal Pd 0.7581; 1-bit 15 dB 0.191) | Have it: 5.8 ms per scene | Report the deviations (no RAC, clipped steps, 4 seed steps). |
| 3 | Consistent NOMP (clip-and-rerun) | P1 | Run (1-bit 15 dB 0.237) | Have it: 76.9 ms per scene | Not GNOMP. Do not label it so. |
| 4 | GNOMP extended to 2-D (CFAR mode P_FA = 0.01, N = 256; and known-L mode, L = 3, reported with strict Pd) | P1 | NOT RUN | High: one greedy pass, FFT Rao test, 2x2 Newton | Unit-test first in 1-D against the paper: theory losses 4.8/2.3/1 dB (Sec. V-B) and the Sec. V-C example, which should stop after two detections near 2.198 and 2.401. Credit as "GNOMP (Zhu et al. 2026) extended by us to 2-D". |
| 5 | EM-VAMP-seeded quantiser ML, Mo et al. 2018 adapted to 16 x 16: V0 (G = 1), V1 (G = 4, 8, damped), V2 (V1 + one quantiser-ML refinement) | P1 | NOT RUN (no EM-VAMP result exists in any note) | Medium-high: a few 16 x 16G matrix products per iteration; damping needed; report divergence counts | Decisive control: EM-VAMP-seeded ML vs CNN-seeded ML on banks 12000-12999 with paired bootstrap. The on-grid V0 ceiling is 17.8% within 1 degree, so report it as faithfulness only. |
| 6 | Learned-method controls: blind floor; size-matched MLP and CNN; label-shuffled network; noise-input network | P1 | Blind floor measured on the SignalNet-style geometry (Pd 0.012-0.021); other three NOT RUN | High for the blind floor (done) and for the small networks; check training memory under about 0.8 GB free RAM | SignalNet argues that a threshold test alone is weak; label-shuffle and noise-input are the proposed stronger tests. |
| 7 | SBRI-X iterative loop (Hu 2025, Algorithms 3-4) on the 2-D (u, v) dictionary, 1 bit only | P2 | NOT RUN | Medium: matrix-free CG with FFT operators; MM bound valid only for a = 1 | Report 2-bit as "not designed for b > 1" rather than as a fair 2-bit baseline. |
| 8 | SignalNet-style successive estimator adapted to 2-D beamspace (our adaptation; no code) | P2 | NOT RUN | Medium: paper budget is 100,000 samples; reduce and chunk to fit RAM | Label as "SignalNet-style (our adaptation)". Threshold-normalised loss needs our prior variance. |
| 9 | Li et al. BLMMSE + NOMP | P2 | Run (negative: below raw NOMP in all 4 cells) | Have it | Report as a negative result; not the strongest linear baseline. |
| 10 | SBRI-X-Net (learned, Hu 2025) | P2 optional | NOT RUN | Low-medium: paper trains 1e6 scenes on GPUs | Only if time allows and only at a reduced budget. |
| 11 | LoRD-Net-style ablation inside our framework: learned surrogate sigma_eff and selection score, PSD preconditioners, staged training, same 13-start search | P3 (ablation, not a baseline) | NOT RUN | High (cheap) | Must include the non-learned "same search, true model" control, since LoRD-Net's own gap to nML is largely a baseline effect. |
| 12 | Park et al. 2026 non-learned beamspace denoiser | P3 | Not read (abstract only) | Unknown until full text is read | Decide after reading the full text. |
| 13 | Atomic-norm (ANM) for 2-D | P3 | NOT RUN | Low: GNOMP reports 13.78 s per scene for ANM at N = 128 (1-D); a 2-D SDP on 16 x 16 would be heavier (our inference, UNVERIFIED) | Cite; run only if time allows. |
| 14 | TransMUSIC | P3 | NOT RUN | Low: needs L = 200 snapshots and unquantised comparators; we have one snapshot | Cite only. |
| 15 | Fesl 2023 (EM-GM-GAMP, BLS) | P3 | NOT RUN | Medium; channel-vector only | Only if we add a channel experiment. |
| 16 | VALSE-EP | P3 | NOT RUN | Low: 1-D method; a 2-D version would be our own derivation | Cite. Do not call our extension "VALSE-EP". |
| 17 | OGIR (Hu 2024's baseline) | P3 | NOT RUN | Not checked | Cite. |

The minimum set needed to claim anything about the state of the art is rows 1-6. Rows 1-3 and 9 already exist. Rows 4-6 are the gap.

---

## 4. Fair-comparison rules

1. **Same data.** Test banks 12000-12999 (1000 scenes per cell), fresh adversarial banks 13000-13999. Training seeds 80,000,000 and up; validation 81,000,000 and up. Tune only on validation seeds.
2. **Same quantiser.** Our per-scene peak AGC and step 2A/2^b (stress_sim.adc). Note that acommon.adc_cells closes the outer cells at +-A, whereas GNOMP and Li keep them open. State which convention each run uses, and run the other if the difference matters.
3. **Same measurement operator.** W and F are the same 16-point DFT codebooks for every method, including the learned ones.
4. **Same knowledge, stated per method.** L = 3 given to every method (state that Mo et al. and GAMP do not need L, and that GNOMP-CFAR estimates it). Noise known to ML and EM-VAMP; report the +-5 dB mismatch arm for each. SBRI-X uses no sigma but has a hand-set (a, b). Report what each method was told.
5. **Same metrics.** Paper-style Pd and strict Pd (missed paths count as failures; needed for any CFAR method that stops early), RMSE in degrees after Hungarian matching. Do not compare our Pd with the papers' MSE, NMSE, Chamfer or hit-rate numbers; their definitions differ.
6. **Same statistics.** Paired bootstrap, 2000 resamples within SNR groups, seed 0. Claim a difference only when the 95% CI excludes zero. Report the replicate seed as well.
7. **Same cost measurement.** Latency on one thread, per scene, on the same CPU (s6_latency.json protocol). Report memory and any divergence counts.
8. **Same tuning budget.** Damping, grid oversampling G, iteration counts, step and kappa are tuned on validation seeds only, under one rule for all methods. Do not tune a baseline less than the proposed method (LoRD-Net's own nML baseline is the warning case).
9. **Include the strongest non-learned method with the same search.** Compare the network against quantiser ML with the same starts and candidates, not only against NOMP.
10. **Report negative results.** The unfolded-v2 null result, the BLMMSE result and the LoRD-Net nML gap all go in the report.
11. **Do not transfer published numbers.** Figure readings from the literature are context only. Our baselines are re-run on our banks.

---

## 5. Conflicts and corrections found while synthesising (must be fixed before reuse)

1. **VALSE-EP DOI.** The sweep note gives 10.23919/JCC.2021.10.005 as if verified. The valse_ep note says the DOI was not seen in any source it opened and is UNVERIFIED. Do not cite that DOI.
2. **VALSE-EP pages 77-95.** From a search snippet only (UNVERIFIED in the valse_ep note).
3. **Consistent NOMP vs GNOMP.** The unfolding survey (row [33]) calls consistent NOMP the GNOMP-like baseline; the gnomp note says that is wrong. Consistent NOMP is a clip-and-rerun loop; GNOMP has never been run here.
4. **Unfolded-v2 files.** bcommon.py, ucore.py, DESIGN.md and criteria.json say "GNOMP-style" or "unfolded from GNOMP". The decoder is unfolded from NOMP and shares only GNOMP's likelihood. The files are outside the notes' folder and were not edited.
5. **GNOMP unknown-noise runtime.** The survey says 3.5-4x. The gnomp note gives 3.55-3.60x in simulation and 5.7-7.5x on real data. Use the note's numbers.
6. **GNOMP source status.** The foundations note could not open the IEEE Xplore page and marks Xplore fields UNVERIFIED. The gnomp note read the published IEEE HTML on 2026-10-16. For GNOMP, the gnomp note supersedes the foundations note.
7. **NOMP-CFAR Xplore fields.** Volume, issue and pages are UNVERIFIED at Xplore; they match only the GNOMP bibliography and the Crossref early-access record.
8. **Two different "Mo" papers.** Mo and Heath 2015 (one-bit capacity, bussgang note) and Mo, Schniter, Heath 2018 (few-bit channel estimation, mo2018 note). Always cite the year and author list in full.
9. **Hybrid latency.** Two notes say the hybrid costs "about 1/5" of multi-start ML. The latency table in the notes has no hybrid row. Verify in RESULTS_SUMMARY before quoting.
10. **SignalNet thresholds.** The paper's frequency thresholds (Eq. 45) appear wrong by a mean-versus-variance mix-up (our Monte Carlo). Do not use the paper's thresholds for any comparison.
11. **Hu 2025 RMSE units.** The text's 2-degree definition conflicts with the plotted RMSE axes. Use only the ordering of RMSE curves. Fig. 6(d) and 6(f) disagree at (a, b) = (1, 1).
12. **LoRD-Net nML.** The paper's nML is about 5x (4 dB) and 14x (8 dB) worse than our faithful nML. The cause is UNVERIFIED. The LoRD-Net "win over nML" should not be cited as evidence that learned surrogates beat their own parent.
13. **TransMUSIC venue.** ICASSP 2024 comes from an index record and is UNVERIFIED on IEEE.

---

## 6. Source files

- Study notes: `D:/ai_ml_project/paper/evidence/adc_program/papers/{foundations,gnomp,mo2018,lordnet,signalnet,onebit_unrolled,valse_ep,bussgang_power,sweep}.md`.
- Our logs cited above: `papers/bussgang_power_check.log`, `papers/gnomp_oracle_pd_our_cells.log`, `papers/snrloss_our_cells.log`, `papers/signalnet_blind_ours.log`, `papers/mo2018_grid_error.log`, `papers/check_hu2025_math.log`.
- Companion document: `D:/ai_ml_project/paper/evidence/adc_program/synthesis/TAKEAWAYS.md`.
