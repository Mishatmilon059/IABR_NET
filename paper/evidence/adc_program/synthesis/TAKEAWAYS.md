# Low-resolution-ADC angle estimation: takeaways for the project

Prepared 2026-10-16 for the project owner. Built only from the nine study notes in `D:/ai_ml_project/paper/evidence/adc_program/papers/` (foundations, gnomp, mo2018, lordnet, signalnet, onebit_unrolled, valse_ep, bussgang_power, sweep). Each paper claim below is as checked in its note. Project numbers are as recorded in `RESULTS_SUMMARY.md` through those notes; nothing was re-run for this document.

Conventions. "Quote" is copied from the study note, which marks it as verbatim with its location; re-check the wording against the PDF before it goes on a slide. Status tags: FULL = full text read; PARTIAL = summary or partial read; ABSTRACT = abstract only; FIG = result only in plots (read by eye, not a table value).

Symbols used below: our problem is a 16x16 beam-domain matrix Y = W^H H F + Z with two angles (AoA, AoD) per path, L = 3 paths, b-bit mid-rise quantiser on Re and Im, per-scene peak AGC. "Pd" is paper-style Pd (1 degree per angle after Hungarian matching) unless stated.

---

## Part 1. One takeaway per paper

### A. Classical decoders (the baselines)

**1. NOMP: Mamandipoor, Ramasamy, Madhow, IEEE TSP 64(19), 2016 (arXiv 1509.01942). Status: FULL.**
- *Problem.* Find the frequencies and complex gains of a few sinusoids in N noisy samples, without a fixed grid.
- *Idea.* FFT grid search for the next tone, Newton refinement off the grid, then re-polish all earlier tones together ("cyclic refinement"). Stop when the residual looks like noise.
- *Result (their 1-D, Gaussian setting).* Faster than AST, Lasso, NLasso and MUSIC in all four scenarios. The one exception is DOMP, which is faster but much less accurate. Accuracy approaches the CRB for well-separated tones.
- *Why it matters to us.* This is our classical decoder, extended to 2-D (16x16 outer-product atoms, 2x2 Newton step). On ideal hardware it is the strongest baseline we have: Pd 0.7581 on the official 8000-scene bank, ahead of the U-Net (0.7366) and ResNet (0.7103) of Lloria et al. Under 1-bit ADCs it collapses (1-bit, 15 dB: Pd 0.191) because its Gaussian likelihood is wrong, not because its search fails.
- *Quote.* "NOMP is faster than all of the other methods in all four Scenarios" (Sec. VI-B, p. 12). Keep the DOMP exception attached to it.
- *Limits.* 1-D only; Gaussian, unquantised noise; the Theorem bound is vacuous at the oversampling used (gamma = 4).

**2. NOMP-CFAR: Xu, Zhu, Fang, Zhang, Xu, IEEE TAES 59(5), 2023 (arXiv 2210.10302). Status: FULL (Xplore volume, issue and pages UNVERIFIED).**
- *Problem.* NOMP's stop rule needs the noise level, which drifts in real radars. The number of targets is unknown.
- *Idea.* Estimate the noise from reference cells around each peak (cell-averaging CFAR). Start with a generous candidate set, then prune or add one candidate at a time.
- *Result.* Keeps the false-alarm rate near nominal when the noise level varies. Costs about 1 dB of SNR. The 1 dB is quoted at P_D 0.8, 0.76 and 0.75 in three places, and the initialisation-gain text does not match its own numbers (2 dB, not 1).
- *Why it matters to us.* It is the reference blind-order method. We have not run it; our blind CFAR and MDL are simpler relatives. Our unquantised M7 check (L ~ U{1..8}) gives P(L_hat = L) = 0.890 at 15 dB and 0.491 at 0 dB for oracle-sigma CFAR, and 0.933 / 0.457 for blind CFAR. Nothing has been run under quantisation.
- *Quote.* "the noise variance is unknown and can often be variable" (Sec. III-A, PDF p. 5).
- *Limits.* Its Gaussian reference-cell assumption fails at 1 bit, where the noise level is not identifiable.

**3. GNOMP: Zhu, Zhang, Zhang, Fang, Qu, IEEE TAES 62, pp. 3933-3951, 2026 (arXiv 2307.00491; published IEEE text read 2026-10-16). Status: FULL (published); arXiv v4 read as PDF.**
- *Problem.* Line spectra from few-bit samples (uniform B-bit quantiser, open outer cells). Estimate the number, frequencies and amplitudes.
- *Idea.* Use the exact cell likelihood (probit cell probabilities), which is the same objective as our quantiser ML. Detect with a Rao (score) test computed by one FFT on pseudo-measurements, with a fixed CFAR threshold (our check: tau = 20.29 for N = 256, P_FA = 0.01). Refine with Newton steps and cyclic re-polishing.
- *Result.* Their theory gives a quantisation SNR loss of 4.8 / 2.3 / 1 dB at 1 / 2 / 3 bits for one scenario (N = 1024, a 6 dB interferer at pi/2). Our reproduction matches the theory within 0.05 dB. The measured losses match within 0.1 dB, except one 1-bit case (5.35 vs 4.8 dB). Under our peak AGC, the median weak-path loss is about 2 dB at 1-3 bits. GNOMP is faster than the ANM method they compare with (0.056 s vs 13.78 s per scene, N = 128, 1-D simulation). The unknown-noise variant costs 3.6x in simulation and 5.7-7.5x on real data, but its algorithm is not described.
- *Why it matters to us.* It is the baseline a reviewer will ask for, and it has not been run. Our likelihood is the same as GNOMP's; our search is heavier (LM from 13 starts plus EM). Consistent NOMP is not GNOMP. Some of our unfolded-v2 files call the design "GNOMP-style"; that is wrong and should be corrected.
- *Quote.* "the noise variance must be estimated in real time from the data" (Sec. IV, after Algorithm 1). The paper's exception is 1 bit.
- *Limits.* 1-D only. The 2-D extension is ours and must be credited as such. The 4.8/2.3/0.9 dB numbers belong to one scenario, not to a per-bit rule.

**4. GAMP: Rangan, ISIT 2011; full version arXiv 1010.5141 v2. Status: FULL (arXiv v2); ISIT pages and city UNVERIFIED.**
- *Problem.* Estimate a sparse x from y = Q(A x + w) for any separable output channel.
- *Idea.* Alternate a linear step (multiply by A and A^T, with an Onsager correction) and two scalar denoisers, one for the prior and one for the output channel. State evolution predicts the error for large i.i.d. Gaussian A.
- *Result.* In one simulation (Bernoulli-Gaussian prior, 500 x 1000, sigmoid output), modelling the nonlinearity gains more than 11 dB over a linearised model. State evolution predicts the simulated error within 0.2 dB (linearised) and 0.4 dB (nonlinear).
- *Why it matters to us.* It is the engine of the Mo-Schniter-Heath few-bit baseline. Its output step for our 1-bit cells equals the cell-likelihood score (checked to 3e-10). Guarantees hold only for i.i.d. Gaussian A, and our DFT beamspace is structured.
- *Quote.* "virtually no difference" (Sec. VII, p. 12), about Algorithms 1 and 2 in simulations the paper does not report.
- *Limits.* One small example, no quantiser, no confidence intervals.

**5. Algorithm unrolling: Monga, Li, Eldar, IEEE SPM 38(2), 2021 (arXiv 1912.10557). Status: FULL (review; no experiments of its own).**
- *Problem.* None; this is a review of the "unfold an iterative algorithm into a trained network" recipe.
- *Idea.* Each iteration becomes a layer. Step sizes, thresholds or matrices become trainable. Layer-specific or shared weights are both possible. A "drop-in" variant replaces one step with a network.
- *Result.* The review reports LISTA is about 20x faster than accelerated ISTA. Its text and Table II disagree in three places (for example CORONA "more than 50x" vs 3.0x in the table). Those numbers are second-hand.
- *Why it matters to us.* It defines "unfolded from NOMP". Our null validation result is typical: learned parts mostly buy speed, not accuracy. Our best ADC system (a CNN seed followed by an unchanged quantiser ML) is the drop-in category, not unrolling.
- *Quote.* "does not exhibit superior performance over ISTA, not even under particular scenarios" (LISTA box, p. 21).

### B. Few-bit classical and Bayesian estimators

**6. Mo, Schniter, Heath, IEEE TSP 66(5), 2018 (arXiv 1610.02735). Status: FULL (arXiv v4, 14 pages); published IEEE text UNVERIFIED.**
- *Problem.* Broadband mmWave channel estimation with few-bit ADCs in a fully digital 64x64 array with 16 delay taps and 1024-5120 training symbols.
- *Idea.* Noisy quantised compressed sensing with the exact cell likelihood. Inference by EM-BG/GM-GAMP or EM-VAMP. Zadoff-Chu pilots make every operator an FFT, so A never has to be stored.
- *Result.* NMSE only, over 100 channel realisations. Plot readings (about +-0.5 dB): the 1-bit penalty is about 2.5 dB at 0 dB SNR and about 10 dB at 15 dB. EM-VAMP converges in about 3 iterations, GAMP in about 8, QIHT in about 35.
- *Why it matters to us.* It is a standard few-bit baseline with exact likelihood, and it has not been run here. Its likelihood is prior art. Our claims must be about seeding, speed or robustness. Our structure check: on the 16x16 DFT grid, A is unitary with one nonzero per column, so on-grid EM-VAMP splits into 256 scalar problems. Even with no noise, snapping to the 16-point grid puts only 17.8% of angle components within 1 degree (68.8% at 4x oversampling, 85.5% at 8x).
- *Quote.* "perform comparably to infinite-bit ADCs at low SNR" (Abstract, p. 1). This is a claim about NMSE, not angle accuracy.
- *Limits.* No angle or off-grid metric. Algorithm 2 line 18 prints the ratio inverted (we use Nx/Ny, checked numerically). Noise variance is assumed known.

**7. Li, Tao, Seco-Granados, Mezghani, Swindlehurst, Liu, IEEE TSP 65(15), 2017 (arXiv 1609.07427). Status: FULL (arXiv v3).**
- *Problem.* One-bit channel estimation for a massive MIMO uplink (Rayleigh channel).
- *Idea.* Bussgang decomposition: the one-bit quantiser is a gain sqrt(2/pi) plus a distortion uncorrelated with the input. Apply LMMSE to the linearised output (BLMMSE).
- *Result.* Error floor 1 - 2/pi = -4.40 dB as SNR grows (their eq. 18, tau = K). Their Fig. 2 curve is about -7.1 dB at 10-20 dB (plot reading). Our reproduction gives about -7.9 dB. The pilot columns they used are not stated, and that choice moves the floor by up to about 4 dB.
- *Why it matters to us.* Our test (sparse beamspace, L = 3, Pd metric): BLMMSE followed by NOMP is worse than raw NOMP in all four cells. At 1 bit, 0 dB: raw NOMP 0.161 vs empirical LMMSE 0.053 (difference -0.108, CI [-0.119, -0.098]). The Gaussian assumption fails for our beamspace (per-beam gain ratio 0.38 vs 1.00 for a Gaussian surrogate). Do not feed a Bussgang-linearised estimate to the network.
- *Quote.* "One-bit ADCs do not require automatic gain control and linear amplifiers." (Sec. I, p. 1). Our model does use peak AGC, so this does not carry over.
- *Limits.* The distortion is uncorrelated with the signal but not independent of it. Its power is 0.743 when |y| < 0.3 and 0.506 when |y| > 1.5 (our check).

**8. Walden, "Analog-to-digital converter survey and analysis," IEEE JSAC 17(4), 1999. Status: FULL (survey).**
- *Problem.* Power cost of converters versus resolution and sampling rate.
- *Idea.* A figure of merit F = 2^SNRbits f_s / P (eq. 4). Survey data give a mean F of about 7.9e10.
- *Result.* Resolution falls by about 1 bit for every doubling of the sampling rate. The "power grows exponentially with bits" claim in Li and Mo is only a comparator count (2^N - 1) at fixed F and f_s; Walden does not state a power law.
- *Why it matters to us.* Motivation only. We do not measure ADC power. Its data period is UNVERIFIED.
- *Quote.* "resolution falls off by 1 bit for every doubling of the sampling rate" (Abstract, p. 539).

**9. Mo and Heath, IEEE TSP 63(20), 2015 (arXiv 1410.7353). Status: FULL. Not an estimator. Do not confuse with item 6.**
- *Problem.* Capacity of one-bit MIMO with transmitter CSI.
- *Idea.* Capacity bounds for one-bit outputs, including mmWave with L paths: log2 K(Nr, L) is an upper bound on the capacity.
- *Result.* One-bit low-SNR power loss of 10 log10(pi/2) = 1.96 dB (their eq. 10). The mmWave rates converge to log2 K(4, L) only above 60 dB.
- *Why it matters to us.* Background for the 1.96 dB minimum loss. It is a capacity result with perfect CSI and says nothing about angle recovery.
- *Quote.* "the power consumption grows exponentially with the resolution" (Sec. I, p. 1), citing Walden.

### C. Learned and unrolled one-bit or few-bit estimators

**10. LoRD-Net: Khobahi, Shlezinger, Soltanalian, Eldar, IEEE TSP 69, 2021 (arXiv 2102.02993, v1 only). Status: FULL (arXiv v1); journal text UNVERIFIED.**
- *Problem.* Detect BPSK symbols from one-bit observations of an unknown, pilot-learned channel (128 x 16 in the main test).
- *Idea.* Unfold the relaxed one-bit ML gradient over 30 layers with positive semi-definite diagonal preconditioners. Stage 1 learns a surrogate channel H (not the true one), with fixed steps. Stage 2 learns the steps with H frozen. About 2,500 trainable parameters, against more than 8 x 10^5 for DeepSIC.
- *Result (figures, digitised).* Against nML (given the true channel), LoRD-Net is much better at 8 dB: 6.07e-4 vs 1.04e-2 for the paper's nML. Our faithful nML gives 7.48e-4 at 8 dB, so the paper's nML baseline looks weak (cause UNVERIFIED). In our re-implementation, plain gradient ascent with the true H beats LoRD-Net's Stage-2 network on all 12 of 12 channels. A learned step on the true model explains about 10x of the reported gap to nML.
- *Why it matters to us.* Its useful ideas are a learned objective (surrogate noise or selection score), staged training, and a PSD preconditioner. It is not a baseline as published: one-bit, BPSK, one fixed channel per training set, and no usable code (the linked repo has only a README and licence).
- *Quote.* "the alternative channel learned by LoRD-Net is in general not the true channel." (Sec. I, p. 2).

**11. SignalNet: Dreifuerst and Heath, IEEE TSP 70, 2022 (arXiv 2106.05490 v2). Status: FULL (arXiv v2); published text UNVERIFIED; code not released.**
- *Problem.* Count and estimate the 1-D sinusoids (amplitude, frequency, phase) from 1- or 3-bit samples, N = 64.
- *Idea.* A CNN detector (softmax over m = 1..5) plus a successive estimator: each block estimates one tone, rebuilds it, subtracts it, and passes the residual on. Loss normalised by a "learning threshold" (the error of the best constant predictor).
- *Result.* Digitised (medium confidence): beats FFT/periodogram and AIC/MDL at 3 bits, but at -20 and -15 dB the 3-bit Chamfer result is worse than the classical baselines. At 1 bit the detector never beats the threshold. The paper's frequency thresholds (Eq. 45) look wrong: our Monte Carlo gives -22.8 to -20.0 dB, not -18.1 to -15.7 dB. With those values the "universal" improvement claim fails at low SNR.
- *Why it matters to us.* The learning-threshold idea is a sanity check, and our blind floor (Pd 0.012-0.021 on our geometry) is far below our results (8-30x). That passes easily and proves little. The paper's own remedy is label-shuffle and noise-input controls, which we have not run.
- *Quote.* "our algorithm is always able to surpass the threshold for three-bit data." (Abstract, p. 1).
- *Limits.* Its 1-bit amplitude curves are unexplained (digitised, UNVERIFIED). The detection threshold of 1.67 cannot be reached by any integer output (best integer 1.74).

**12. Hu, Sun, Zhang, "Model-based learning for DOA estimation with one-bit single-snapshot sparse arrays," IEEE JSTSP 19(6), 2025 (arXiv 2502.17473). Status: FULL (published text plus figures).**
- *Problem.* 1-D DOA from one-bit single-snapshot data on sparse linear arrays (18 and 10 elements), 2-3 sources.
- *Idea.* SBRI: sparse-Bayes MAP with a probit likelihood, solved by MM/IRLS, with a Taylor gap for off-grid targets. SBRI-X: a logistic likelihood with shape parameters (a, b). SBRI-Net and SBRI-X-Net unroll these loops, replacing the matrix inverse with small 1-D CNNs and the gap update with FC layers. The slope b is learned; a is fixed at 1.
- *Result (figure readings, no table).* On-grid, 18 elements, 0 dB: SBRI-X-Net 0.70 vs SBRI-X 0.71; SBRI-Net 0.56 vs SBRI 0.30. Off-grid, 3 targets, 0 dB, 10 elements: 0.56 and 0.57 vs 0.245 and 0.065 for the parents. Tests go upward only (35-50 dB). Nothing is tested below 0 dB.
- *Why it matters to us.* It is the closest one-bit learned comparator and should be implemented on our 2-D dictionary. Its probit likelihood is our 1-bit cell likelihood. Its learned b = 0.19 is confounded with the amplitude scale (our check L), so it is not a noise estimate. Its MM majoriser holds only for a = 1, and its RIP assumption fails for its own dictionaries.
- *Quote.* "SBRI-X-Net achieves comparable RMSE to SBRI-X at SNRs from 0 dB to 15 dB" (Sec. V-A, p. 921).
- *Limits.* The RMSE axes conflict with the paper's own 2-degree threshold (use the ordering only). Fig. 6(d) and 6(f) disagree at (a, b) = (1, 1). No code.

**13. VALSE-EP: Zhu, Zhang, Meng, China Communications 18(10), 2021 (arXiv 1811.05680 v3 draft). Status: FULL (arXiv draft); journal pages and DOI UNVERIFIED.**
- *Problem.* Gridless line-spectral estimation from quantised samples, 1-D.
- *Idea.* Expectation propagation turns each quantiser cell into a Gaussian pseudo-measurement with its own variance. VALSE (variational Bayes) runs on that pseudo-model. Noise variance is learned by an approximate EM rule. Model order comes from a greedy evidence search.
- *Result.* Plots only (no table, no NMSE number). Claimed "near CRB". Comparators are only VALSE with an AQNM input and unquantised VALSE. No NOMP, ML, atomic-norm or grid method is compared.
- *Why it matters to us.* Not usable as a baseline: 1-D, and the beam-domain Y is not a Vandermonde sum (checked for one 16-point DFT codebook). The idea of per-measurement pseudo-noise and learned noise is a design input. The DOI that the sweep note gives (10.23919/JCC.2021.10.005) is not confirmed by any source the valse_ep note opened; do not cite it.
- *Quote.* "Since diagonal EP performs better than scalar EP" (Sec. IV, p. 9).

**14. Hu, Sun, Zhang, off-grid one-bit DOA precursor, Asilomar 2024 (arXiv 2412.10976). Status: FULL (text); results FIG.**
- *Problem.* 1-D off-grid one-bit DOA, sparse linear arrays (18 and 10 elements), 2 targets.
- *Idea.* Unrolled sparse-Bayes with two blocks: fixed-grid refinement, then amplitude and gap updates. Training is two stages (MSE plus BCE in stage 2).
- *Result.* FIG only. Baselines OGIR and CNN-DNN.
- *Why it matters to us.* A precursor to item 12. Weaker baselines; cite it, do not treat its numbers as ours.
- *Quote.* "the proposed approach outperforms state-of-the-art methods across a range of SNR scenarios" (Conclusion).

**15. TransMUSIC: Ji, Mao, Xi, Chen, arXiv 2309.08174 v2 (ICASSP 2024 record UNVERIFIED). Status: FULL (text); results FIG.**
- *Problem.* 1-D DOA with one-bit complex sign, ULA with M = 8, L = 200 snapshots.
- *Idea.* A Transformer denoises snapshot tokens, a MUSIC-type spectrum is computed from the estimated noise subspace, a learned peak finder gives gridless DOA, and a small network estimates the source count.
- *Result.* FIG only. The paper claims one-bit TransMUSIC beats unquantised MUSIC and other unquantised methods.
- *Why it matters to us.* Its comparators are unquantised. It needs many snapshots, and our data has one. It cannot be run as published on our problem. Cite only.
- *Quote.* "surpasses the classic MUSIC algorithm and other counterparts that rely on unquantized data" (Conclusion).

**16. Yeganegi, Eamaz, Esmaeilbeig, Soltanalian, arXiv 2405.09712 (SAM 2024 per index). Status: FULL (text, partly garbled); results FIG.**
- *Problem.* 1-D DOA from one-bit data.
- *Idea.* Uniform dither on the complex samples, then a sample covariance of the dithered data, then LISTA recovers the sparse source vector.
- *Result.* FIG only, against MUSIC.
- *Why it matters to us.* Shows that LISTA can converge on dithered one-bit covariance data. Our pipeline has no dither, so that route needs a hardware change. Parameter values in the cached text are garbled and UNVERIFIED.
- *Quote.* "LISTA converges even under coarse quantization of high-resolution measurements" (Fig. 2 caption, per the sweep note).

**17. Fesl, Turan, Böck, Utschick, arXiv 2309.04014 (venue UNVERIFIED). Status: FULL (text); results FIG.**
- *Problem.* Channel vector (not DOA), N = 64 antennas, P = 1 pilot, B in {1, 2, 3} bits.
- *Idea.* A conditionally Gaussian latent prior (GMM, MFA or VAE) learned from quantised pilots only. Given the latent component, the Bussgang decomposition applies, and a linear MMSE estimator is averaged over components.
- *Result.* FIG only. The text says the proposed estimator beats BLS, Buss-Scov and EM-GM-GAMP over the whole SNR range.
- *Why it matters to us.* A channel-vector baseline. It becomes relevant only if we add a channel experiment. Its learned-prior-from-quantised-data idea is a candidate design for a seed.
- *Quote.* "are outperformed with a considerable performance gap over the whole SNR range" (Sec. VI).

**18. Nguyen, Nguyen, Swindlehurst, FBM-CENet, arXiv 2107.11958 (venue UNVERIFIED). Status: PARTIAL (WebFetch summary, no verbatim quotes).**
- *Problem.* Few-bit massive MIMO channel estimation and data detection, K users, b in {1, 2, 3}, pilot length 5K.
- *Idea.* Unroll gradient ascent on a sigmoid-reformulated ML objective, with per-layer step sizes and a learned pilot (the pilot is the first-layer weight).
- *Result.* Qualitative in the text: slightly beats a one-bit SVM at medium to high SNR at one bit, and beats the others clearly at few bits. No dB values.
- *Why it matters to us.* The closest conceptual cousin of our hybrid: a learned start followed by quantiser-aware refinement. Read the full text before citing any detail.

**19. Doshi and Andrews, arXiv 2211.08635 (journal UNVERIFIED). Status: FULL (text).**
- *Problem.* One-bit mmWave channel estimation from pilots, CDL-A to CDL-E channels.
- *Idea.* A WGAN-GP generator learned from channel data. The channel is estimated by optimising the generator's latent input against one-bit pilot measurements.
- *Result.* The individually trained WGAN-GP beats BG-GAMP by 0.5 dB (CDL-B, C), 1 dB (CDL-A) and 5 dB (CDL-D, E). BG-GAMP's sparsity rate was tuned with genie knowledge.
- *Why it matters to us.* A caution about tuned baselines. Our NOMP and quantiser ML must be tuned under the same rule, on validation seeds only.
- *Quote.* "by a careful tuning of the sparsity rate" (Sec. V-C1, on BG-GAMP).

**20. Park, Kim, Choi, arXiv 2605.08855 (May 2026). Status: ABSTRACT.**
- *Problem.* Beamspace channel denoising for mmWave with low-resolution ADCs.
- *Idea.* A non-learned Bayesian binary hypothesis test under a Bernoulli-complex Gaussian prior, with a closed-form threshold and an FPGA implementation (per abstract).
- *Why it matters to us.* A non-learned beamspace comparator, possibly closer to our setting than the channel papers. Read the full text before deciding whether it is a baseline. No numbers.

**21. Other abstract-only or bibliography-only items. Status: ABSTRACT or BIB-ONLY.**
- *Fesl, Banna, Utschick, arXiv 2405.03542.* A GMM prior picks a mixture component, which feeds a one-bit EM estimator. No numbers read.
- *Sheng et al., Electronics 10(6):738, 2021.* ADCnet plus AtomicNet for DOA with low-bit ADCs. MDPI returned 403, so the body is unread.
- *Zhou et al., arXiv 2408.10501; Karakoca et al., arXiv 2306.08882; He et al., arXiv 2006.16628.* Diffusion-based, cGAN-based and unsupervised beamspace channel estimators with quantised data. Abstract only; latency claims UNVERIFIED.
- *Not found or not read:* Azhdari and Farhang (title not found), Zhan, Xi, Chen (ICICSP 2022, not found), Meng and Zhu generalised SBL for one-bit DOA (ref. 33 of VALSE-EP, not read). Bibliography-only records: Rahman et al., Sun et al., Li and Gao et al., Li et al. (Sensors 2022), Katyal et al., Li et al. (OJCOMS 2021), Xu et al. (TWC 2022).

---

## Part 2. The big picture of low-resolution-ADC estimation (10 bullets)

1. **Peak AGC changes the problem.** At 1 bit only signs survive, so the amplitude scale is lost and decoders are scale-free. At b = 2-3 our peak AGC makes weak paths behave almost like 1-bit (median loss about 2 dB, the pi/2 minimum, at every bit depth).
2. **The exact cell likelihood is prior art.** The probit or interval-censored Gaussian likelihood appears in GAMP-based few-bit estimation (Mo et al. 2018), in GNOMP and in our quantiser ML. Novelty has to be in seeding, speed, robustness or selection, not in the likelihood.
3. **Linearisation fails for sparse beamspace.** Bussgang BLMMSE followed by NOMP is worse than raw NOMP in every cell we ran. The Gaussian assumption behind the arcsine law does not hold for our beams.
4. **Quantisation-aware refinement is the main classical gain.** NOMP (Gaussian) at 1-bit, 15 dB: Pd 0.191. Quantiser ML (13 starts + EM): 0.540. NOMP on ideal hardware: 0.758.
5. **At 15 dB detection is not the bottleneck.** The oracle bound says all three paths are detectable in about 92% of scenes. Yet ML started from the true angles reaches only 0.626, so the headroom is in estimation within 1 degree and in selection.
6. **At 0 dB the third path is information-limited.** Its effective SNR is below the rough 12 dB detectability level in 58-62% of scenes. Its oracle detection probability is 0.39-0.42, so the low-SNR ceilings (0.439 for ML from true angles at 1 bit) are expected.
7. **The second bit helps in the estimation regime.** Multi-start ML improves 0.540 to 0.682 at 15 dB (1 to 2 bits), while the detection theory predicts almost no gain. The theory does not cover strong-signal estimation.
8. **SNR mismatch matters, and at 1 bit the noise level is not identifiable.** ML loses 0.04-0.07 Pd when its SNR is 5 dB off. GNOMP, NOMP-CFAR and VALSE-EP each learn or estimate a noise level in their own way, and each reports sensitivity or a 1-bit caveat.
9. **Learned parts gain where classical methods are weak, not in refinement.** Our CNN seeds help only at 1 bit, 15 dB (+0.032, CI excludes 0). Our unfolded refinement did not beat its initialisation on validation. LoRD-Net's gap to nML is largely a baseline effect. Unrolling mostly buys speed (Monga review).
10. **Most learned work is 1-D, one-bit and under-compared.** The learned DOA papers (SBRI-Net, TransMUSIC, Hu 2024, Yeganegi) are 1-D. Their comparators are MUSIC on unquantised data, OGIR, CNN-DNN, or unquantised MUSIC. Few results are tables; most are figure readings. Only the Hu 2025 SBRI-X line is a quantisation-aware, one-bit single-snapshot comparator.

---

## Part 3. Glossary (plain English)

- **AGC (automatic gain control).** Scales the signal before quantising. Ours is per scene, from the peak of max(|Re|, |Im|) over the 256 beam samples.
- **AQNM.** Additive quantisation noise model: treats the quantiser error as independent Gaussian noise. Used by VALSE-EP only for initialisation and as a baseline input.
- **Bussgang decomposition.** A nonlinear quantiser equals a linear gain plus a distortion that is uncorrelated with the input (but not independent of it).
- **BLMMSE.** LMMSE applied to the Bussgang-linearised output (Li et al. 2017).
- **CFAR (constant false-alarm rate).** A threshold that keeps the false-alarm rate fixed whatever the noise level. NOMP-CFAR uses cell averaging; GNOMP uses a Rao-test threshold.
- **Cyclic refinement.** Re-polishing every earlier tone against the data minus all the other tones. NOMP's feedback step.
- **Cell (quantiser).** The interval of unquantised values that map to one output level. The likelihood is the probability that the noisy value lies in the observed cell.
- **Consistent NOMP.** Our clip-and-rerun loop (projection of the model into the known cells, then NOMP). It is not GNOMP.
- **EM-VAMP / EM-GM-GAMP.** Iterative Bayesian estimators whose sparse prior is learned by EM. Mo et al. 2018.
- **Hungarian matching.** Pairs estimated angles with true angles to minimise total error before scoring.
- **Integrated SNR.** Total path energy over noise, with N samples integrated: about 19.3 dB above our per-sample SNR for an average path, about 13 dB for the weakest of three (median).
- **Learning threshold (SignalNet).** The error of the best constant predictor. A learned estimator should beat it if it used the input.
- **LISTA.** Learned iterative shrinkage-thresholding; ISTA with learned matrices and thresholds.
- **nML.** Near-maximum-likelihood one-bit detector of Choi, Mo, Heath (2016): projected gradient on the relaxed likelihood.
- **NOMP.** Newtonized orthogonal matching pursuit: greedy FFT detection plus Newton refinement (Mamandipoor et al. 2016).
- **Pd (paper-style).** Fraction of scenes where all angle components are within 1 degree after Hungarian matching. Strict Pd counts missed paths as failures.
- **Probit / logistic likelihood.** Probability of a one-bit output as a Gaussian CDF (probit) or a sigmoid (logistic).
- **Rao (score) test.** Tests whether one more sinusoid is present using only the slope of the likelihood at "no sinusoid", so the new amplitude never has to be fitted.
- **SBRI / SBRI-X.** Sparse-Bayes one-bit DOA loops (probit; logistic with shape a, b). SBRI-Net and SBRI-X-Net are the unrolled versions.
- **SNR loss (GNOMP).** How many dB more SNR a B-bit receiver needs to match an ideal one at the same Pd and false-alarm rate.
- **Snapshot.** One time sample of the received vector. Single-snapshot means one per problem.
- **State evolution.** A scalar recursion that predicts the per-iteration error of an AMP-type algorithm, exact only for large i.i.d. Gaussian matrices.
- **Unfolding / unrolling.** Turning T iterations of an algorithm into T network layers with some parameters learned.
- **VALSE-EP.** Gridless variational Bayes with expectation propagation for quantised line spectra (Zhu et al. 2021).
