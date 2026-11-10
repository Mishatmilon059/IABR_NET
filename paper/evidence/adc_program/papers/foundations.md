# Foundations: NOMP, NOMP-CFAR, GAMP and algorithm unrolling. Study guide for our ADC angle-estimation pipeline

**What this note covers.** These four papers supply the building blocks of our pipeline:

| paper | role in our project |
|---|---|
| (a) NOMP, Mamandipoor et al. 2016 | our classical decoder; the strongest ideal-hardware baseline |
| (b) NOMP-CFAR, Xu et al. 2023 | how to stop when the number of paths L is unknown and the noise level is unknown |
| (c) GAMP, Rangan 2011 | the inference engine of the Mo-Schniter-Heath few-bit baseline (see `mo2018.md` in this folder) |
| (d) Algorithm unrolling, Monga et al. 2021 | the design principle behind our learned decoders ("unfolded from NOMP") |

Each paper gets the same 10 sections. Paper (a) is treated at full length; (b)-(d) are shorter.

## Verified bibliographic entries

For every paper, the authors, title, venue, DOI, Crossref pages and arXiv version history were checked against the cached Crossref records (`foundations/cr_*.json`) and the arXiv abstract pages (`foundations/*/abs.html`). An IEEE Xplore fetch returned an empty page on the resume pass (2026-10-16), so every Xplore-only field below is UNVERIFIED at Xplore.

| key | entry |
|---|---|
| (a) | B. Mamandipoor, D. Ramasamy, U. Madhow, "Newtonized Orthogonal Matching Pursuit: Frequency Estimation Over the Continuum," *IEEE Trans. Signal Process.*, vol. 64, no. 19, pp. 5066-5081, 1 Oct. 2016. DOI 10.1109/TSP.2016.2580523 (Xplore document number not checked). arXiv:1509.01942 (v1 7 Sep 2015; v2 28 Apr 2016, read here). |
| (b) | M. Xu, J. Zhu, J. Fang, N. Zhang, Z. Xu, "CFAR Based NOMP for Line Spectral Estimation and Detection," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 59, no. 5, pp. 6971-6990, 2023 (volume, issue and pages UNVERIFIED at Xplore; they match the bibliography of the GNOMP arXiv paper, `gnomp_html.html` bib.bib35, and Crossref gives only pages 1-19). Month and first-online date UNVERIFIED. DOI 10.1109/TAES.2023.3282610. arXiv:2210.10302 (v1 19 Oct 2022, the only version, read here). |
| (c) | S. Rangan, "Generalized approximate message passing for estimation with random linear mixing," in *Proc. IEEE Int. Symp. Inf. Theory (ISIT)*, Jul. 2011, pp. 2168-2172 (Crossref). DOI 10.1109/ISIT.2011.6033942. The city (St. Petersburg) is UNVERIFIED. arXiv:1010.5141 (v1 25 Oct 2010; v2 13 Aug 2012, the 22-page full version, read here). |
| (d) | V. Monga, Y. Li, Y. C. Eldar, "Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing," *IEEE Signal Process. Mag.*, vol. 38, no. 2, pp. 18-44, Mar. 2021. DOI 10.1109/MSP.2020.3016905 (Crossref; Xplore number not checked). arXiv:1912.10557 (v3 7 Aug 2020, read here). |

Notes on the bibliographic data:
- **(b), Crossref is stale.** Crossref still lists only the early-access record (pages "1-19", no volume or issue). The final volume, issue and pages come from IEEE Xplore. The GNOMP arXiv paper (same group; cached as `C:/Users/Milon/iabr_deep_research_cache/gnomp_html.html`, ref. bib.bib35) cites the same volume and pages.
- **(c), wrong pages in the self-citation.** arXiv v2 cites its own conference version as pp. 2174-2178 (its ref. [43]). That is wrong: Xplore and Crossref give pp. 2168-2172.
- **(c), conference versus full version.** The ISIT paper itself (5 pages) is paywalled, and I did not read its body. Its Xplore abstract covers the same G-AMP algorithm and state-evolution result as arXiv v2. arXiv v2 says it adds all the proofs, derivations and more simulations. Any statement below is located in arXiv v2. Whether a statement also appears in the 5-page ISIT paper is UNVERIFIED.
- **Published versus arXiv text.** For (a), (b) and (d), only the arXiv text was read (the IEEE versions are paywalled). The Xplore abstracts were not re-checked on the resume pass. Edits made after the arXiv version are UNVERIFIED.

## How the papers were read and checked

- All four papers were read from the cached full texts (extracted by the interrupted attempt). On the resume pass, each numbered claim was re-checked against those texts by grep and targeted reading; passages marked 'not re-checked' were not re-read. The cached files are under `C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/foundations/{nomp,cfar,gamp,unroll}/paper.pdf|paper.txt|abs.html`. Crossref records are in `cr_*.json` in the same folder.
- **Page numbers.**
  - (a), (c) and (d) are cited by the page number printed on the arXiv PDF.
  - (b) has an unnumbered title page, so it is cited by **PDF page**. The printed number is the PDF page minus 1.
- **Our own checks.**
  - Script: `D:/ai_ml_project/paper/evidence/adc_program/papers/foundations_checks.py`. Its output is in `foundations_checks.log` (numpy/scipy, under 1 minute on CPU).
  - It recomputes the papers' printed thresholds and constants, and applies their formulas to our setting.
  - In the text, "log A1", "log B3" and so on refer to lines of that log.
  - Project numbers come only from the verified reports. These are `deep_research/RESULTS_SUMMARY.md`, `stress/RESULTS.md`, `stress/adversarial/VERDICT.md`, `stress/check_sim.log`, `unfolded_v2/build/deviations.txt`, the `unfolded_v2/build/train_*.log` files and `unfolded_v2/diag/REPORT.md`.
- **Quoting.** A copyright limit allows at most one short verbatim quote. The single quote is claim (a)-4, under 15 words. Every other "citable claim" is a close paraphrase with its exact location (section, page, equation or figure). Check the wording against the PDF before putting it in a manuscript.
- **The previous attempt.** It downloaded and text-extracted the four PDFs and left a partial `foundations.md`. The resume pass checked that draft against the cached texts and corrected it; the corrections are listed at the end of this note.

---

# (a) NOMP: Newtonized Orthogonal Matching Pursuit (Mamandipoor, Ramasamy, Madhow, IEEE TSP 2016)

## (a)1. Plain-English summary

A receiver hears a few pure tones mixed together with noise and wants to know the exact pitch and loudness of each tone. NOMP finds the tones one at a time.

Each time, it first makes a rough guess by looking at a fine but finite list of candidate pitches, which is fast to do with an FFT. It then polishes that guess with a few steps of Newton's method, so the pitch can land anywhere, not only on the list.

The clever part is what happens after each new tone is added. NOMP goes back and re-polishes all the tones it found before, because the new tone may have been fooling them. When what is left over looks like plain noise, it stops. "Looks like noise" is decided by a threshold set from the known noise level, so that false detections stay rare.

The authors show that this simple loop is almost as accurate as theory allows, and that it is faster than the competing methods they tested (MUSIC, atomic-norm denoising, Lasso, NLasso), except DOMP, a fixed-grid method that is much less accurate.

## (a)2. Problem, system model, and symbol mapping

**Their model (Sec. I, eqs. 1-2, p. 1).**
- There are N equally spaced samples y = sum over l = 1..K of g_l x(omega_l) + z, with z ~ CN(0, sigma^2 I_N).
- The atom x(omega) = [1, e^{j omega}, ..., e^{j(N-1) omega}]^T / sqrt(N) is a unit-norm sinusoid.
- The unknowns are K, the gains g_l and the frequencies omega_l in [0, 2 pi).
- For a uniform linear array (ULA), omega = 2 pi (d/lambda) sin(theta) (p. 1).
- SNR_l = |g_l|^2 / sigma^2 is the "integrated" SNR (Remark 1, p. 5). The per-sample SNR is SNR_l / N.
- Their simulations use N = 256 throughout (p. 3). They are 1-D only. Section VII extends the model to a known measurement matrix A (eq. 32, p. 13).

**Our model.**
- The measurement is Y = W^H H F + Z, a 16x16 matrix, with 16x16 DFT codebooks W and F. The decoders work on H~ = W Y F^H. This map is unitary: in `stress/check_sim.log`, the noise variance in Hant is 0.9990 for sigma^2 = 1.
- H = sqrt(256) sum_l alpha_l a_r(u_l) a_t(v_l)^H, where a_r and a_t are unit-norm steering vectors (`stress_sim.py` docstring).
- The spatial frequencies are u = pi cos(psi) (AoA) and v = pi cos(phi) (AoD) (`verify/nomp_mine.py`).
- So our problem is a 2-D version of theirs with N = 256 samples. Each atom is a 16x16 outer product.

| their symbol | meaning | our symbol / value |
|---|---|---|
| y in C^N, N = 256 | measured vector | vec(H~) (256 entries) or, equivalently, vec(Y) (unitary map; with ideal codebooks, decoding on Y or on H~ is the same problem) |
| x(omega) | unit-norm sinusoid | vec(a_r(u) a_t(v)^H): a 2-D atom with two frequencies (u, v) |
| omega_l | frequency | the pair (u_l, v_l) = (pi cos psi_l, pi cos phi_l) |
| g_l | complex gain | 16 alpha_l, with alpha_l ~ CN(0, 1/L), sorted (`stress/RESULTS.md` setup) |
| K | number of sinusoids | L (= 3 in the benchmark; given to every method) |
| sigma^2 | noise variance per sample | 10^(-SNR/10) per beam entry (`stress_sim.base_scenes`) |
| SNR_l = \|g_l\|^2/sigma^2 (integrated) | per-path SNR | our SNR is per-sample, total over paths. Mean-path integrated SNR = our SNR + 19.31 dB; median of the weakest of 3 paths = our SNR + 12.95 dB (log A5) |
| A (Sec. VII) | known measurement matrix | W^H (.) F. With ideal unitary DFT codebooks it can be removed (to_ant). With impaired or quantised codebooks it is a known or unknown A. NOMP's Sec. VII extension covers a *known* A |
| gamma | grid oversampling | os_ = 4 (64x64 2-D FFT) in `nomp_greedy` / `nomp_mine.py` |
| R_s, R_c | Newton steps per new atom; cyclic rounds | 4 Newton steps for the new path; 3 cyclic rounds x 2 Newton steps (`notebook_src.py` lines 544-585) |
| tau | stopping threshold | not used in the benchmark (known L); used in stress M7 rule (a): tau = 10.145 sigma^2 |
| W, F, b, A_AGC | codebooks, ADC bits, AGC full scale | **not in NOMP**: its noise is Gaussian and unquantised |

## (a)3. The method step by step

1. **Single tone, maximum likelihood (Sec. II-A, eq. 3, p. 3).** Minimising the residual power over (g, omega) is the same as maximising S(g, omega) = 2 Re{y^H g x(omega)} - |g|^2 ||x(omega)||^2. In words, this asks how well a tone explains the data, minus a penalty for its energy.
2. **Eliminate the gain (eqs. 4-5, p. 3).** For a fixed omega, the best gain is g = x^H y / ||x||^2. What remains is the GLRT (generalized likelihood ratio test) score G_y(omega) = |x(omega)^H y|^2 / ||x||^2. It is a normalised periodogram: how much energy the data has at that frequency.
3. **Detection (p. 3).** Evaluate G on the oversampled grid {2 pi k/(gamma N)}, with gamma = 4 in all their simulations, using a zero-padded FFT. Take the maximum.
4. **Newton refinement (eqs. 6-8, p. 3).**
   - Update omega' = omega - S_dot / S_ddot, where S_dot = Re{(y - g x)^H g x'} and S_ddot = Re{(y - g x)^H g x''} - |g|^2 ||x'||^2.
   - In words: one Newton step on the likelihood, taken only where the function curves downward (S_ddot < 0).
   - The gain is then re-fitted.
   - The **refinement acceptance condition (RAC)** keeps a step only if G strictly increases. This guarantees that the residual energy never goes up (p. 3).
5. **Several tones (Algorithm 1, p. 4).** Each iteration has four steps:
   - IDENTIFY a new tone on the residual.
   - SINGLE REFINEMENT: R_s Newton steps on the new tone.
   - CYCLIC REFINEMENT: R_c rounds, in which every tone in turn is refined against the data minus all other tones. This is the "feedback".
   - LEAST-SQUARES: refit all gains, g = X^dagger y.
   - Their settings: R_s = 1; R_c in {1, 3, 5} depending on difficulty (p. 4; Table II, p. 10).
6. **Stopping (Sec. III-A, eqs. 12-13, p. 5).**
   - Stop when the residual's largest DFT-grid periodogram value is below tau.
   - If the residual is pure noise, P(max > tau) = 1 - (1 - e^{-tau/sigma^2})^N (eq. 12). Setting this equal to P_fa gives the exact tau.
   - The large-N form is tau = sigma^2 (ln N - ln ln(1/(1-P_fa))) (eq. 13).
   - Our log A1 gives tau = 10.1453 sigma^2 for N = 256 and P_fa = 0.01. The exact and asymptotic forms agree to 2e-5.
7. **Miss probability (eqs. 14-15, p. 6).** The FFT peak of an off-grid tone is reduced by a factor alpha. Averaged over a uniform offset, E[alpha] is printed as 0.88. Our numerical average for N = 256 is 0.8727 (log A2). So "0.88" is a slight round-up of 0.873.
8. **Why oversample (Sec. IV, pp. 6-7).**
   - Newton's method converges to the peak only if the coarse guess lies within about +-0.45 DFT bins of the true frequency.
   - So the grid must be finer than the DFT grid: a minimum factor of about 1.12, and they use 4. Our recomputation gives a largest interval of 0.4502, a spacing of 0.9003 bins and a minimum factor of 1.111 (log A3). This matches their "about 1.12" up to rounding.

**Re-implementation pseudo-code** (the paper's Algorithm 1, 1-D). The 2-D form we use replaces omega by (u, v) and the scalar Newton step by a 2x2 Newton step.

```
input: y (N complex), threshold tau  [or: known K], gamma = 4, R_s = 1, R_c in {1,3,5}
P = []                                                   # list of (g, w)
def G(r, w):  return |x(w)^H r|^2 / ||x(w)||^2
def newton(r, g, w):                                     # one step on S(g, w) for measurement r
    e  = r - g x(w)
    Sd = Re{ e^H g x'(w) };  Sdd = Re{ e^H g x''(w) } - |g|^2 ||x'(w)||^2
    if Sdd < 0:  w2 = w - Sd/Sdd;  g2 = x(w2)^H r / ||x(w2)||^2
                 if G(r, w2) > G(r, w): return g2, w2      # RAC: accept only strict improvements
    return g, w
loop:
    r = y - sum_{(g,w) in P} g x(w)
    if max_{k=0..N-1} G(r, 2 pi k / N) < tau: break      # stopping test on the N-point DFT grid
    w = argmax over {2 pi k/(gamma N)} of G(r, .)  (zero-padded FFT);  g = x(w)^H r / ||x||^2
    repeat R_s: g, w = newton(r, g, w)
    P.append((g, w))
    repeat R_c: for each l in P: r_l = y - sum_{k != l} g_k x(w_k);  (g_l, w_l) = newton(r_l, g_l, w_l)
    X = [x(w_1) ... x(w_m)];  [g_1..g_m] = pinv(X) y          # least-squares gain update
return P
```

**How our implementation differs** (`notebook_src.py` lines 544-585; independent copy `verify/nomp_mine.py`, scene-by-scene equal to 5.6e-14 rad, `RESULTS_SUMMARY.md` section 1):
- It is 2-D (64x64 FFT, i.e. gamma = 4 in each dimension).
- There is no RAC check. A step is taken where the 2x2 Hessian is negative definite, and each component is clipped to 2 pi/64.
- The new path gets 4 Newton steps instead of R_s = 1.
- The cyclic refinement runs 3 rounds of 2 Newton steps, with the LS gains re-fitted before each path.
- In the benchmark, L is known, so there is no CFAR stop.

## (a)4. Assumptions

| question | answer in the paper |
|---|---|
| Noise variance known? | **Yes**, for the CFAR threshold (Sec. III-A). The contributions list says it can use an *estimate* of the noise variance (p. 2), but no estimator is specified or evaluated. |
| Number of tones known? | **No.** It is estimated by CFAR stopping (Sec. III) or by BIC (Remark 2, App. I). |
| Noise model | Circular white Gaussian, unquantised. No dithering, no AGC, no ADC. |
| Snapshots | One (single measurement vector). |
| Grid | Gridless output. The grid is used only for coarse detection (gamma = 4); refinement is on the continuum. |
| Separation | Well-separated (Delta omega_min = 2.5 DFT bins) or close (0.5 bins) tones (Table I). Separation is needed for well-posedness (p. 4). |
| Measurement matrix | Identity in the main text; a known random A in Sec. VII (compressive, M = 64). |

## (a)5. Reported results (N = 256, K = 16, 300 runs per scenario unless stated)

| result | numbers | where |
|---|---|---|
| Scenarios | (1) 25 dB, sep 2.5 bins; (2) 25 dB, sep 0.5; (3) U[15,35] dB, sep 2.5; (4) U[15,35] dB, sep 0.5 | Table I, p. 9 |
| NOMP settings | R_c = 1/3/1/3, R_s = 1, gamma = 4 | Table II, p. 10 |
| Run time for 300 runs, Sc. 1/2/3/4 (s) | NOMP 6.92 / 14.26 / 6.88 / 14.19; AST 1.09e3 / 1.02e3 / 1.12e3 / 1.18e3; NLasso 29.68 / 29.63 / 34.32 / 36.37; Lasso 26.63 / 27.62 / 32.06 / 33.71; DOMP 2.66 / 2.81 / 2.79 / 2.80; MUSIC 19.83 / 20.15 / 20.07 / 19.69 | Table III, p. 12 |
| Run time vs R_c = 1 / 3 / 5 (s, 300 runs) | Sc. 1: 6.92 / 14.24 / 21.67; Sc. 2: 7.01 / 14.26 / 21.71; Sc. 3: 6.88 / 14.21 / 21.61; Sc. 4: 6.96 / 14.19 / 21.50 | Table IV, p. 12 |
| Accuracy (MSE vs CRB/ZZB) | Sc. 1: AST, NOMP and NLasso reach the CRB at high SNR; MUSIC tracks the bound at all SNRs but does not reach it; DOMP and Lasso hit error floors. In Sc. 2 is not described in the text we read (figure only; not re-checked). Sc. 4: NOMP outperforms all others and tightly follows the CRB; AST is best only at very low SNR. Values are read from figures only (no table) | Sec. VI-A, Figs. 6-11, pp. 10-11 |
| More cyclic rounds | Gains diminish; little beyond R_c = 3 (Sc. 2 and 4) | Fig. 12, p. 11 |
| Oversampling | gamma = 20 costs about 5% more run time. DOMP time grows from 3.25 to 7.96 s as gamma goes 20 -> 100, with only a marginal MSE gain. Lasso time grows from 72.16 to 191.22 s as gamma goes 20 -> 50 | pp. 8, 12-13 |
| Convergence (noiseless, K = 16, sep 2.5) | With cyclic refinement, the residual drops to machine precision after 16 iterations | Fig. 5, p. 8 |
| CFAR stopping | The measured false-alarm rate tracks the nominal one at 14, 15 and 18 dB. False alarm = over-estimating K. 300 runs per point, K = 16, sep 2.5 | Fig. 2, p. 5-6 |
| Miss / ROC | At 18 dB the miss rate is negligible; at 14 dB (below the ZZB threshold) it stays clearly above zero. Measured and computed ROCs agree | Fig. 3, p. 6 |
| Model order | CFAR and BIC stopping are both accurate in all 4 scenarios. DOMP over-estimates K; Lasso under-estimates it (Sc. 2) | Figs. 14-16, pp. 13-14 |
| Compressive (M = 64, Sc. 1, R_c = 3) | K = 13 approaches the CRB; K = 16 gives large errors | Fig. 17, pp. 14-15 |

Our recomputation of eqs. (10)-(11) for N = 256 (log A7):
- The single-tone ZZB is within 3 dB of the CRB from an integrated SNR of 14 dB upward. It is 14.8 dB above the CRB at 12 dB and 26 dB above at 10 dB.
- This is consistent with their statement that 14 dB lies at or below the threshold.

## (a)6. Citable claims (location; use)

1. **The feedback step matters.** Cyclic re-refinement of the earlier tones is where NOMP departs from forward greedy pursuit. The authors call it "crucial for fast convergence and high estimation accuracy" (Sec. II, Steps 7-9, p. 4; Fig. 5, p. 8). The paper has no Sec. II-B. *Use:* justifies our cyclic re-estimation; cite whenever describing our decoder.
2. **Oversampling is needed even in the easiest case.** Coarse detection must use a grid finer than the DFT grid, even for one noiseless tone (minimum factor about 1.12, p. 6). Sec. IV, p. 7, says "γ = 4 or more works very well, independent of the number" of sinusoids. *Use:* justifies os_ = 4.
3. **A noise-based stopping rule.** With known sigma^2, tau = sigma^2 ln N - sigma^2 ln ln(1/(1-P_fa)) gives a false-alarm rate close to the nominal one in simulation (Sec. III-A, eq. 13, p. 5; Fig. 2). *Use:* our unknown-L rule (a) in stress M7.
4. **Speed (the single verbatim quote).**
   - Quote: "NOMP is faster than all of the other methods in all four Scenarios" (Sec. VI-B, p. 12).
   - The comparison set is AST, Lasso, NLasso and MUSIC. DOMP (OMP on the 20x grid) is faster in Table III (2.66 s vs 6.92 s) but much less accurate. The sentence just before the quote says so, so the quote must not be cut away from it.
   - *Use:* rebuts papers that call NOMP slow (`lit/baselines_survey.md`).
5. **Near-optimal for separated tones.** With one cyclic round, NOMP reaches the CRB in the well-separated scenarios (Sec. VI-A-3, p. 11). In the close-spacing scenario with unequal SNRs it beats all compared methods and tracks the CRB, except at very low SNR, where AST is better (p. 11). *Use:* why NOMP is a strong ideal-hardware baseline.
6. **On-grid greedy methods have a floor.** DOMP's error floor does not go away with a 100x grid. The authors attribute it to hard decisions, not only to grid mismatch (Sec. VI-C, p. 12). *Use:* why on-grid estimators (including on-grid GAMP/VAMP, paper (c)) need a continuous refinement step.
7. **The theory is pessimistic.** The convergence bounds ignore the refinement steps, and quantifying the benefit of refinement is left open (Sec. V, p. 7; Sec. VIII, p. 15). *Use:* honest limits when we invoke NOMP theory.
8. **Known measurement matrices are allowed.** The algorithm extends directly to y = sum g_l A x(omega_l) with known A, by using normalised atoms A x / ||A x|| (Sec. VII, eq. 32, p. 13). *Use:* decoding with the receiver's known quantised codebook (stress M4: "network not needed").

## (a)7. Limitations: what the paper does not show

- **1-D only.** The 2-D joint AoA/AoD version (16x16 outer-product atoms, 2x2 Newton step) is our extension.
- **Gaussian, unquantised noise only.** There is no ADC, AGC, clipping or dithering, and no hardware impairments. All its optimality evidence (CRB, ZZB) assumes Gaussian noise.
- **The bound is vacuous at the oversampling actually used.**
  - Theorems 2-3 carry the factor (1 - 2 pi/gamma)^-1. This is undefined or negative for gamma <= 2 pi.
  - So at the gamma = 4 used in every simulation, the rate bound says nothing. It becomes meaningful only for gamma > 6.28; it is 4.66 at gamma = 8 and 1.46 at gamma = 20 (log A4, our arithmetic).
  - The paper notes that the bound approaches OMP's for large gamma (p. 8), but it does not point out that it is vacuous at gamma = 4.
- **The noise variance is assumed known** for stopping. No estimator for it is evaluated; paper (b) addresses this gap.
- **The miss analysis ignores interference** from other tones (Sec. III-B). The false-alarm and miss statistics hold for separated tones above the ZZB threshold.
- **Its metrics are not our metric.** Results are given as MSE and CCDF of squared frequency error. A "detection" is a tone within 0.25 DFT bins. Our Pd uses 1 degree per angle, which is 0.14 DFT bins at broadside and 0.07 at psi = 30 deg in a 16-element array (log A8). Their accuracy numbers therefore do not translate directly into our Pd.
- **Run times lack context.** The paper states no hardware, language or thread count (only that code is available [1]); UNVERIFIED. The absolute times are not comparable to our ms/scene.
- **A small rounding.** E[alpha] is printed as 0.88; we get 0.873 (log A2).

## (a)8. Relevance to us

- **What we reuse.** NOMP is our classical decoder, in the 2-D form described in (a)3. Credit it as "NOMP (Mamandipoor et al., 2016)". Any learned decoder derived from it must be called "unfolded from NOMP".
- **Baseline status: mandatory, and already the strongest on ideal hardware.**
  - On the official 8000-scene bank, NOMP-only gives Pd 0.7581.
  - It beats the U-Net (0.7366) and the ResNet (0.7103) of Lloria et al.
  - It beats our own all-neural pipeline, 0.7508 (-0.0073 [-0.0091, -0.0056]*) (`RESULTS_SUMMARY.md` sections 1 and 5).
- **Fair comparison.**
  - Same banks, same Pd/strict-Pd/RMSE metric with Hungarian matching, same L for every method.
  - Report NOMP's settings (gamma = 4, 3 cyclic rounds) and our deviations: no RAC, clipped steps, 4 seed steps.
  - Show that larger settings do not change the conclusion: the NOMP os8 rc6 configuration exists in `unfolded_v2/build/ev.py`. The legal candidate-selection rules in `unfolded_v2/diag/REPORT.md` gain only +0.0025 to +0.0033* over os4rc3, which that report places at about the os8rc6 level.
- **Under low-resolution ADCs, NOMP is a reference, not the main rival.** At 1 bit / 15 dB:

  | method | Pd |
  |---|---|
  | NOMP | 0.191 |
  | consistent NOMP | 0.237 |
  | ADC-CNN read-out alone | 0.419 |
  | quantiser ML (13 starts + EM) | 0.540 |
  | hybrid (CNN seeds + quantiser ML) | 0.572 |

  NOMP fails here because its Gaussian likelihood is wrong (section (a)7), not because its search fails.
- **SNR context.**
  - Our project SNR is per sample. A mean path sits about 19.3 dB higher in NOMP's integrated scale, and the weakest of 3 paths (median) about 13 dB higher (log A5).
  - So at our SNR <= 0 dB, the weakest path is around or below the single-tone ZZB threshold of about 13-14 dB integrated (log A7).
  - This is a 1-D analogy, not a 2-D bound, so treat it as approximate. It agrees with the verified finding that at low SNR the oracle-seeded gap comes from the objective itself, not from NOMP's search (`RESULTS_SUMMARY.md` section 1).
- **Network design.**
  - NOMP's three ingredients (coarse grid -> Newton -> cyclic feedback with LS gains) are the right skeleton for a decoder.
  - The learned MLP step of `lurd.pt` correlates 0.91-0.92 with the Newton step, so learning largely re-discovered Newton (`RESULTS_SUMMARY.md` section 5).
  - Room for learning exists only where NOMP's assumptions break: quantisation, impairments, unknown noise level, seeding.

## (a)9. If a reviewer asks...

1. **"Your network does not beat NOMP on ideal hardware. Why should we care?"**
   - Correct, and we report it: NOMP-only 0.7581 vs all-neural 0.7508 (-0.0073*).
   - Our claims are restricted to regimes where NOMP's model is wrong. With 1-bit ADCs at 15 dB, CNN seeds + quantiser ML reach 0.572, vs 0.540 for multi-start quantiser ML (+0.032 [+0.022, +0.042]*, replicated +0.028* / +0.036*). The CNN read-out alone reaches 78-92% of ML's Pd at about 1/80 of its cost.
2. **"Is your NOMP a strawman?"**
   - No. An independent from-scratch implementation matches the cached NOMP predictions scene by scene (max difference 5.6e-14 rad).
   - Its settings lie inside the paper's range (gamma = 4, R_c = 3; Table II).
   - Heavier settings or legal candidate selection add about +0.003 at most. We list our deviations (no RAC, clipped steps) openly.
3. **"NOMP assumes Gaussian noise. Comparing it under 1-bit quantisation is unfair to NOMP."**
   - Agreed. That is why the ADC tables also include quantisation-consistent NOMP and the quantiser-ML decoder (exact cell likelihood, 13 starts + EM).
   - We claim the hybrid's +0.032 only against the strongest classical method (quantiser ML), never against NOMP.

## (a)10. Glossary

- **Line spectral estimation:** finding the frequencies of a few sinusoids in noisy samples. Array angle estimation is the same problem in space.
- **Atom / dictionary:** one candidate sinusoid x(omega); the set of all of them.
- **OMP (orthogonal matching pursuit):** greedy method that adds the best-matching atom, then re-fits all gains by least squares.
- **Basis mismatch / off-grid effect:** the error that arises when the true frequency is not on the search grid.
- **GLRT cost G(omega):** the normalised periodogram |x^H y|^2/||x||^2, i.e. how much energy one tone at omega would explain.
- **Newton step:** a jump to the peak of a local quadratic fit of the objective.
- **RAC:** keep a refinement only if it improves the fit.
- **Cyclic refinement ("feedback"):** re-polish every earlier tone against the data minus all other tones.
- **DOMP:** OMP on a fixed oversampled grid, without refinement. **NOMP-:** NOMP without cyclic refinement.
- **CFAR:** a threshold that keeps the probability of a false detection at a chosen value P_fa, using only a model of the noise.
- **ROC:** the trade-off curve between miss rate and false-alarm rate.
- **CRB:** the lowest variance any unbiased estimator can reach. **ZZB:** a bound that also captures the large "threshold" errors at low SNR.
- **Integrated vs per-sample SNR:** total tone energy over noise per sample, vs that divided by N.
- **Atomic norm / AST:** a convex, gridless sparse estimator, and its soft-thresholding solver.
- **Lasso / NLasso:** l1-regularised fit on a grid; NLasso is Lasso followed by NOMP's cyclic refinement.
- **BIC:** Bayesian information criterion, a model-order rule that penalises extra parameters.

---

# (b) NOMP-CFAR (Xu, Zhu, Fang, Zhang, Xu, IEEE TAES 2023)

## (b)1. Plain-English summary

NOMP's stopping rule needs the noise level, and in real radars the noise level drifts. This paper makes NOMP estimate the noise from the data itself, the way radar CFAR detectors do: by averaging the power in "reference cells" around the candidate peak.

It works in two phases:
1. Run NOMP for a generous, fixed number of steps (twice the true number of targets in their simulations) to get a candidate list.
2. Repeatedly test each candidate against its own data-driven threshold. Remove the worst candidate if it fails, re-polish the rest, and add a new candidate only if one passes.

The result keeps the false-alarm rate at the chosen value even when the noise level changes from trial to trial. It costs about 1 dB of SNR compared with NOMP given the true noise level.

## (b)2. Problem, system model, and symbol mapping

- **Their model (Sec. II, eqs. 1-8, PDF pp. 4-5).** A D-dimensional line spectrum Y = sum_k x_k a_{N1}(omega_{1,k}) o ... o a_{ND}(omega_{D,k}) + eps, with eps ~ CN(0, sigma^2) and **sigma^2 unknown**.
- Both K and the frequencies must be estimated (PDF p. 5).
- SNR is the integrated SNR, N |x_k|^2 / sigma^2 (eq. 9).
- Extensions: multiple snapshots (MMV, eq. 49) and compressive y = Phi(.) (eq. 53).

| theirs | ours |
|---|---|
| Y (D-dim), D = 2, N1 = N2 = 16, N = 256 | H~ (or Y) 16x16 |
| omega_{1,k}, omega_{2,k} | (u_l, v_l) |
| x_k | path gain (16 alpha_l in NOMP normalisation) |
| K, K_max | L (unknown in stress M7, L ~ U{1..8}); L_max = 10 in M7 |
| sigma^2 (unknown) | sigma^2 = 10^(-SNR/10) (oracle in rule (a); unknown in the blind rules) |
| N_r reference cells, guard cells | not used by us. Our blind CFAR estimates sigma^2 from the whole residual: sigma_hat^2 = E_10/(256-20) (`stress/m7_order.py`) |
| alpha (threshold multiplier) | kappa_b = 13.0 (tuned on separate banks) in M7 |
| S snapshots | 1 |
| Phi (compressive) | W^H (.) F: unitary with ideal codebooks, so no compression |
| ADC bits b, AGC A | **not modelled** (Gaussian noise only) |

## (b)3. The method step by step

1. **Soft CFAR test of one peak (Sec. III-A, eqs. 11-16, PDF pp. 6-7).**
   - Take the normalised D-dimensional DFT of the data and find its peak, the "cell under test".
   - Estimate sigma^2 as the mean power of N_r reference cells. Leave guard cells next to the peak, and exclude cells near other detected targets (PDF pp. 6-7).
   - Declare a target if peak / sigma_hat^2 > alpha.
2. **The threshold multiplier (Proposition 1, eqs. 19-20, PDF p. 7).**
   - Assume the reference cells are i.i.d. Gaussian (eq. 18).
   - Then P_fa = 1 - E_x[(1 - e^{-alpha x/N_r})^N], with x ~ Gamma(N_r). Solve this for alpha.
   - Our log B1 reproduces their numbers: N = 256, N_r = 50, P_fa = 1e-2 gives alpha = 11.221 (10.50 dB). The approximation eq. 34 gives 11.254. NOMP's known-noise multiplier is 10.145.
   - Remark 1: alpha -> NOMP's value as N_r -> infinity (log B4: 10.364 at N_r = 236).
   - A Monte Carlo of the detector under assumption (18) gives a realised P_fa of 0.0092 (log B5).
3. **Soft output (eq. 37, PDF p. 10).** Delta = 10 log10(peak / threshold), i.e. by how many dB the peak clears (Delta > 0) or misses (Delta < 0) its threshold.
4. **NOMP-CFAR (Algorithm 2, PDF p. 15).**
   - Initialisation: run NOMP for K_max steps to get candidates.
   - Then loop:
     - For every candidate k, add it back to the residual and compute Delta_k.
     - If the smallest Delta_k < 0, remove that candidate, then run cyclic refinement and LS on the rest.
     - Otherwise, run the CFAR test on the residual. If a new peak passes, add it (single refinement); if not, stop.
   - Only one candidate is removed or added per iteration. This handles a strong target that NOMP split into two weaker ones (PDF p. 14).

```
alpha = solve_{alpha} P_fa(alpha; N, N_r) = target     # eq. 19 (eq. 52 for S snapshots)
def CFAR(Y): Yt = DFT_D(Y)/sqrt(N); n* = argmax |Yt|^2; s2 = mean_{n in ref(n*)} |Yt_n|^2;  return 10 log10(|Yt_n*|^2 / (alpha s2))
K = NOMP(Y, K_max steps)                               # candidate set
while 0 <= |K| <= K_max:
    D_k = CFAR(residual(Y, K) + x_k A(w_k)) for k in K;  k* = argmin D_k
    if D_k* < 0: K.remove(k*); cyclic_refine(K); LS(K); continue
    if CFAR(residual(Y, K)) >= 0: K.add(single_refine(new peak)); continue
    break
return K
```

## (b)4. Assumptions

- **sigma^2:** unknown, but constant over the reference cells of one test.
- **Noise:** Gaussian, white, unquantised.
- **Reference cells:** treated as independent CN(0, sigma^2) (eq. 18, an approximation).
- **K:** unknown, with an upper bound K_max (K_max = 2K in the simulations; K_max = 32 in the real data, Sec. V, "unless stated otherwise").
- **Snapshots:** single, with an MMV extension.
- **Grid:** the oversampling gamma = 4 is inherited from NOMP. The CFAR test itself runs on the DFT grid (eq. 12).
- **Not modelled:** dithering, AGC and quantisation. Separation > 2.5 DFT bins in the simulations.

## (b)5. Reported results (N = 256, K = 16, sigma^2 = 1, P_fa = 1e-2, N_r = 50, gamma = 4, K_max = 2K, 3000 Monte Carlo trials)

| result | numbers | where |
|---|---|---|
| Threshold multipliers | NOMP-CFAR 11.22, CFAR approximation 11.25, NOMP 10.15 (10.50 / 10.51 / 10.06 dB). MMV: S = 10 gives 2.81, S = 50 gives 1.67. **All reproduced** (log B1-B2) | Sec. IV-A, Fig. 1, PDF pp. 17-18 |
| CFAR property, constant noise | Measured P_fa close to nominal (0.01-0.12) at 14 / 15 / 18 dB, for both NOMP and NOMP-CFAR. NOMP detects more; the gap shrinks with SNR | Sec. IV-B, Fig. 2, PDF p. 18 |
| Benefit of initialisation | At P_D = 0.6: NOMP 14 dB, NOMP-CFAR 15 dB, forward-only CFAR 17 dB. Above 17 dB, P_D is about 1 for both NOMP variants but about 0.8 for forward-only | Sec. IV-C, Fig. 3, PDF pp. 18-19 |
| SNR sweep 12-20 dB | P_D = 0.75 at 14 dB (NOMP) vs 15 dB (NOMP-CFAR), i.e. 1 dB loss. The MSE reaches the CRB from 16 dB (NOMP) and 17 dB (NOMP-CFAR) | Sec. IV-D, Fig. 4, PDF pp. 20-21 |
| MMV (10 dB) | P_fa close to nominal for S >= 3. NOMP and NOMP-CFAR give similar P_D | Sec. IV-E, Fig. 5, PDF p. 21 |
| Compressive (K = 8, 22 dB) | For P_D = 0.74, M/N = 0.24 (NOMP) vs 0.32 (NOMP-CFAR): 4/3 as many measurements. P_D -> 1 for M/N >= 0.5 | Sec. IV-F, Fig. 6, PDF pp. 21-22 |
| Noise level varying over trials (U[-u, u] dB, 28 dB per target) | NOMP violates the CFAR property; NOMP-CFAR keeps it. P_D >= 0.9993 | Sec. IV-G, Fig. 7, PDF pp. 22-23 |
| Real data (TI IWR1642, 77 GHz) | Exp. 1 range-azimuth: NOMP 19 points vs NOMP-CFAR 2 (two people). Exp. 2 range-Doppler: CFAR misses one person and has 1 false alarm; NOMP 47 points; NOMP-CFAR 24. Exp. 3: NOMP 28 vs NOMP-CFAR 10 (all 3 objects found). Threshold multipliers 11.03 / 11.06 **reproduced** (log B3) | Sec. V, Figs. 8-15, PDF pp. 22-27 |

## (b)6. Citable claims (location; use)

1. **NOMP's original stop assumes a known, constant noise variance.** The authors say that in practice "the noise variance is unknown and can often be variable" (Sec. III-A, PDF p. 5). *Use:* motivates blind stopping.
2. **Desired false-alarm rate without noise knowledge.** NOMP-CFAR keeps the chosen false-alarm rate without knowing the noise variance, including when the variance fluctuates (abstract, PDF p. 1; Sec. IV-G, Fig. 7). *Use:* the reference blind-order method.
3. **Cost of not knowing sigma^2: about 1 dB.** This is about 1 dB of SNR vs NOMP with known sigma^2 at P_fa = 1e-2. The abstract states it at P_D 0.8, the contributions at 0.76 (PDF p. 3), and Sec. IV-D at 0.75 (PDF p. 20). *Use:* expected price of blind stopping; cite as "about 1 dB".
4. **Many reference cells recover NOMP's rule.** With N_r -> infinity, the CA-CFAR multiplier tends to NOMP's known-noise threshold (Remark 1, PDF p. 9). *Use:* our blind CFAR, which estimates sigma^2 from the whole residual, is the large-N_r end of this family.
5. **Initialise with surplus candidates, then prune one at a time.** This beats forward-only CFAR detection, which is how NOMP normally stops (Sec. III-B, PDF p. 14; Sec. IV-C). *Use:* design of a stop / prune head.
6. **Keep target energy out of the noise estimate.** Reference cells must exclude guard cells and cells near other targets. Otherwise the noise estimate is biased and the false-alarm rate is off (Sec. III-A, PDF pp. 6-7). *Use:* any blind sigma^2 estimator for our 16x16 beamspace, where a few strong beams dominate.
7. **Their false alarm is counted per estimate.** It is the fraction of estimated frequencies at least 0.5 DFT bins from every true one (Sec. IV, PDF p. 17). NOMP instead counts "over-estimated K" per run (paper (a), p. 5). *Use:* align definitions before comparing rates.

## (b)7. Limitations

- **Gaussian, unquantised noise only.** Under 1-2-bit ADCs, eq. 18 (Gaussian reference cells) and eq. 19 do not hold. At 1 bit the noise level is not identifiable at all (project finding).
- **Inconsistencies inside the paper.**
  - The "1 dB" loss is quoted at P_D 0.8, 0.76 and 0.75 in three places.
  - In Sec. IV-C, the text says initialisation gives a 1 dB gain, but its own numbers (15 vs 17 dB at P_D = 0.6) imply 2 dB.
  - The multiplier is said to come from eq. (29), but it is eq. (19).
  - Whether the published version fixed these is UNVERIFIED.
- **Narrow comparisons.** Simulations compare only against NOMP. FFT-CFAR is excluded because it is poor in multiple-target scenarios (PDF p. 17), and VALSE is dismissed in one sentence without data (PDF p. 2). There are no MDL/BIC/AIC comparisons.
- **Loose detection radius.** It is 0.5 DFT bins, 3.6x looser than our 1-degree tolerance at broadside (log A8).
- **No accuracy ground truth for the real data.** Those experiments judge false-alarm counts visually; there are no error statistics against ground truth.

## (b)8. Relevance to us

- **What we reuse.**
  - The CFAR stopping logic for unknown L.
  - The lesson that a threshold derived for the N-point DFT grid must be applied on that grid.
  - Our stress M7 rule (a) applied NOMP's eq. 12 threshold (10.145 sigma^2) to the max over the 4x-oversampled 64x64 grid. The realised false-alarm rate on noise-only data was then 0.0741 / 0.0723 instead of 0.01. On the 16x16 grid it was 0.0104 / 0.0105 (`stress/check_sim.log`). Our 1-D check gives 0.0105 vs 0.0287 (log A6).
  - The fixed rule (a') uses a Monte Carlo kappa = 12.32. The same oversampled-grid issue is present in `unfolded_v2/build/ev.py: nomp_stop`, which uses eq. 13 on the 64x64 grid. Fix it or document it before reporting any unknown-L number from that code.
- **What we already have (verified, unquantised, L ~ U{1..8}; `stress/RESULTS.md` M7; fresh-seed check in `stress/adversarial/VERDICT.md` section 4).**

  | rule | P(L_hat = L) at 15 dB | P(L_hat = L) at 0 dB |
  |---|---|---|
  | NOMP + CFAR with oracle sigma^2 | 0.890 | 0.491 |
  | blind CFAR (kappa_b = 13) | 0.933 | 0.457 |
  | MDL | 0.922 | 0.463 |
  | CNN heat-map count | 0.868 | 0.476 |
  | AIC | never stops | never stops |

  These are the original-seed values (VERDICT, column 'orig'). Fresh-seed values (VERDICT section 4): oracle CFAR 0.880 / 0.478; blind CFAR kappa_b = 13 0.924 / 0.448; the Monte Carlo kappa re-derived on a fresh seed is 12.27 (original 12.32).

  The path recall of the CFAR rules is within 0.002 of known-L NOMP at 15 dB and 0.009-0.015 below it at 0 dB.
- **Baseline status.**
  - Any claim about unknown L must include NOMP-CFAR (Algorithm 2, with the CA-CFAR alpha from eq. 19) or state that it was not run. **It has not been run in this project.**
  - Our blind CFAR is a simpler relative: one global sigma^2 from the residual, and a tuned multiplier. Its kappa_b = 13 is not directly comparable to Xu's alpha (10.36 for N_r = 236 on the DFT grid, log B4), because it thresholds the oversampled-grid maximum.
  - **Fair comparison:** same banks, same Pd and P(L_hat = L) metrics, same L_max, and the same grid for the test statistic as the one the threshold was derived for.
- **Network design.**
  - A learned stop head should be judged against blind CFAR and MDL, not against oracle-sigma^2 CFAR only.
  - The unfolded-v2 stop head did not improve validation (deviations.txt items 5 and 10).
  - Unknown L **with** ADC quantisation is open: no verified result exists, and Gaussian CFAR theory does not apply there (see `gnomp.md` in this folder for the few-bit line of work).

## (b)9. If a reviewer asks...

1. **"Your benchmark gives every method the true L = 3."**
   - True for the main tables. The stress study M7 (L ~ U{1..8}) shows that NOMP with CFAR or blind stopping recovers L in 0.89-0.93 of scenes at 15 dB and about 0.46-0.49 at 0 dB, with path recall close to known-L NOMP.
   - We have no unknown-L result under ADC quantisation, and we say so.
2. **"Why not NOMP-CFAR (Xu et al.) as the blind baseline?"** It has not been implemented yet. Until it is, our blind-CFAR and MDL rules are reported as simpler blind baselines, and no claim of beating NOMP-CFAR is made.
3. **"Is your threshold computed correctly?"** Xu's multipliers are reproduced exactly (11.22, 11.25, 10.15, 2.81, 1.67, 11.03, 11.06; log B1-B3). We also found and documented that applying the DFT-grid threshold to an oversampled grid raises the false-alarm rate (0.07 in 2-D). The Monte Carlo-calibrated kappa corrects this.

## (b)10. Glossary

- **CFAR:** constant false alarm rate, a threshold scaled to a running noise estimate.
- **CA-CFAR:** cell-averaging CFAR, where the noise estimate is the mean power of nearby cells. **OS-CFAR:** uses the r-th smallest reference value instead of the mean.
- **CUT:** cell under test (the peak). **Reference / training cells:** cells used to estimate noise. **Guard cells:** cells skipped next to the CUT because they may contain target energy.
- **Threshold multiplier alpha:** the threshold divided by the noise estimate.
- **Soft CFAR Delta:** by how many dB the peak clears its threshold.
- **K_max:** upper bound on the number of targets.
- **MOE:** model-order estimation.
- **MMV:** multiple measurement vectors (snapshots).
- **Target masking:** a strong target hiding a weak nearby one.

---

# (c) GAMP: Generalized Approximate Message Passing (Rangan, ISIT 2011; full version arXiv 1010.5141v2)

## (c)1. Plain-English summary

Many sensing problems look like this: an unknown vector is mixed by a known matrix, and then each mixed value is measured by some noisy, possibly nonlinear device, for example a coarse quantiser. Exact Bayesian estimation is hopeless because the matrix couples everything.

GAMP turns the big problem into a loop of two cheap steps:
1. Multiply by the matrix and its transpose.
2. Solve many tiny one-number problems. At the outputs it asks "given this measurement, what was the value before the device?"; at the inputs it asks "given this noisy guess, what does the prior say?".

The loop usually settles in 10-20 iterations. For large random Gaussian matrices, a simple set of scalar equations ("state evolution") predicts its error exactly. Any per-measurement likelihood can be plugged in, which is why GAMP became the engine for quantised compressed sensing and for the few-bit mmWave estimator of Mo et al.

## (c)2. Problem, system model, and symbol mapping

**Their model (Sec. I, eq. 1, p. 1).**
- x ~ prod_j p_{X|Q}(x_j | q_j), an i.i.d. prior; z = A x with A known, of size m x n; y_i ~ p_{Y|Z}(y_i | z_i), a separable output channel.
- Estimate x and z, either as MMSE (sum-product) or as MAP (max-sum).
- The analysis is real-valued, while Algorithm 1 is written for complex quantities (p. 4).

| theirs | ours (for a GAMP-based few-bit baseline such as Mo et al.) |
|---|---|
| x in R^n (i.i.d. prior) | sparse coefficients on a (16G)x(16G) grid of (u, v); n = 256 G^2 (complex) |
| A (m x n) | dictionary that maps grid coefficients to the 256 beam samples: W^H a_r(u_i) a_t(v_j)^H F stacked |
| z = A x | noiseless beamspace measurement W^H H F |
| p_{Y|Z} | our ADC: y = cell of the b-bit mid-rise quantiser applied to Re and Im of z + noise, with step 2A/2^b and per-scene peak AGC A; interval-censored Gaussian likelihood |
| tau^w | per-real-component noise variance sigma^2/2, with sigma^2 = 10^(-SNR/10) |
| p_{X|Q}, q | Bernoulli-Gaussian prior (as in Mo et al.); q unused |
| sparsity rho | about L / n (3 paths); off-grid paths leak over several grid points |
| beta = n/m | n/256 = G^2 (overcomplete) |
| L | not a GAMP parameter; extracted afterwards as the top-L support peaks |

## (c)3. The method step by step

- **Output linear step (eqs. 5a-5c).** p_hat = A x_hat - tau^p s_hat(t-1), with tau^p = |A|^2 tau^x. This is the prediction of z with an "Onsager" correction for the reuse of old information.
- **Output nonlinear step (eqs. 6a-6b).** s_hat = g_out(p_hat, y, tau^p) and tau^s = -dg_out/dp_hat. Each measurement corrects its own prediction.
- **Input linear step (eqs. 7a-7b).** tau^r = 1 / (|A^T|^2 tau^s) and r_hat = x_hat + tau^r A^T s_hat. Each coefficient now looks like the truth plus Gaussian noise of variance tau^r.
- **Input nonlinear step (eqs. 8a-8b).** x_hat = g_in(r_hat, q, tau^r) and tau^x = tau^r dg_in/dr. This is a scalar denoiser using the prior.
- **Choice of the scalar functions (Sec. IV, Table I, p. 6).**
  - Sum-product (MMSE): g_in is the posterior mean of X given R = X + N(0, tau^r), and g_out = (E[z | y, p_hat, tau^p] - p_hat)/tau^p (eq. 36).
  - Equivalently, g_out = d/dp_hat log p(y | p_hat, tau^p) (eq. 38): the output step is the score of the measurement likelihood with z ~ N(p_hat, tau^p).
  - For AWGN, g_out = (y - p_hat)/(tau^w + tau^p) (eq. 42), and GAMP reduces to AMP.
- **Our check for our ADC (log C1-C2).** For 1- and 2-bit cells, eq. 36 equals the numerical derivative of the cell log-likelihood (eq. 38) to 3e-10 (1 bit) and 2.5e-7 (2 bit). The variance identity eq. 37 holds to 3e-9 and 4.6e-6. A very narrow cell reproduces the AWGN formulas (42)-(43) exactly. So GAMP's output step for our quantiser is exactly the cell-probability model that our quantiser-ML code (`mlq.py`) already evaluates.
- **State evolution (Sec. V, Algorithm 3, Claim 1, pp. 8-9).** For i.i.d. Gaussian A with n/m -> beta, each coefficient behaves asymptotically like a scalar Gaussian channel whose noise level follows simple scalar recursions.

```
# sum-product GAMP, complex form, adapted to our ADC (Re/Im quantised separately)
init: x_hat = E[x] (prior mean), tau_x = var[x], s_hat = 0
repeat T (about 10-25) times:
    tau_p = |A|^2 tau_x ;  p_hat = A x_hat - tau_p * s_hat
    for each beam i (Re and Im separately; per real part the prior is N(Re p_hat, tau_p/2), noise var sigma^2/2):
        z0, vz = mean and variance of z_i given z_i ~ CN(p_hat_i, tau_p_i) and "Re(z_i + w_i) in cell_R, Im(z_i + w_i) in cell_I"
        s_hat_i = (z0 - p_hat_i)/tau_p_i ;  tau_s_i = (1 - vz/tau_p_i)/tau_p_i
    tau_r = 1/(|A|^H tau_s) ;  r_hat = x_hat + tau_r * (A^H s_hat)
    x_hat, tau_x = posterior mean / variance of x_j under the BG prior given r_hat_j ~ x_j + CN(0, tau_r_j)
    (optional EM: update the BG sparsity and variance from the posteriors; optional damping)
output: support map |x_hat| on the (u, v) grid -> top-L peaks -> continuous refinement (NOMP / quantiser-ML)
```

## (c)4. Assumptions

- **Distributions:** the prior and the output channel must be known (EM is suggested when they are not; Conclusions, p. 13).
- **Noise variance:** known, as part of p_{Y|Z}.
- **Theory:** large n and m with a fixed ratio, A i.i.d. Gaussian, and pseudo-Lipschitz scalar functions (Sec. V-B, p. 8). In Sec. VII, A_ij ~ N(0, 1/n), while Sec. V-B states N(0, 1/m) (a minor inconsistency).
- **Not specified:** grid, snapshots, dithering, AGC. GAMP is generic; quantisation enters only through p_{Y|Z} (Sec. II-A b, p. 3).

## (c)5. Reported results

- **Single simulation (Sec. VII, Fig. 4, pp. 11-13).** Bernoulli-Gaussian x (rho = 0.1), (m, n) = (500, 1000), i.i.d. Gaussian A, sigmoid output y = 1/(1 + e^{-6.1 z}) + w with tau^w = 0.01, 100 trials, median NSE.
- **Gain from modelling the nonlinearity.** The true-nonlinearity GAMP is more than 11 dB better than a linearised GAMP.
- **Convergence.** Both converge in about 10-15 iterations.
- **Accuracy of state evolution.** It predicts the simulations to within 0.2 dB (linearised) and 0.4 dB (nonlinear), including a small rise in the linearised error after iteration 7.
- **Algorithm variants.** The simplified Algorithm 2 gives "virtually no difference" from Algorithm 1 in simulations the paper does not report (Sec. VII, p. 12).

## (c)6. Citable claims (location; use)

1. **Generality.** GAMP admits essentially arbitrary priors and separable output channels, including nonlinear ones, and covers both MAP and MMSE variants (Sec. I, p. 1). *Use:* the quantised likelihood is prior art in this framework; credit it.
2. **Quantisation is in scope.** Quantisation noise is named as an application of non-Gaussian output channels (Sec. II-A b, p. 3; quantised compressed sensing cited in the Conclusions, p. 13). *Use:* same as claim 1.
3. **Cheap iterations.** The cost per iteration is dominated by multiplying by A and A^T, which is O(mn) and less for structured transforms such as Fourier. Typically 10-20 iterations suffice (Sec. III-A, p. 4; Sec. I, p. 2). *Use:* latency expectations for a GAMP baseline.
4. **The output step is a score function.** The sum-product output function is the derivative of log p(y | p_hat, tau^p) (eq. 38, Sec. IV-B, p. 7). *Use:* links GAMP to our quantiser-ML gradient (log C1).
5. **Guarantees only for i.i.d. Gaussian A.** Extending them to other matrices is listed as an open question (Sec. V; Conclusions, p. 13). *Use:* GAMP has no guarantee for our structured DFT beamspace dictionary.
6. **The main result is a claim, not a theorem.** The proof is only sketched (Sec. I-A, p. 3; Appendix A, p. 14). *Use:* cite as "argued" or "claimed".
7. **Known distributions are a key limitation.** Both variants need the true input and output distributions; EM-based adaptation is the suggested remedy (Conclusions, p. 13). *Use:* GAMP, like quantiser ML, needs sigma^2; our CNN read-out does not.
8. **Modelling the nonlinearity pays.** In their example it gains over 11 dB vs a linearised model (Sec. VII, p. 12). *Use:* supports exact-likelihood decoding over "quantisation = Gaussian noise".

## (c)7. Limitations

- **The theory does not cover our setting.** State evolution holds only for large i.i.d. Gaussian A; our dictionary is a structured, overcomplete DFT frame with only 256 measurements.
- **The prior does not match our signal.** GAMP's prior is i.i.d. over grid coefficients, whereas our signal is 3 off-grid paths. The on-grid output therefore needs continuous refinement (paper (a), claim 6).
- **One small example.** There is a single i.i.d. simulation, with no structured matrices, no quantiser and no confidence intervals.
- **Convergence is not addressed.** The paper says nothing about convergence or damping for non-i.i.d. A beyond calling it open. Later remedies are outside this paper and are UNVERIFIED here.

## (c)8. Relevance to us

- **What it is to us.** GAMP is the engine of the Mo-Schniter-Heath few-bit baseline (EM-GM-GAMP / EM-BG-VAMP; `mo2018.md`). We reuse its output-channel view; our quantiser likelihood is the same object (log C1).
- **Baseline status.**
  - A GAMP/VAMP few-bit estimator **must** appear as a baseline in the ADC study: reviewers in this area expect it.
  - **It has not been run.** `mo2018.md` states that no EM-VAMP/GAMP result exists yet, so nothing may be claimed against it.
- **Fair comparison.**
  - Same banks (test seeds 12000-12999) and the same Pd/strict/RMSE metric with Hungarian matching.
  - The same L, given to the peak extraction.
  - The same known sigma^2 as quantiser ML, plus a ±5 dB mismatch arm, because both need sigma^2.
  - Use the variants pre-registered in `mo2018.md` section 8:
    - V0: faithful G = 1 grid. Its noise-free ceiling is only about 0.18.
    - V1: oversampled G = 4 / 8, with damping, reporting how often it diverges.
    - V2: V1 + one quantiser-ML refinement from the L peaks. This is the direct classical rival to "CNN seeds + quantiser ML".
  - Tune only on validation seeds 81,000,000+, and measure latency on 1 thread.
- **What it implies for the network.**
  - GAMP's posterior support map is the training-free counterpart of our CNN heat map.
  - The +0.032 hybrid gain at 1 bit / 15 dB counts as evidence for *learned* seeds only if CNN-seeded ML beats GAMP/VAMP-seeded ML with a CI that excludes 0.
  - An unrolled GAMP/VAMP (LAMP/LVAMP) is an alternative unfolding route. It is not tried here.

## (c)9. If a reviewer asks...

1. **"Why no comparison with GAMP-based few-bit estimation (Mo et al.)?"** It is planned, with pre-registered variants, but not yet run. Until it is, we make no claim of beating it.
2. **"The exact quantised likelihood is not your contribution."** Agreed. It belongs to the GAMP framework and to Mo et al.; we credit it. Our claims concern seeding quality, speed (the hybrid at about 1/5 of the cost of multi-start ML) and the sigma^2-free read-out.
3. **"GAMP has rigorous guarantees and your network does not."** GAMP's state evolution is proved (as a sketched claim) only for large i.i.d. Gaussian A, not for our DFT-structured beamspace. Neither method has a guarantee in our setting, so we rely on paired bootstrap CIs on fresh seeds.

## (c)10. Glossary

- **Loopy belief propagation (BP):** passing probability messages on a graph with cycles.
- **Max-sum / sum-product:** the BP variants that compute MAP and MMSE estimates.
- **AMP:** approximate message passing, the Gaussian/quadratic approximation of BP for dense matrices. **GAMP:** AMP with an arbitrary output channel.
- **Onsager term:** the correction -tau^p s_hat(t-1) that removes self-feedback.
- **g_in / g_out:** the scalar denoising functions at the inputs and outputs.
- **State evolution (SE):** scalar recursions that predict the per-iteration error.
- **Scalar equivalent model:** each coefficient seen as truth plus Gaussian noise.
- **Replica method:** a statistical-physics prediction of the optimal error.
- **Bernoulli-Gaussian prior:** zero with probability 1 - rho, Gaussian otherwise.
- **Pseudo-Lipschitz:** a growth condition needed by the proofs.
- **EM:** expectation-maximisation, a way to learn prior parameters from data.

---

# (d) Algorithm Unrolling (Monga, Li, Eldar, IEEE SPM 2021)

## (d)1. Plain-English summary

Many signal-processing algorithms repeat the same step again and again: a matrix multiply followed by a simple nonlinearity. Write out a fixed number of those repetitions, one after another, and you have something that looks exactly like a neural network.

"Unrolling" does this on purpose. Each iteration becomes a layer, and the step sizes, thresholds and even matrices become trainable weights, learned from examples. The resulting network:
- is much smaller than a generic one;
- needs less training data;
- can be read like the algorithm it came from;
- often runs faster than the original algorithm, because a few trained layers replace many iterations.

This review explains the recipe, starting from the first example (learned ISTA for sparse coding). It surveys uses in imaging, medicine, vision, speech, power grids and communications, connects unrolling to Kalman filters, differential equations and statistical inference, and lists open problems, chiefly how to train such networks well.

## (d)2. Problem, system model, and symbol mapping

This is a review with no single system model. Its generic template is z_{l+1} = h(z_l; theta_l) for l = 0..L-1 (Fig. 4, p. 6). The canonical instance is ISTA for min 0.5 ||y - W x||^2 + lambda ||x||_1 (eqs. 2-3, p. 5).

| theirs | ours |
|---|---|
| y (input) | Y (16x16, possibly b-bit quantised; side input bits b, SNR) |
| h(.; theta_l), one iteration | one NOMP detection, cyclic-refinement round, or Levenberg-Marquardt (LM) polish layer of quantiser ML |
| theta_l (learned per layer) | learned step sizes, gates, damping factors, hypernet outputs (unfolded-v2: 12 LM layers with a learned per-layer factor; deviations.txt item 3) |
| L (number of layers) | **not our L** (our L = number of paths). Our unrolled depth: 3 cyclic rounds / 12 LM layers |
| x^L (output) | path angles (u, v), hence (psi, phi) |
| training pairs (y_n, x*_n) | simulated scenes with true angles (training seeds 80,000,000+) |
| W_t, W_e (LISTA matrices) | no direct analogue: our linear operators are fixed by physics (atoms, LS gains) |
| A, b, SNR | appear only as side inputs |

## (d)3. The method step by step

- **Pick an iterative algorithm** whose single step is cheap and, if possible, smooth (Sec. V-B, p. 23 warns about hard thresholds).
- **Map each iteration to a layer** and stack a finite number L of them (Fig. 4).
- **Decide what to learn.** Options are step sizes, thresholds, filters or whole matrices. Parameters can be shared across layers (RNN-like) or layer-specific (more expressive, but the original algorithm's convergence guarantees may be lost; Sec. V-B, p. 22).
- **Train end to end** on (input, ground-truth) pairs by back-propagation, with a task loss such as MSE (LISTA box, eq. 5, p. 5).
- **Variant: replace one step** of the algorithm with a full neural network ("drop-in", e.g. a denoiser or a likelihood block), without unrolling the rest (Sec. V-C, Fig. 16, p. 23).

```
# generic recipe (Fig. 4) instantiated for an unfolded NOMP decoder
layers = [h_1, ..., h_T]            # h_t = one cyclic round: for each path j: Newton/LM step on (u_j, v_j) against y - others(LS gains)
params = {step_scale_t, damping_t, gate_t}  (per layer, initialised to the classical values)
forward(y): (u, v) = seeds(y);  for t: (u, v) = h_t(y, u, v; params_t);  return (u, v)
loss = mean over scenes of Hungarian-matched angle error (or a differentiable hit surrogate)
train with SGD/Adam; validate against the classical configuration (params at their init) on held-out seeds
```

## (d)4. Assumptions

- **Training data.** Paired training data must exist (ours is simulated, so it does).
- **Value of the base algorithm.** It must already encode useful domain knowledge, the "high bias, low variance" model of Sec. V-A (p. 20).
- **Smooth layers.** Layers should be differentiable or smoothly approximable.
- **Not covered.** There is no assumption about noise, grids, snapshots or quantisation; these come from the base algorithm.

## (d)5. Reported results

All numbers below are **reported by the review from the cited works**. I did not open those works, so they are UNVERIFIED at source.

| result | as printed | where |
|---|---|---|
| LISTA | reaches a given accuracy about 20x faster than accelerated ISTA | Sec. II-B, p. 4; also III-E, p. 14, where it is attributed to [18], which looks like a citation slip for [13] |
| ADMM-CSNet | same accuracy with 10% less sampled data; about 40x faster; about 3 dB PSNR over state-of-the-art networks at 20% sampling | Sec. III-B, p. 9 |
| SCN | 0.3-1.6 dB PSNR gain | Sec. III-A, p. 9 |
| DetNet (MIMO detection) | similar to a semidefinite-relaxation detector, more than 30x faster, no noise-variance knowledge needed | Sec. V-B, p. 22 |
| Table II run time (s) / parameter count | ADMM-CSNet 2.61 / 7.8e4 vs D-AMP 12.59 vs ReconNet 2.83 / 3.2e5; CORONA 5.07 / 1.8e3 vs FISTA 15.33 vs ResNet 5.36 / 8.3e3; DUBLID 1.47 / 2.3e4 vs Perrone et al. 1462.90 vs DeblurGAN 10.29 / 1.2e7 | Table II, p. 18 |

**Table II ratios (log D1).** Iterative-to-unrolled time is 4.8x / 3.0x / 995x for the three rows. Generic-to-unrolled time is 1.08x / 1.06x / 7.0x, and generic-to-unrolled parameters 4.1x / 4.6x / 522x.

**The text and Table II disagree** in three places (Sec. III-E, p. 14):

| text claim | Table II gives |
|---|---|
| CORONA is more than 50x faster than FISTA | 3.0x |
| DUBLID is about 2x faster than DeblurGAN | 7.0x (the text also cites [33]) |
| ADMM-CSNet is about twice as fast as ReconNet | 1.08x |

Possibly different settings; cite Table II values only, and with care.

## (d)6. Citable claims (location; use)

1. **Definition.** Unrolling maps each iteration to one network layer, turns the algorithm's parameters into learnable weights, and trains the stack end to end, so the trained network reads as an optimised algorithm (Sec. I, p. 2; Fig. 1, p. 3). *Use:* defines what "unfolded from NOMP" means.
2. **Expected benefits.** Unrolled networks typically have far fewer parameters, need less training data, inherit domain structure and tend to generalise better than generic networks (Sec. I, p. 2; Sec. II-C, p. 4). *Use:* motivation. These are the review's general claims, not guarantees.
3. **Speed, not accuracy, is the established gain of LISTA.** The review says LISTA "does not exhibit superior performance over ISTA, not even under particular scenarios" (LISTA box, p. 21), while noting its empirically higher efficiency (Sec. IV-E, p. 20). *Use:* realistic expectation; supports our null result.
4. **Learned and analytic weights can be equivalent.** Theory shows that LISTA's learned weights asymptotically obey a coupling, and analytically chosen weights can match trained ones (Sec. IV-E box, p. 21, reporting ALISTA). *Use:* parallels our finding that the learned step correlates 0.91-0.92 with the Newton step.
5. **Layer-specific parameters have a cost.** They depart from the original algorithm and may lose its convergence guarantees; shared parameters make the network RNN-like and prone to exploding or vanishing gradients (Sec. V-B, p. 22). *Use:* describes our design choice and its risk.
6. **Unfolding is not unique.** Which algorithm to unroll and which parameters to train are open choices without a formal theory (Sec. VI, p. 24). *Use:* justifies reporting ablations and naming the parent algorithm explicitly.
7. **Training unrolled networks is hard.** Initialisation, vanishing and exploding gradients are named open challenges, and layer-wise pre-training is common (Sec. VI, p. 24). *Use:* frames our zero-gradient saddle and gradient spikes as known failure modes.
8. **The drop-in alternative is a separate category.** A conventional network can replace one step of an iterative algorithm (Sec. V-C, Fig. 16, p. 23). *Use:* our best ADC system, "CNN seeds -> classical quantiser ML", belongs here, not to unrolling.

## (d)7. Limitations

- **No experiments of its own.** All performance numbers are second-hand, and three text claims contradict its own Table II (section (d)5).
- **Qualitative scores.** Table III (p. 22) scores performance, efficiency, parameter count, interpretability and generalisability as High/Middle/Low without measurements.
- **No failure criterion.** There is no guidance on when unrolling will fail, e.g. when the base algorithm is already near-optimal.
- **Thin coverage of our area.** Communications coverage is thin: DetNet plus two cited surveys. There is nothing on quantised or ADC-limited estimation or on gridless spectral estimation.

## (d)8. Relevance to us

- **The principle we use.** Our learned decoders are "unfolded from NOMP":
  - `lurd.pt`, 4,803 parameters (`RESULTS_SUMMARY.md` section 5);
  - unfolded-v2 A and B, 910,515 and 506,800 parameters (deviations.txt item 1).
- **Verified outcome so far: null on validation.**
  - In S1, the front end plateaued (validation 0.6827 -> 0.6854; `train_S1.log`).
  - In S2B, training the hypernet and fusion made validation worse (0.6864 at step 0 -> 0.6788; deviations.txt item 10). The step-0 checkpoint was kept.
  - S2A (step MLP + selector + stop head on top of B) reached a best validation of 0.6851, still below B's 0.6864 (`train_S2A.log`).
  - Gradient-norm spikes in S2A reached 1.1e16 at step 1950 (`train_S2A.log`).
  - A zero-gradient saddle (zero-initialised MLP x zero gate) was found and fixed (deviations.txt item 10).
  - These are exactly the training pathologies the review lists (claims 5 and 7).
  - S3 and the test evaluation were never run. No test-bank claim exists.
- **Why the null result is unsurprising.**
  - The review's Fig. 15 argument (Sec. V-A, pp. 20-21) says iterative algorithms have high bias and low variance, generic networks the reverse, and unrolled networks sit between them with low bias and variance together.
  - On ideal hardware, NOMP has almost no legal headroom: a truth-based candidate oracle gives +0.0287, but only +0.0028 of it comes from candidates with a lower residual than NOMP's, i.e. from candidates a rule could find without the truth (`unfolded_v2/diag/REPORT.md`).
  - Under ADC, the candidate pool has an oracle pick +0.036 to +0.060* above 13-start ML, but lowest-NLL selection only reaches about ML's level. The headroom is in *selection*, which a learned selector failed to capture.
- **Honest framing for the paper.**
  - The verified win (+0.032* at 1 bit / 15 dB) comes from CNN seeds followed by an *unchanged* classical quantiser ML. That is the review's "drop-in" category (claim 8), not unrolling.
  - Describe `lurd` as "unfolded from NOMP", with the fixed classical components listed. Never call it "fully neural".
- **Design implications.**
  1. Initialise every learned parameter at its classical value. Validate against the classical configuration of the same network (design B at step 0 is that reference).
  2. Prefer few, shared, smooth parameters; clip gradients per module; freeze the front end, since front-end drift hurt (deviations.txt item 8).
  3. Spend learning where the classical method lacks information (seeding, sigma^2-free operation, selection under quantisation), not on re-learning Newton steps.

## (d)9. If a reviewer asks...

1. **"Is your decoder really an unrolled network, or a CNN bolted onto NOMP?"**
   - Both exist, and we name them separately.
   - `lurd` is unfolded from NOMP. The heat-map seeding, LS gains, residual subtraction and self-calibration gate are classical; one shared MLP replaces the Newton step.
   - The best ADC system is a hybrid: CNN seeds + classical quantiser ML, i.e. a network replacing the initialisation step.
2. **"Unrolling is supposed to beat its parent algorithm. Yours does not. Why?"** On validation, the learned components did not improve over their classical initialisation, and we report that. On ideal hardware, NOMP leaves almost no detectable headroom. The review itself notes that LISTA's established gain is speed, not accuracy.
3. **"Where are the promised interpretability and parameter savings?"**
   - `lurd` (4.8k parameters) is small and interpretable: its step correlates 0.91-0.92 with Newton.
   - Unfolded-v2 (0.5-0.9 M parameters) is not parameter-efficient, and we do not claim it is.
   - Its latency is also higher: the learned decoder takes about 98-101 ms, vs 4.8-6.9 ms for NOMP (`RESULTS_SUMMARY.md` section 5).

## (d)10. Glossary

- **Unrolling / unfolding:** turning T iterations of an algorithm into T network layers with trainable parameters.
- **ISTA:** iterative shrinkage-thresholding, a gradient step followed by soft thresholding. **LISTA:** learned ISTA.
- **Soft thresholding:** shrink values toward zero by lambda, and set small ones to zero.
- **Weight coupling:** a theoretical constraint linking LISTA's two matrices.
- **Layer-specific vs shared parameters:** different weights per layer vs the same weights in every layer (RNN-like).
- **End-to-end training:** fitting all layers jointly to the final loss.
- **Drop-in / plug-and-play:** replacing one algorithm step with a trained network.
- **Bias-variance view:** iterative algorithms are rigid (high bias, low variance), generic networks are flexible (low bias, high variance), and unrolled networks sit in between.
- **Greedy layer-wise pre-training:** training layers one at a time before joint training.

---

# Cross-paper summary: what to write, and what not to

| topic | may say (with source) | must not say |
|---|---|---|
| classical decoder | NOMP (Mamandipoor et al. 2016) with 2-D atoms, gamma = 4, cyclic refinement; strongest ideal-hardware baseline (0.7581) | that our network beats NOMP on ideal hardware; that NOMP is slow (paper (a) Table III) |
| unknown L | NOMP + CFAR (paper (a)) and blind CFAR / MDL results from M7; NOMP-CFAR (paper (b)) as the reference blind method | any unknown-L result under ADC quantisation; any claim against NOMP-CFAR (not run) |
| quantised likelihood | exact cell likelihood as in the GAMP framework (paper (c)) and Mo et al.; it is prior art | that the quantiser likelihood is our contribution; any comparison with GAMP/VAMP before it is run |
| learned decoder | "unfolded from NOMP" (paper (d) definition); validation null result; hybrid = drop-in category | "fully neural"; "unrolling improves accuracy" |

Files: `foundations_checks.py` / `foundations_checks.log` (this folder). Paper copies: `C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/foundations/`.

---

# Corrections made on the resume pass (2026-10-16)

- **Bibliography.** Crossref and arXiv fields are confirmed. The earlier claim of an Xplore read could not be reproduced (the fetch returned an empty page), so Xplore-only fields are UNVERIFIED. For (b), volume, issue and pages match only the GNOMP paper's bibliography.
- **(a)1 summary.** NOMP is faster than all methods it was compared with except DOMP, which is faster but much less accurate (Table III, p. 12).
- **(a)5 and (a)6.** The cyclic-refinement sentence is in Sec. II, Steps 7-9 (p. 4); there is no Sec. II-B. The oversampling quote is from Sec. IV, p. 7. The Scenario 2 accuracy statement is not in the text and was removed.
- **(b)4, (b)6, (b)7.** Real-data K_max is 32, not 32-48. The paper's wording on the known-noise assumption is "unknown and can often be variable" (not "unrealistic"). FFT-CFAR is excluded because it performs poorly with many targets.
- **(b)8.** The M7 values are original-seed values; fresh-seed values from VERDICT section 4 are added, along with the re-derived kappa (12.27 vs 12.32).
- **(c)5.** Algorithm 1 vs Algorithm 2: "virtually no difference" in unreported simulations (Sec. VII, p. 12).
- **(d)6 and (d)8.** LISTA is quoted directly (LISTA box, p. 21). The Fig. 15 argument is restated as a bias-variance argument, not as "underfitting".
- **Re-checked, no change needed.** NOMP Tables I-IV (including the Table III run times and the oversampling timings), eqs. 12-15, Remarks 1-2, Sec. VII eq. 32, the Fig. 5 convergence claim, the 0.25-bin detection radius; the CFAR multipliers (11.22/11.25/10.15, 2.81/1.67), the Sec. IV-C numbers (14/15/17 dB, 1 dB vs 2 dB), eqs. 29/19, the 0.5-bin false-alarm definition, Remark 1; GAMP eqs. 5-8, 36-38, 42-43, the Fig. 4 settings (rho=0.1, a=6.1, tau_w=0.01, 100 trials, 11 dB, 0.2/0.4 dB), the "10 to 20 iterations" statement and the Conclusions; Monga Table II and its three text-versus-table conflicts, Table III, the DetNet 30x, the LISTA 20x, the CORONA 50x, the ADMM-CSNet figures, the bias-variance and drop-in passages, and the open problems.
- **Project numbers (a)8, (b)8, (d)8.** Checked against `RESULTS_SUMMARY.md`, `stress/RESULTS.md`, `stress/adversarial/VERDICT.md`, `stress/check_sim.log`, `unfolded_v2/build/deviations.txt`, `train_S2A.log` (including the 1.1e16 gradient-norm spike at step 1950), and `unfolded_v2/diag/REPORT.md`. All match.
- **Not re-checked.** Figure-only statements (Fig. 2 false-alarm tracking, the Figs. 6-11 accuracy ordering beyond the text), the CFAR real-data counts beyond the lines cited, and the (b) equation numbers beyond those cited in the log.

