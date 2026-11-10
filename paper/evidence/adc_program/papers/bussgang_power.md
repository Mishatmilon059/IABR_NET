# Bussgang linearisation of one-bit ADCs, BLMMSE estimation, and ADC power vs bits

Scope: Li et al. (TSP 2017, arXiv 1609.07427) read in full (arXiv v3 text). Walden (JSAC 1999) read from the cached full text and page images. Mo & Heath (TSP 2015, arXiv 1410.7353) read in full; appendix A (proof of bound (57)) skimmed, not re-derived. Numbers marked [run] come from scripts in this folder with logs saved beside them. Anything not checked is marked UNVERIFIED. Page numbers are the paper's own printed/PDF page numbers as read from the cached text (Li: PDF pages; Mo: PDF pages; Walden: JSAC page where printed, else PDF page).

Cached copies (not republished): C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/bussgang/ (li2017.txt/.pdf, mo.txt/.pdf, walden_pdf/walden1999.pdf, walden_pdf/walden1999.txt, walden_pdf/p3_1.png, p6_1.png, li_p9_0.png, cr_*.json = Crossref records).

## 0. Verified bibliographic entries

(a) Y. Li, C. Tao, G. Seco-Granados, A. Mezghani, A. L. Swindlehurst, L. Liu, "Channel estimation and performance analysis of one-bit massive MIMO systems," IEEE Trans. Signal Process., vol. 65, no. 15, pp. 4075-4089, Aug. 2017, doi 10.1109/TSP.2017.2706179; arXiv:1609.07427 (cached text = v3, 21 Mar 2017). Authors, journal, volume, issue, pages and DOI confirmed by Crossref (cr_10.1109_TSP.2017.2706179.json). UNVERIFIED: that the TSP text equals arXiv v3 (only v3 was read).

(b) R. H. Walden, "Analog-to-digital converter survey and analysis," IEEE J. Sel. Areas Commun., vol. 17, no. 4, pp. 539-550, Apr. 1999, doi 10.1109/49.761034. Confirmed by Crossref and the cached full text (page header 539). Abstract and eq. (4) verified from the text and page image (section 5c).

(c) J. Mo, R. W. Heath Jr., "Capacity analysis of one-bit quantized MIMO systems with transmitter channel state information," IEEE Trans. Signal Process., vol. 63, no. 20, pp. 5498-5512, Oct. 2015, doi 10.1109/TSP.2015.2455527; arXiv:1410.7353 (cached v3, 6 Nov 2015). Venue, volume, issue and pages confirmed by Crossref. Li et al. cite it as the same TSP paper (their ref. [17]).

## 1. One-paragraph plain-English summary

A normal receiver turns an incoming radio signal into numbers with many bits per sample. A one-bit receiver keeps only the sign of the real and imaginary parts. That is very cheap and power-efficient, but it throws away most of the amplitude information. Li et al. observe that a one-bit quantiser behaves, on average, like a scaled copy of the signal plus an extra error term that is uncorrelated with the signal (the Bussgang decomposition). Because the error is uncorrelated with the signal, ordinary linear estimators (LMMSE) can be applied to the quantised output to get a channel estimate; they call this BLMMSE. They then analyse achievable rates and conclude that a one-bit massive MIMO system needs about 2.2-2.3 times more antennas than an unquantised one for the same sum rate. The catch, which matters for us, is that the error term is uncorrelated but not independent of the signal, and the linear recipe rests on a Gaussian assumption. Our own runs show that this recipe, applied to sparse beamspace data, makes NOMP worse, not better.

## 2. Problem and system model; symbol mapping

Problem (Li Sec. II, p. 3): single-cell uplink, M receive antennas each with two one-bit ADCs (sign of Re and Im), K single-antenna users, Rayleigh channel H (M x K). The received signal is y = sqrt(rho) H s + n (eq. 1), quantised as r = Q(y) with Q(.) = (sign Re + j sign Im)/sqrt(2) (eq. 2). Training: Y_p = sqrt(rho_p) H Phi^T + N_p (eq. 3); vectorised y_p = Phi_bar h + n_p with Phi_bar = Phi (x) sqrt(rho_p) I_M (eq. 4); r_p = Q(y_p) (eq. 5).

Mo & Heath (Sec. II, p. 3): point-to-point Nr x Nt MIMO, y = Hx + n (eq. 1), r = sgn(Hx + n) componentwise (eq. 2), perfect CSIT and CSIR. mmWave model with L paths (Sec. VII, eq. 59): H = sum_l alpha_l a_r(phi_rl, theta_rl) a_t^*(phi_tl, theta_tl).

Symbol mapping (our side: D:/ai_ml_project/paper/evidence/deep_research/stress/stress_sim.py and unfolded_v2/build/ucore.py):

| Our symbol | Meaning in our project | Li et al. (2017) | Mo & Heath (2015) | Note |
|---|---|---|---|---|
| Y | 16x16 complex beamspace measurement, Y = W^H H F + Z | y_p = vec(Y_p) (eq. 4) | y (eq. 1) | Li measures all antennas directly; we measure through beams |
| W | 16-point DFT receive codebook | no analogue: Phi_bar uses I_M | n/a | analog combiner absent in Li |
| F | 16-point DFT transmit/probing beams | pilot matrix Phi (tau x K, eq. 3) | n/a | our (F^T (x) W^H) plays the role of Phi_bar |
| H | 16x16 antenna-domain channel | H (M x K) (eq. 1) | H (Nr x Nt) | Li's h = vec(H) (eq. 4) |
| L | 3 paths (sparse, specular) | none: i.i.d. Rayleigh, no paths | number of paths (Sec. VII) | Li does not model sparsity |
| b | ADC bits per Re and Im | one-bit only | one-bit only | our stress_sim.adc: b = 1..4 |
| A | per-scene AGC full scale = max(abs Re, abs Im) over the 256 beams | "no AGC" (p. 1) | AGC "may not be required" (p. 2) | not the same A as Li's A_p |
| A_p (Li) | Bussgang gain sqrt(2/pi) diag(C_y)^(-1/2) (eq. 7, 10) | - | - | our decoders are scale-invariant |
| SNR | per-beam SNR, -10..25 dB | rho_p, rho_d (eq. 1, 3) | not used for estimation | exact definition: as in stress_sim |
| u, v | per-path continuous atom parameters in unfolded_v2 (ucore.py line 138: angle in degrees = arccos of cos) | none (no angles) | angles phi, theta (eq. 58-59) | neither paper estimates angles |
| r | quantised observation | r_p (eq. 5) | r (eq. 2) | - |
| q | quantiser distortion, q = r - A y (Bussgang) | q_p (eq. 6) | none | - |
| h_hat / g_hat | our de-quantised beamspace estimate | h_hat_BLM (eq. 12, 14) | none | - |
| Pd | paper-style: fraction of angle components within 1 deg after Hungarian matching | MSE (eq. 17), per element | capacity (bit/s/Hz) | no Pd analogue in either paper |

Note on pilots: Li's DFT pilot choice takes "K columns of the tau x tau DFT matrix" (Sec. III-A, p. 4), but does not say which columns. This matters (section 5d, row F1).

## 3. The method, step by step

Core idea (Bussgang decomposition, Li Sec. III-A, eq. 6-8, p. 3): for a Gaussian input y, a memoryless nonlinearity can be written as

  r = A y + q,   with A chosen so that E[q y^H] = 0  (eq. 6-7)

so A = C_yr^H C_y^{-1}. For one-bit quantisation C_yr = sqrt(2/pi) C_y diag(C_y)^(-1/2) (eq. 8), giving A = sqrt(2/pi) diag(C_y)^(-1/2) (eq. 10). Plain English: the quantiser acts as a linear gain of about 0.8 plus a distortion that is "orthogonal" to the input. The distortion has power 1 - 2/pi = 0.363 per component at high SNR (eq. 32 approximation). Scalar check [run, bussgang_power_check.log, A1]: y ~ CN(0,1), A = 0.7976 (sqrt(2/pi) = 0.7979), E|q|^2 = 0.3632 (1-2/pi = 0.3634), correlation of q with y = 0.0000.

The step that breaks least squares: uncorrelated is not independent. Same run, A1: E|q|^2 given |y| < 0.3 is 0.7432, but given |y| > 1.5 it is 0.5059. The distortion depends on the signal amplitude, so any estimator that treats it as independent additive noise is wrong.

Algorithm, Li's BLMMSE (Sec. III-A, eq. 12-14, p. 4):
1. Form the pilot covariance C_y = Phi_bar Phi_bar^H + sigma^2 I (Gaussian model) and Sigma = diag(C_y).
2. Gain: A_p = sqrt(2/pi) Sigma^(-1/2) (eq. 10). For DFT pilots with entries of magnitude one, A_p = sqrt(2/pi) (K rho_p + 1)^(-1/2) I (eq. 11; the square root is lost in the PDF text and was reconstructed from eq. 11 and checked by [run] A2).
3. Output covariance via the arcsine law (eq. 13): C_r = (2/pi) [ arcsin(X) + j arcsin(Y) ], X = Sigma^(-1/2) Re(C_y) Sigma^(-1/2), Y = Sigma^(-1/2) Im(C_y) Sigma^(-1/2).
4. Estimate (eq. 12, 14): h_hat = C_h Phi_tilde^H C_r^{-1} r, Phi_tilde = A_p Phi_bar. Li simplify to Phi_tilde^H C_r^{-1} r with C_h = I, using that the quantiser noise is uncorrelated with h (Appendix A).
5. Error (eq. 17) for tau = K, DFT pilots: M_BLM = 1 - 2 K rho / (pi (K rho + 1)); error floor 1 - 2/pi = -4.40 dB as rho -> infinity (eq. 18).

Low-SNR approximation (Li Sec. III-C, eq. 31-33, p. 5): when C_y is diagonally dominant, C_qp is approximately (1-2/pi) I, so the quantiser looks like uncorrelated additive noise. This is the approximation of the earlier work [28] (Mollen et al., not opened) and of this paper's rate analysis.

Mo & Heath's relevant method (Sec. VI, p. 9): input design by convex optimisation. For each possible quantised output r, solve max d s.t. diag(r) Re/Im(H x) >= d (eq. 54-55), keep the nonzero solutions. This is capacity-oriented; it does not estimate a channel.

Pseudo-code to re-implement the BLMMSE + NOMP baseline (matches bussgang_power_check.py, B2):

    input:  Y_q (16x16 complex, quantised by stress_sim.adc with b bits), C_g (prior covariance of
            vec(beamspace), from prior banks, seeds 80,000,000+), sigma2 (from SNR, as in stress_sim)
    1. scale: decoder is scale-invariant, so work with unit-power beams (divide by sqrt(mean |Y|^2)).
    2. analytic variant (1 bit only):
       C_y = C_g + sigma2 * I;  Sigma = diag(C_y);  A = sqrt(2/pi) * Sigma^(-1/2)
       X = Sigma^(-1/2) Re(C_y) Sigma^(-1/2);  Y = Sigma^(-1/2) Im(C_y) Sigma^(-1/2)
       C_r = (2/pi) * (arcsin(X) + 1j*arcsin(Y))               # arcsine law, Li eq. 13
       g_hat = C_g A^H C_r^{-1} r                               # BLMMSE with Phi_tilde = A, Li eq. 12
    2b. empirical variant (any b): C_gr and C_r by Monte-Carlo on prior banks; g_hat = C_gr C_r^{-1} r
    3. reshape g_hat to 16x16; run NOMP: common.nomp_greedy(G_hat, L=3, 4, 3)
    4. score: paper-style Pd and RMSE on the test bank (12000-12999), paired bootstrap 2000, seed 0

Control: a white prior (C_g = I) reduces the LMMSE to a scalar gain. NOMP then gives identical predictions (identical fraction 1.000 in [run] B2), so a global scale does not change NOMP.

## 4. Assumptions (what the method needs)

- Known SNR / noise variance: yes; rho_d and rho_p are assumed known at the BS (Li Sec. II, p. 3).
- Number of paths or users: Li: K known (K <= tau); no path count (Rayleigh). Mo: L < min(Nt, Nr) (Sec. VII, p. 10). Our benchmark gives L = 3 to every decoder (true L), as in finetune/adversarial/s3_adc_test.py.
- Dithering: none. Neither Li nor Mo discusses dither. UNVERIFIED that none was used in their simulations, but the text describes none.
- AGC: none for one bit. Li p. 1: "One-bit ADCs do not require automatic gain control and linear amplifiers." Mo p. 2: AGC "may not be required." Our b-bit model does use peak AGC (stress_sim.adc), but for b = 1 the output is sign(x) A/2, which is scale-free.
- Snapshots: tau pilot symbols per block (one per column); coherence interval T = 200 symbols (Li Sec. VI, p. 8).
- Grid: none. Li: continuous Gaussian channel. Mo: continuous angles, uniform on [0, 2 pi] and [-pi/2, pi/2] (Sec. VIII-D, p. 12).
- Gaussian input: required. Li's A_p, C_r and error formulas assume y Gaussian (Sec. III-A). For rate bounds, quantiser noise is treated as worst-case Gaussian (p. 6, citing [41], [42]).
- Equal power / power control: all users have the same large-scale SNR (Li p. 3), "some type of power control".
- DFT pilots with equal-magnitude entries (Li p. 4): diagonal of Phi Phi^H equals K.
- Per-realisation Bussgang for the data phase (Li Sec. IV-A, p. 5): A_d = sqrt(2/pi) diag(rho_d H H^H + I)^(-1/2), i.e. a different gain per channel draw.
- Low SNR / diagonal dominance: needed for the uncorrelated-noise approximation (eq. 31-32).
- Mo & Heath: perfect CSIT and CSIR; narrowband; no estimation error (Sec. IX, p. 13).

Our benchmark violates the Gaussian input assumption: the sparse L = 3 beamspace is not Gaussian (section 5d, row G1).

## 5. Reported results (exact numbers, figure/table, conditions)

### 5a. Li et al. (arXiv v3)

- Eq. (17) (Sec. III-A, p. 4): M_BLM = 1 - 2 K rho/(pi (K rho + 1)), tau = K, DFT pilots. Reproduced by Monte-Carlo [run] A2 (M = 16, K = 4, tau = 4, 2000 channels): -20 dB -0.11 (MC) vs -0.11 (eq.); -10 dB -0.87 vs -0.87; 0 dB -3.10 vs -3.09; 10 dB -4.20 vs -4.21; 20 dB -4.38 vs -4.38; 40 dB -4.41 vs -4.40.
- Eq. (18) (p. 4): error floor "1 - 2/pi = -4.40 dB" as rho -> infinity (tau = K).
- Fig. 2 (p. 9): NMSE vs SNR, M = 16, K = 4, tau = 20; LS of [21], nML of [25], BLMMSE, additive-noise model of [28]. Text (p. 9): "our proposed BLMMSE approach outperforms the other previously proposed approaches." Curve values below are read off the figure (cached PNG of p. 9; reading error about +/-0.2 dB): BLMMSE about -0.5 dB at -20 dB, -3.2 at -10 dB, -6.9 at 0 dB, about -7.1 at 10-20 dB. Our reproduction [run] A3: -0.50, -3.25, -7.22, -7.95, -7.94 dB at -20, -10, 0, 10, 20 dB. The low-SNR points agree; the 10-20 dB floor is about 0.8 dB lower than the plot. Cause not established (UNVERIFIED); see row F1.
- Fig. 3 (p. 9): spatially correlated channel, M = 16, K = 1, tau = 2; text: the gap "grows to over 1 dB" (p. 9).
- Fig. 4 (p. 9): sum SE vs SNR, K = tau = 8, M in {32, 64, 128}; text: at M = 128, SNR = -10 dB, the gap between approximation and lower bound is 0.19 (MRC) and 0.38 (ZF) bit/s/Hz.
- Sec. V-A (p. 7), eq. (51), Fig. 5 (p. 10): with fixed training power, scaling user power by 1/M keeps the sum SE constant as M grows.
- Sec. V-B, Lemma 2 (p. 8): the optimal training length is not always K.
- Fig. 6 (p. 10): bit energy; text: optimal allocation reduces bit energy by a factor 1.9 at 15 bit/s/Hz for M = 128; doubling M from 128 to 256 gives a factor about 2.2.
- Fig. 8 (p. 11): one-bit vs conventional, rho = -10 dB. Text: at M = 400, one-bit gives 23.2 (MRC) and 24.6 (ZF) bit/s/Hz, which is 73.68% and 69.76% of conventional.
- Fig. 9 (p. 11): ratio kappa = M_one / M_conv; text: 2.5 without optimal allocation (MRC), 2.2-2.3 with it. The 2.5 figure is attributed to [28]; not re-derived here.
- Conclusions (p. 11-12): "approximately 2.2-2.3 times more antennas are required."

### 5b. Mo & Heath (arXiv v3; page numbers from the PDF)

- Lemma 1 (p. 3, eq. 5): SISO one-bit capacity C = 2 [1 - H_b(Q(|h| sqrt(P_t)))], achieved by rotated QPSK.
- Proposition 1 (p. 3, eq. 8): MISO one-bit capacity 2 [1 - H_b(Q(||h|| sqrt(P_t)))].
- Low-SNR loss (p. 4, eq. 10): one-bit gives "10 log10(pi/2) = 1.96 dB power loss" (MISO, CSIT).
- Proposition 2 (p. 4, eq. 12): log2(4 Nr) <= C_SIMO(infinite SNR) <= log2(4 Nr + 1).
- Proposition 3 (p. 6, eq. 20-22): MIMO bounds at infinite SNR, with K(Nr, Nt) of eq. 21.
- Proposition 4 (p. 7, eq. 27): finite-SNR upper bound 2 Nr [1 - H_b(Q(sqrt(P_t sigma_max^2 / Nr)))].
- Corollary 3 (p. 10, eq. 61): mmWave with L < min(Nr, Nt) paths: log2 K(Nr, L) <= C <= log2(K(Nr, L) + 1).
- Figs. 9-11 (p. 11): 2 x 2 converges to 4 bit/s/Hz; text: log2 K(3, 2) = log2 52 approx. 5.7 bit/s/Hz for 3 x 2.
- Fig. 15 (Sec. VIII-D, p. 12-13): 4 x 256 mmWave with L = 1, 2, 3 paths. Text: the rates "converge to log2 K(4, L) bps/Hz, which is 4, 7 and 7.9 bps/Hz when L = 1, 2, 3," and "only converge at very high SNR which is larger than 60 dB."

### 5c. Walden (JSAC 1999)

- Eq. (4) (Sec. II; PDF page 3 of the cached copy): F = 2^(SNRbits) f_samp / P_diss, verified in the page image.
- Sec. V (JSAC p. 544 header; PDF page 6): "Most of the ADC's have values of F <= 7.9 x 10^10 (the mean)." Two fictitious examples with that F: 13 SNR-bits, 10 Ms/s, 1.1 W; and 7 SNR-bits, 1 Gs/s, 1.75 W.
- Abstract (p. 539): resolution "falls off by 1 bit for every doubling of the sampling rate" between 2 Ms/s and 4 Gs/s.
- Abstract: "Average improvement is only 1.5 bits for any given sampling frequency over the last six-eight years."
- Sec. II: stated resolution vs SNR-bits degrade by about 1.5 bits at a given sampling rate.
- Arithmetic [run, bussgang_power_check.log, section C]: F(example 1) = 2^13 x 1e7 / 1.1 = 7.45e10; F(example 2) = 2^7 x 1e9 / 1.75 = 7.31e10. Both match the stated mean, so the examples use this F definition.

### 5d. Our own numbers (new runs; 16x16 DFT benchmark, L = 3 given, peak AGC, paper-style Pd)

Source: D:/ai_ml_project/paper/evidence/adc_program/papers/bussgang_power_check.py and .log (final run, 11:29); bussgang_power_followup.py and .log (11:35). Test seeds 12600 (0 dB) and 12615 (15 dB), 1000 scenes each. Paired bootstrap 2000, seed 0, within SNR groups.

| Cell | NOMP, raw quantised | empirical LMMSE + NOMP | diff vs NOMP [95% CI] | analytic Bussgang LMMSE + NOMP | diff [95% CI] |
|---|---|---|---|---|---|
| 1-bit, 0 dB | 0.1613 | 0.0533 | -0.1080 [-0.1188, -0.0975] | 0.0735 | -0.0878 [-0.0987, -0.0772] |
| 2-bit, 0 dB | 0.3728 | 0.1212 | -0.2517 [-0.2672, -0.2363] | not run | - |
| 1-bit, 15 dB | 0.1883 | 0.0992 | -0.0892 [-0.0995, -0.0793] | 0.1408 | -0.0475 [-0.0575, -0.0377] |
| 2-bit, 15 dB | 0.2780 | 0.1587 | -0.1193 [-0.1317, -0.1073] | not run | - |

All four differences are negative and the CIs exclude 0. The white-prior control (scalar gain only) gives identical predictions to raw NOMP (diff 0.0000, identical fraction 1.000). For context, the project's fresh-seed numbers (RESULTS_SUMMARY.md, lines 73-76; different banks, so compare only to about 0.01): NOMP 0.163 (1-bit, 0 dB) and 0.191 (1-bit, 15 dB); consistent NOMP 0.237 (1-bit, 15 dB); quantiser-ML 0.540 (1-bit, 15 dB); hybrid 0.572 (1-bit, 15 dB).

Diagnostics (same logs):
- G1 Gaussian-input check (1-bit, 0 dB, 8000 scenes): arcsine-law C_r vs empirical C_r, off-diagonal deviation in Monte-Carlo standard errors: real beamspace 1.10, Gaussian surrogate 1.00. Per-beam ratio E[g_i r_i^*] / (C_g A^H), median: real beamspace 0.377 (5-95%: 0.326-0.435), Gaussian surrogate 0.998. At 15 dB: 3.03 (real) vs 1.00 (surrogate); ratio 0.353 vs 1.000. The Gaussian assumption behind the Bussgang gain fails for our beamspace.
- NMSE of the de-quantised beamspace, 1-bit, 0 dB: empirical LMMSE -0.20 dB; analytic Bussgang +0.20 dB (worse than predicting zero); raw scalar LMMSE -0.19 dB. At 15 dB: empirical -0.43, analytic +0.42, raw -0.37. The analytic recipe is badly scaled for this data (cause UNVERIFIED; most likely the mismatch in G1).
- F2 (15 dB, seed 12615): LMMSE matrix diagonal |weight| max/min = 15.0 (1-bit) and 14.3 (2-bit); off-diagonal energy fraction 0.31 (1-bit), 0.18 (2-bit). The diagonal-only variant gives the same Pd as the full matrix (0.0970 vs 0.0992, 1-bit), so the off-diagonal part does not drive the loss.
- B3 (40 dB, seed 12640, 500 scenes): NOMP Pd for infinite / 4 / 2 / 1 bit: 0.9857 / 0.6633 / 0.2317 / 0.1800 (RMSE 0.058 / 0.477 / 0.553 / 0.575 deg). The error floor is set by the quantiser, not by noise.
- C (Walden arithmetic; not a claim of Walden): with constant F = 7.9e10 and f_s = 1 GS/s, P = 2^b f_s / F gives 0.025 W (1 bit), 3.2 W (8 bit), 52 W (12 bit). This extrapolates at fixed F and is not a statement in Walden; do not cite it as his.
- F1 (pilot columns, Fig. 2 setting, M = 16, K = 4, tau = 20, 2000 channels): BLMMSE NMSE at 20 dB depends on which DFT columns are used: adjacent (0,1,2,3) -7.99; equispaced (0,5,10,15) -4.68; mixed (0,3,7,12) -9.02; pairs (0,1,10,11) -7.44 dB. At 0 dB: -7.23, -6.14, -7.59, -7.04. Li's Fig. 2 floor (about -7.1 dB) lies inside this range, but the paper does not say which columns it used (UNVERIFIED which).

## 6. Citable claims (quotes at most 15 words)

1. Li et al., Abstract, p. 1: "The resulting channel estimator outperforms previously proposed approaches across all SNRs." Cite for: BLMMSE beats LS and nML in their Fig. 2 setting only; our runs do not confirm this for the sparse beamspace (section 5d).
2. Li et al., Sec. I, p. 1: "One-bit ADCs do not require automatic gain control and linear amplifiers." Cite for: why one-bit front ends are cheap, with the caveat that our b-bit model still uses AGC.
3. Li et al., Sec. III-A, p. 3: "The matrix Ap is chosen to make qp uncorrelated with yp." Cite for: the Bussgang definition; it says uncorrelated, not independent.
4. Li et al., Sec. III-A, p. 4 (eq. 18): "an error floor for the channel estimate as the training power increases" (12 words). Cite for: the one-bit BLMMSE error floor, tau = K only.
5. Li et al., Sec. I, p. 1: "grows exponentially with the number of quantization bits" (8 words). Cite as Li's claim only; Walden does not state it in these words (section 7, V1).
6. Li et al., Sec. VII, p. 11-12: "approximately 2.2-2.3 times more antennas are required." Cite for: their rate-model result under Rayleigh MRC/ZF; not a claim about estimation or angle recovery.
7. Mo & Heath, Sec. I, p. 1: "the power consumption grows exponentially with the resolution" (8 words), citing Walden [8]. Cite as their claim, with Walden as the source they name.
8. Mo & Heath, Sec. VIII-D, p. 13: "which is 4, 7 and 7.9 bps/Hz when L = 1, 2, 3" (11 words). Cite for: mmWave capacity limited by the number of paths (infinite-SNR regime, above 60 dB).
9. Walden, Abstract, p. 539: "resolution falls off by 1 bit for every doubling of the sampling rate" (13 words). Cite for: the speed-resolution trade-off; not power.
10. Walden, Sec. V, JSAC p. 544: "Most of the ADC's have values of F <= 7.9 x 10^10 (the mean)." Cite for: the typical power efficiency of the converters in his survey.

## 7. Limitations and what the papers do not show

- Li et al. do not evaluate angle or path recovery, sparsity, or any metric like Pd. Their only estimator figure is MSE vs SNR (Fig. 2) with M = 16, K = 4, tau = 20, and i.i.d. Rayleigh channels.
- Li et al. treat one-bit only for estimation. Their rate analysis uses MRC/ZF at low SNR and large M; the 2.2-2.3 antenna ratio is a rate result (UNVERIFIED for b > 1).
- The DFT column choice for pilots is not specified (F1 above), and it changes the high-SNR NMSE by about 4 dB. The Fig. 2 floor cannot be reproduced exactly without it.
- Li's eq. (11) square root is lost in the PDF text; reconstructed from eq. (10) and (17) and checked against our Monte-Carlo (A2).
- The text cached is arXiv v3, not the TSP version (UNVERIFIED differences).
- Mo & Heath do not estimate channels. Introduction, p. 2: "Incorporating estimation error and studying CSI feedback are left to future work." Their conclusion (Sec. IX, p. 13) defers estimation too. They use perfect CSI at both ends, and their rates are capacities, not estimator performance.
- Mo & Heath's mmWave results (Fig. 15) hold only above 60 dB SNR (text, p. 13); our benchmark uses -10..25 dB.
- Walden 1999 is a survey of converters reported in the literature. It reports figure of merit and resolution, not the measured power of a specific converter in our setting. Its data period was not verified, so present-day values are UNVERIFIED.
- Not checked: the TI product pages cited for "several watts" (Li [10], Mo [11]); Li's "a few milliwatts" for a one-bit comparator (uncited in the text); Mo's [39] (one-bit low power, T. Sun et al.). All UNVERIFIED.

Power-vs-bits statements, verified:
- V1: Walden, Sec. IV (flash architecture; PDF page 6): "The number of comparators grows exponentially with N." This is a comparator count, not a power law. Mo & Heath's claim that power grows exponentially with resolution follows from this comparator count, and from eq. (4) only at fixed F and f_s. Walden does not state a power law in words.
- V2: Walden's "1 bit per doubling of sampling rate" is about resolution vs sampling rate, not power.

## 8. Relevance to us

What to reuse:
- The Bussgang gain and distortion statistics (eq. 10, 13, 32) as a diagnostic: they predicted the 0.36 distortion power and the arcsine covariance (A1, A3). They are a correct check on our quantiser code.
- Scalar-gain invariance (B2 control): NOMP is invariant to a global gain, so at 1 bit no amplitude normalisation is needed by the decoder.
- The DFT-pilot gain (eq. 11) as a sanity check on beam codebook normalisation.

Must it be a baseline? Yes, but only as a named classical reference. Li et al. (2017) BLMMSE is the standard linear one-bit channel estimator, so a reviewer will ask. Fair setup: same quantiser (stress_sim.adc, b bits, peak AGC), same measurement operator (F, W DFT codebooks), same 1000-scene test banks (12000-12999), same paper-style Pd (1 deg, Hungarian), same true L = 3, same SNR grid, paired bootstrap 2000 (seed 0). Report both the analytic (1-bit only) and the empirical (any b) variant, and report that the empirical one is worse than raw NOMP. The stronger classical baselines are consistent NOMP and quantiser-ML (RESULTS_SUMMARY). Do not present BLMMSE as the best classical baseline.

What it implies for our network design:
- Do not feed the network a Bussgang-linearised estimate as its primary input. Our runs show the linearisation hurts NOMP (section 5d), and the Gaussian assumption fails (G1).
- If the network has a linear front end, initialise it as a scalar gain, not as BLMMSE, and let training learn the rest. A learned front end has to beat the scalar-gain baseline and raw NOMP, not BLMMSE alone.
- The distortion is signal-dependent (A1: its power depends on |y|), so a loss or prior that assumes independent Gaussian noise will mislead. Use the quantiser-ML likelihood (mlq.py) if a noise model is needed.
- At 1 bit the amplitude is lost by construction (sign only), so no decoder can recover it. This sets the ceiling on any 1-bit decoder. The 40 dB floor (B3) is a quantiser floor, not a noise floor.

## 9. "If a reviewer asks..."

Q1. "Why did you not show Li et al.'s BLMMSE beating NOMP, as their Fig. 2 claims?"
A: Their Fig. 2 is MSE vs SNR for a Rayleigh channel with 16 antennas, K = 4 users and tau = 20 pilots, with DFT columns the paper does not name. Our benchmark is a sparse 16x16 beamspace with L = 3 paths and a Pd metric. We implemented their recipe (B2), and it lowers Pd against raw NOMP (1-bit 0 dB: 0.161 raw, 0.073 analytic, 0.053 empirical; CIs exclude 0). We cannot tell whether the cause is the sparsity, the non-Gaussian statistics (G1), or the pilot setup. We report it as a negative result, not as a refutation of their paper.

Q2. "Li et al. claim one-bit massive MIMO needs only 2.2-2.3 times more antennas. Does this apply to our angle estimation?"
A: No. Their result is a rate bound for MRC/ZF with Rayleigh channels at low SNR and large M (Sec. V-C, p. 8-11). We do not compute rate, and Pd is not a rate. Mo & Heath's 1.96 dB low-SNR loss (eq. 10) is a capacity statement with CSIT, also not an estimator result. Neither paper gives a Pd or angle-error number.

Q3. "Is the power-grows-exponentially-with-bits claim sourced?"
A: Partly. Li cite Walden [9] and Mo & Heath cite Walden [8]. Walden's text supports exponential growth of the flash comparator count (2^N - 1, Sec. IV), and gives P = 2^b f_s / F (eq. 4), so power grows as 2^b only at fixed F and f_s. We do not measure ADC power. Cite it as motivation with that caveat, not as a property of our system.

## 10. Glossary (plain English)

- ADC: analog-to-digital converter; here, a comparator per real and imaginary part (one-bit).
- AGC: automatic gain control; scales the signal to the converter's range. Not needed at one bit.
- AQNM: additive quantisation noise model; treats the quantiser error as independent Gaussian noise (Mo eq. 49).
- Arcsine law: the exact correlation of the outputs of two one-bit quantisers for Gaussian inputs (Li eq. 13).
- Beamspace: the signal after the DFT codebooks W and F.
- Bussgang decomposition: writes a nonlinear quantiser as a linear gain plus an uncorrelated error term.
- BLMMSE: Bussgang linear MMSE estimator; LMMSE applied to the Bussgang-linearised output (Li eq. 12, 14).
- DFT pilots: pilot columns taken from a discrete Fourier matrix, all of equal magnitude.
- Dithering: adding known noise before quantisation. Not used in these papers.
- FOM (Walden): figure of merit F = 2^SNRbits f_samp / P_diss; higher means more efficient.
- LMMSE: linear minimum mean-squared-error estimator.
- NMSE: normalised MSE, relative to the signal power, in dB.
- NOMP: normalised orthogonal matching pursuit (Mamandipoor et al., 2016); our classical reference.
- Pd (paper-style): fraction of angle components within 1 deg after Hungarian matching of estimated to true angles.
- Quantiser ML: maximum likelihood with exact cell probabilities for the quantised data (our mlq.py).
- SNR-bits: effective number of bits from measured SNR, (SNR dB - 1.76)/6.02 (Walden).
- Steering vector: the response of an array to a direction; here, the DFT atom at a given angle.
- Uncorrelated vs independent: uncorrelated means zero covariance; independent means the conditional distribution does not depend on the signal. The Bussgang error is uncorrelated but not independent (A1).

## 11. Files and reproduction

- Scripts: D:/ai_ml_project/paper/evidence/adc_program/papers/bussgang_power_check.py (+ .log, .json, run1 log) and bussgang_power_followup.py (+ .log, .json). Both use stress_sim (deep_research/stress) and common.nomp_greedy(H, 3, 4, 3), as in finetune/adversarial/s3_adc_test.py.
- Large caches: C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/bussgang/ (text, PDFs, page images, Crossref JSON).
- Run time: about 1 minute, 2 threads, memory-light (decoding in chunks of 100 scenes).
