# GNOMP (Zhu et al., IEEE TAES 2026): a study guide for our few-bit ADC work

**Verified bibliographic entry.** J. Zhu, H. Zhang, N. Zhang, J. Fang, and F. Qu, "Line Spectrum Estimation and Detection With Few-Bit ADCs: Theoretical Analysis and Generalized NOMP Algorithm," *IEEE Transactions on Aerospace and Electronic Systems*, vol. 62, pp. 3933-3951, 2026. DOI: 10.1109/TAES.2025.3649136 (IEEE Xplore document 11316606, date of insertion 26 December 2025). Preprint: arXiv:2307.00491 (v1 2 Jul 2023, v2 30 Dec 2023, v3 18 Jan 2024, v4 3 Aug 2024).
Checked 2026-10-16 against Crossref (title, five authors, container, vol. 62, pages 3933-3951, year 2026; Crossref gives no issue number), the IEEE Xplore page metadata (pages 3933-3951, DOI, date of insertion) and the arXiv abstract page (four versions; no journal reference listed there).

**What was read (2026-10-16; this pass re-verified every statement against the sources).**
- (P) The **published** full text: the IEEE Xplore HTML of document 11316606, opened in the session's browser (the full text was served). Read in full: Secs. I-VI, eqs. (1)-(70), Algorithm 1, all figure captions, Appendices A-B. The reference list was read through the IEEE REST endpoint (refs. 1-30) and Crossref (all 36). Images inspected directly: Table I, Fig. 5 and Fig. 8. The HTML has no page numbers, so (P) locations are section, equation, figure and algorithm numbers. **This is the version we cite.**
- (A) **arXiv v4**: the arXiv HTML and the PDF (38 pages, single column, footer "August 6, 2024 DRAFT"). "v4 p.N" counts PDF pages (the printed page number is N-1). Text extracted locally (C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/gnomp_v4pdf/v4.txt); pages 12 and 22 were rendered and read as images. Equation numbers below are those of the v4 **PDF**; the arXiv HTML numbers them differently (for example, the SNR loss is (30) in the PDF and (48) in the HTML).
- (P) and (A) differ. (P) adds Figs. 1-3 (checks of the asymptotic distributions), Sec. V-E (noise-variance mismatch), Sec. V-H (ROC), Sec. V-J (comparison with an ANM method) and Appendix B. (P) drops: v4's 1-bit closed-form loss analysis (v4 eqs. (31)-(35), p.12-13), the "about 12 dB" detectability rule of thumb (v4 p.12), the multi-target product bound (v4 Sec. III-C), the dynamic-range study (v4 Sec. V-D) and the random-linear-measurement experiment against AST (v4 Sec. V-H). The SNR loss is eq. (31) in (P) and eq. (30) in the v4 PDF.
- **Copyright note:** verbatim quotation is limited to one short quote in this note (claim 4 in Sec. 6). Every other claim is a close paraphrase with its exact location, so it can be checked.
- **No official code** was found (web searches 2026-10-16; neither version gives a code link). This absence is UNVERIFIED.

Our own runs for this note (all stdout saved next to this file; all CPU, light):
- `repro_gnomp_vb.py` -> `repro_gnomp_vb.log`: reproduction of (P) Sec. V-B / Fig. 6 (SNR loss and detection probability), 10^4 trials per point.
- `repro_checks.py` -> `repro_checks.log`: CFAR thresholds, off-grid check, run-time ratios.
- `snrloss_our_cells.py` -> `snrloss_our_cells.log`: the paper's SNR-loss theory applied to our ADC cells (OUR extension).
- `gnomp_oracle_pd_our_cells.py` -> `gnomp_oracle_pd_our_cells.log`: the paper's oracle detection bound applied to our ADC cells (OUR extension).

---

## 1. Plain-English summary

Cheap radars and receivers would like to use analog-to-digital converters (ADCs) that keep only 1-4 bits per sample, because high-resolution ADCs cost a lot of power and money. With so few bits the recorded signal is badly distorted. The usual tools assume clean samples: the FFT, matched filters and the greedy NOMP algorithm. On such data they see ghost peaks and lose weak targets. This paper asks three questions. How much signal strength does a weak target effectively lose because of the coarse ADC and because of the other, stronger targets? How should we decide, with a controlled false-alarm rate, whether one more target is present? How can we find all targets quickly and accurately?

Their answer is a greedy loop called **GNOMP** (generalized NOMP). Each round does four things:
1. It checks whether one more target is hiding in the data. The check uses a test statistic that can be computed with one FFT and compared with a threshold chosen for a fixed false-alarm probability.
2. It finds that target's rough frequency on a fine grid.
3. It polishes all targets found so far with Newton steps on the exact probability of the observed coarse samples.
4. It stops when the test says "nothing more".

The paper also gives a formula for the "SNR loss" caused by the quantiser and by the other targets. Simulations and two real 77 GHz radar recordings support the theory and show that GNOMP is faster than its Bayesian and convex competitors and detects targets as well as or better than they do. Everything is for one-dimensional frequencies (one number per target).

## 2. The problem and the system model

**Their model** ((P) eqs. (1)-(3), Sec. II). There is one snapshot of N samples:

y = Q_C( sum_{k=1..K} a(omega_k) x_k + eps ),  with a(omega) = [1, e^{j omega}, ..., e^{j(N-1) omega}]^T and eps ~ CN(0, sigma^2 I).

- Q_C quantises the real part and the imaginary part separately with the same uniform B-bit quantiser ((P) eqs. (1)-(3), citing Shlezinger et al. [24] for the quantiser).
- The quantiser has b = 2^B levels and full-scale range gamma. Its thresholds are tau_d = d*2gamma/b - gamma for d = 1..b-1, with tau_0 = -inf and tau_b = +inf, so the two outer cells are open. The output is the cell midpoint (mid-rise), and samples with |x| > gamma are clipped to the outer level ((P) eq. (3)).
- Each output sample tells the receiver which cell [l(y_n), u(y_n)) its real part fell into, and likewise for the imaginary part.
- The unknowns are the frequencies omega_k, the complex amplitudes x_k and the number of sinusoids K.
- "Time-domain SNR" is |x|^2/sigma^2 and "integrated SNR" is N|x|^2/sigma^2. Unless stated otherwise, "SNR" in (P) means the integrated SNR (Sec. II).
- (P) notes, after eq. (20), that the Rao test applies unchanged to compressive measurements y = Q_C(Phi sum a(omega_k) x_k + eps): the atom is replaced by Phi a.

**Our model** (stress_sim.py, finetune/adversarial/mlq.py). Y = W^H H F + Z is a 16x16 matrix of beam samples (N = 256 complex values). H = 16 sum_l alpha_l a_r(u_l) a_t(v_l)^H, and W and F are unitary 16-point DFT codebooks. The ADC uses per-scene peak automatic gain control (AGC): A = max(|Re|, |Im|) over the 256 noisy samples, then a b-bit uniform mid-rise quantiser with step 2A/2^b on Re and Im separately. In the paper's notation this is exactly the "compressive" form with Phi = F^T kron W^H applied to vec(H), with the noise added after Phi, as in theirs.

| Paper symbol | Meaning in the paper | Our project |
|---|---|---|
| y in C^N | quantised samples | vec(Y_q), Y_q = ADC_b(Y) in C^{16x16}, N = 256 |
| N | number of samples (= DFT bins of the detector) | 256 beam samples. A 2-D detector has 16x16 = 256 DFT bins |
| a(omega), unit-modulus entries | 1-D sinusoid atom | 2-D beamspace atom vec(W^H 16 a_r(u) a_t(v)^H F). Entries are not unit-modulus, but the squared norm is 256 = N |
| Phi (compressive extension) | linear map before the noise | F^T kron W^H (unitary). The antenna-domain data H~ = W Y F^H is Phi^H y |
| omega_k | frequency | the pair (u_k, v_k), u = pi cos(psi) (AoA), v = pi cos(phi) (AoD): two frequencies per path |
| x_k | complex amplitude | 16 alpha_k (path gain; alpha ~ CN(0, 1/L), sorted strongest first, iabr2_sim.sample_paths) |
| K | number of sinusoids, unknown, found by CFAR | L = 3, known to all our decoders |
| B, b = 2^B | bits, number of levels | our "b" bits, 2^b levels |
| gamma | full scale; in Sec. V an oracle rule gamma = max(abs(x_1), ..., abs(x_K), 3 sigma/sqrt2) | A = per-scene peak of the noisy samples (data-driven AGC) |
| step 2gamma/b, outer cells open | quantiser | step 2A/2^b. acommon.adc_cells closes the outer cells at +-A, because with peak AGC no sample exceeds A |
| sigma^2, known | complex noise variance | 10^(-SNR/10), complex. Per-component std sqrt(sigma^2/2), the same convention as the paper's sigma/sqrt2 |
| integrated SNR N abs(x)^2/sigma^2 | per-target SNR after coherent integration | 256 abs(alpha_k)^2/sigma^2. An average path (abs(alpha)^2 = 1/3) gets our SNR + 19.3 dB (`repro_checks.log`) |
| zeta | "thresholds" = the sum of the already-detected (or, in the theory, the other) paths | beamspace contribution of the other paths |
| Y, W, F, H, L, b, A, SNR, u, v | (our symbols) | as defined above |

**The key difference.** The paper is **1-D** (one frequency per target, one snapshot). Our problem is **2-D**: two angles per path, measured through a beamformer. Nothing in (P) handles 2-D frequencies. Only arXiv v4 footnote 2 (v4 p.5) lists "multidimensional LSE" among settings GNOMP can be extended to, and admits these details are not presented. (P) drops that sentence; its list of settings (Sec. I-B) is unknown noise variance, time-varying thresholds and random linear measurements.

## 3. The method, step by step

### 3.1 The exact likelihood of coarse samples ((P) eqs. (14), (43))
Every real number we record is only known to lie in a cell [l, u). If the model predicts mean mu for that number and the noise std is s = sigma/sqrt2, the probability of the observation is Phi((u - mu)/s) - Phi((l - mu)/s), where Phi is the standard normal CDF (a "probit" cell probability). The log-likelihood l(y; omega, x) sums the logs of these probabilities over all real and imaginary parts.
*One sentence:* this measures how well a candidate set of paths explains which cell every sample fell into.

For fixed frequencies it is concave in the amplitudes ((P) Sec. IV, citing Zymnis, Boyd and Candes, IEEE SPL 2010 [34]), so amplitudes are easy. Frequencies make it non-concave, so a good starting point is needed ((P) Sec. IV-A).
**This is the same objective as our quantiser-ML** (finetune/adversarial/mlq.py, docstring: sum log(Phi((hi - mu)/s) - Phi((lo - mu)/s)), s = kappa*sigma). We already optimise GNOMP's likelihood, but with a different search.

### 3.2 Pseudo-measurements phi_B ((P) eq. (18))
For each sample, given the known part zeta of the signal (the paths already found), phi_B is minus (phi(u') - phi(l')) / (Phi(u') - Phi(l')) with u' = (u - zeta)/s and l' = (l - zeta)/s, separately for Re and Im.
*One sentence:* phi_B is "the best guess of the leftover unquantised signal, divided by the noise std", computed from the cell alone.

- It equals (E[z | z in cell, mean zeta] - zeta)/s. This identity is OUR reading, from the standard truncated-normal mean; the paper does not state it.
- With infinitely many bits it becomes (y - zeta)/s ((P) eqs. (27)-(28)).
- With 1 bit and zeta near 0 it is about sqrt(2/pi)(sign Re y + j sign Im y), so it does not depend on sigma (v4 p.12, v4 only; checked on the rendered page).

### 3.3 Per-sample information h_B ((P) eqs. (9a)-(10), (12))
h_B(x, sigma^2) is the Fisher information of one quantised real sample whose mean is x, relative to an unquantised sample (it equals 1 when B is infinite; (P) eqs. (11), (65)-(66)).
*One sentence:* h_B says how much a sample is still worth after quantisation. At 1 bit it is 2/pi when the sample's mean sits on the threshold, and it decays roughly like exp(-x^2/sigma^2) when the mean is far from every threshold (v4 eq. (32), p.12).
The paper uses h_{B,+} = (h_B(Re) + h_B(Im))/2 and h_{B,-} = (h_B(Re) - h_B(Im))/2 ((P) eq. (9)). The Fisher information matrix of the amplitude is (2/sigma^2)[a^H diag(h_+) a I + terms in a^T diag(h_-) a] ((P) eq. (7)); its inverse is the CRB ((P) eq. (8), Proposition 1; proof via Lemma 1 of [22] in Appendix A).

### 3.4 The Rao test ((P) eqs. (13), (19), (20), (32), (36))
The test asks: if I add a tiny sinusoid at frequency omega to the current model zeta, does the likelihood rise significantly? The Rao (score) test needs only the derivative at "no new sinusoid", so the new amplitude never has to be fitted. (P) Sec. III-A explains the choice: GLRT and Wald have the same asymptotic performance but need the ML estimate.

T(omega) = [S+ |a^H phi_B|^2 - Re{S-(omega) (a^H phi_B)^2}] / [S+^2 - |S-(omega)|^2], with S+ = a^H diag(h_{B,+}(zeta)) a and S- = a^T diag(h_{B,-}(zeta)) a.

*One sentence:* it is a matched filter (an FFT) applied to the pseudo-measurements and normalised by how informative the samples are; the second term only corrects for unequal I/Q information ((P) text after eq. (19)).
- For unit-modulus atoms, S+ = sum_n h_{B,+,n} for every frequency ((P) eq. (20)). a^H(omega_g) phi_B for all DFT frequencies is one FFT, and S-(omega_g) is one IFFT of h_{B,-} ((P) text after eq. (35)).
- Dropping h_- gives the simpler test T' = |a^H phi_B|^2 / (a^H diag(h_+) a) (v4 eq. (28), p.11). (P) eq. (31) says the h_- term is usually very small.
- The detector takes the maximum of T over the N DFT bins ((P) eq. (36)).

### 3.5 CFAR threshold ((P) eqs. (21), (33), (38)-(41))
Under "no target", T is asymptotically chi-square with 2 degrees of freedom ((P) eq. (21): large data record and weak signal, citing Kay [28]). On the grid this is a conjecture ((P) eq. (33)), checked by simulation in (P) Figs. 1-2. If the bins are treated as independent ((P) Fig. 3 shows correlations near 0), the threshold for a false-alarm probability P~FA over the whole spectrum is

tau_th = -2 ln(1 - (1 - P~FA)^{1/N})  ((P) eq. (39)), approximately -2 ln(P~FA/N)  ((P) eq. (41)).

*One sentence:* the threshold depends only on the chosen false-alarm rate and the number of bins, not on the data, so GNOMP keeps a constant false-alarm rate (CFAR) ((P) Sec. IV, "Main idea").
- The paper uses P~FA = 0.01, following NOMP [25]; this corresponds to a per-cell false-alarm rate of about 0.01/N ((P) end of Sec. III-B).
- For N = 128 this gives tau_th = 18.90, or 25.53 dB when written as 20 log10. That matches the paper's "25.5 dB" in Sec. V-C (our reading of their dB convention). For our N = 256 grid, tau_th = 20.29 (`repro_checks.log`).

### 3.6 Detection probability and SNR loss ((P) eqs. (22), (23b), (30), (31), (34), (35), (40))
With a target present, T is non-central chi-square with non-centrality lambda = (2/sigma^2) beta_g [S+ |x|^2 + Re{S- x^2}] ((P) eq. (34)). The factor beta_g <= 1 is the loss from the target sitting between grid points (a Fejer kernel, (P) eq. (35)). Then Pd = Q1(sqrt(lambda), sqrt(tau_th)), where Q1 is the Marcum Q-function ((P) eq. (40)). Pd is defined as a threshold crossing **in the correct bin** (the bin nearest the true frequency).

**SNR loss** = lambda_unquantised / lambda_B, approximately N / (a^H diag(h_{B,+}(zeta)) a) ((P) eq. (31)). The "effective" integrated SNR is lambda_B/2 ((P) eq. (30)).
*One sentence:* to reach the same Pd and false-alarm rate, a B-bit receiver needs 10 log10(SNR loss) dB more SNR than an ideal one ((P) text after eq. (31)).
- The loss depends on sigma, B and the other targets zeta. Strong other targets push samples far from the thresholds, and those samples then carry almost no information about a weak target.
- Two v4-only special cases (v4 p.12-13): at 1 bit with zero threshold (zeta = 0) the loss is pi/2 = 1.96 dB, the minimum; when all thresholds are large, the loss grows rapidly with min|zeta|/sigma (v4 eq. (34)).
- In the oracle case where the other targets are known exactly, this Pd is an **upper bound** for the multi-target problem ((P) Sec. II, last paragraph). v4 Sec. III-C (eqs. (64)-(66), v4 only) multiplies the per-target bounds into an upper bound on the probability that all K targets are detected.
- v4 p.12 (v4 only) gives a rule of thumb: a target is reliably detectable when its effective integrated SNR (integrated SNR minus the loss) exceeds about 12 dB.

### 3.7 Newton refinement on the quantised likelihood ((P) eqs. (50)-(55), (57), (58))
After a detection:
- Take the coarse frequency as the argmax of T on a 4x oversampled grid ((P) eqs. (50)/(52); gamma_os = 4 for robust Newton convergence, citing NOMP [25]). (P) first describes the exact alternative, a per-grid-point amplitude ML ((P) eqs. (48)-(49)), and rejects it as too expensive.
- Fit the amplitude by maximising the concave likelihood with the frequency fixed ((P) eqs. (51)/(53)).
- "AM": one Newton step on the frequency with the amplitude fixed ((P) eq. (54)), then one Newton step on the amplitude with the frequency fixed ((P) eq. (55)).
- "Cyclic refinement" (block coordinate descent, (P) eq. (57)) revisits every detected path in turn, treating all the others as known zeta, and applies the same AM update. Finally all amplitudes are refit jointly by Newton with the frequencies fixed ((P) eq. (58)).
- Each refinement step is accepted only if it increases the log-likelihood ((P) Sec. IV-B, after eq. (58)).
- The formulas for the frequency derivatives are omitted in both versions ("for brevity", v4 p.19). We give them below as our own derivation.

### 3.8 Spurious-component suppression ((P) Algorithm 1 lines 21-29, Sec. IV-B)
A very strong target (their example: 60 dB) can make GNOMP accept a small fake component before the next real target. Rule: from the third iteration on (iteration counter m > 2), if the second-to-last component in the list has less than half the amplitude of the last one, re-test it with the Rao test against all the other components. If it fails the CFAR threshold, remove it, then rerun cyclic refinement and the amplitude update.

### 3.9 Pseudo-code (detailed enough to re-implement)
The structure follows (P) Algorithm 1 and Sec. IV. The derivative formulas, the starting amplitude and the step-acceptance details are OUR derivation or implementation choices, consistent with the paper's text.

```
Inputs: y (N complex, quantised), cells (lo_R, hi_R, lo_I, hi_I) per sample, sigma^2 (known),
        P_FA (default 0.01), gamma_os = 4, R_s = 1 (default), R_c = 3 (default; R_c = 4 for the unknown-sigma variant in Sec. V-I),
        Newton iterations per amplitude update Iter = 2..5 (paper: "usually about 2~5").
s = sigma/sqrt(2);  tau_th = -2 ln(1 - (1 - P_FA)^(1/N))          # (P) eq. (39)

def per_sample(mu_R, mu_I):                     # mu = real/imag model mean per sample
    for c in {R, I}: a = (lo_c - mu_c)/s; b = (hi_c - mu_c)/s; P = Phi(b) - Phi(a)
                     g_c  = (phi(a) - phi(b)) / (s P)               # d logP / d mu   (phi(+-inf) = 0)
                     g2_c = [(a phi(a) - b phi(b))/P - ((phi(a)-phi(b))/P)^2] / s^2   # d2 logP / d mu2 <= 0  (x*phi(x) -> 0 at +-inf)
    return g, g2                                 # phi_B = s*(g_R + j g_I)  ((P) eq. (18)); (P) eq. (17) carries an extra overall minus sign

def h_B(m):                                      # (P) eq. (10), thresholds tau_0..tau_b of the quantiser
    return sum_d [phi((tau_{d+1}-m)/s) - phi((tau_d-m)/s)]^2 / [Phi((tau_{d+1}-m)/s) - Phi((tau_d-m)/s)]

def rao_spectrum(zeta, grid_size):               # (P) eqs. (19)/(20)/(32)
    phi_B = s * (g_R + j g_I) evaluated at mu = zeta
    hp = (h_B(Re zeta) + h_B(Im zeta))/2;  hm = (h_B(Re zeta) - h_B(Im zeta))/2
    c[g]  = sum_n phi_B[n] exp(-j w_g n)         # zero-padded FFT, w_g = 2 pi g / grid_size
    Sp    = sum_n hp[n]                          # = a^H diag(hp) a for unit-modulus atoms
    Sm[g] = sum_n hm[n] exp(+j 2 w_g n)          # IFFT-type sum
    T[g]  = (Sp |c[g]|^2 - Re{Sm[g] c[g]^2}) / (Sp^2 - |Sm[g]|^2)
    return T, c, Sp

def loglik(paths): mu = sum_k a(w_k) x_k;  return sum over Re and Im of log(Phi((hi-mu)/s) - Phi((lo-mu)/s))   # (P) eq. (43)

def newton_amplitudes(paths, idx):               # concave; (P) eqs. (55)/(58)
    J = d mu / d [Re x_idx, Im x_idx]  (Re-sample n -> [Re a_n, -Im a_n]; Im-sample n -> [Im a_n, Re a_n]; (P) eq. (63))
    grad = J^T g;  Hess = J^T diag(g2) J;  step = -Hess^{-1} grad;  accept only if loglik increases (else halve, then skip)

def newton_frequency(paths, k):                  # (P) eq. (54); a_n(w) = e^{j w n}
    dmu  = [Re, Im] of (j n a_n(w_k) x_k);   d2mu = [Re, Im] of (-n^2 a_n(w_k) x_k)
    l1 = sum g dmu;  l2 = sum (g2 dmu^2 + g d2mu)
    if l2 < 0: w_k <- w_k - l1/l2, keep only if loglik increases       # as in NOMP: step only where locally concave (our choice)

paths = []; zeta = 0; m = 0
while max_{g in N-point DFT grid} rao_spectrum(zeta, N).T > tau_th:          # detection on Omega_DFT, Algorithm 1 line 3
    m += 1
    T, c, Sp = rao_spectrum(zeta, gamma_os * N)                              # identify on Omega_os, line 6
    g* = argmax T;  w = 2 pi g*/(gamma_os N);  x = s * c[g*] / Sp            # one Fisher-scoring step from x = 0 (our start)
    repeat Iter: newton_amplitudes on x (others fixed in zeta)               # line 7, (P) eq. (53)
    repeat R_s: newton_frequency on w; newton_amplitudes on x               # single refinement, line 9
    paths.append((x, w))
    repeat R_c rounds: for each path k: treat the others as zeta_k; newton_frequency(k); newton_amplitudes(k)   # cyclic refinement, lines 13-17
    joint Newton on all amplitudes with frequencies fixed                    # amplitude update, line 19, (P) eq. (58)
    if m > 2 and |x_{last-1}| < 0.5 |x_last|:                                # spurious suppression, lines 21-29
        zeta_r = sum of all paths except (last-1)
        if Rao statistic T(zeta_r, at w_{last-1}) <= tau_th: remove path last-1; redo cyclic refinement + amplitude update
    zeta = sum_k a(w_k) x_k                                                  # line 30
return paths            # K_hat = len(paths)
```

**Unknown-noise-variance variant.** Neither version describes the algorithm. v4 footnote 2 (v4 p.5) says such extensions are not explicitly presented; (P) only reports results (Secs. V-I, V-K). In the real-data section the known-sigma GNOMP is given an **FFT-based noise-variance estimate** ((P) Sec. V-K), and the unknown-sigma variant is run there only for B > 1. For 1 bit with unknown sigma, the paper says the amplitudes are ambiguous and sigma^2 is simply set to 1 ((P) Sec. V-E). So "GNOMP (sigma^2 unknown)" is UNVERIFIED as an algorithm and cannot be re-implemented faithfully from the paper. An obvious re-implementation (adding sigma as a Newton/ML variable) would be OUR design and must be named as such.

**Our 2-D adaptation (OUR proposal, not in the paper).**
- Replace omega by (u, v) and the atom by the beamspace atom G(u, v) = W^H 16 a_r(u) a_t(v)^H F (the paper's own compressive form with Phi = F^T kron W^H).
- a^H phi_B becomes a 2-D zero-padded FFT of the antenna-domain pseudo-measurement W [phi_B] F^H, on a (4x16) x (4x16) grid.
- S+(u, v) = sum_{q,p} h_+,qp |G_qp(u, v)|^2 and S-(u, v) = sum_{q,p} h_-,qp G_qp(u, v)^2. Both separate into products of 16-point receive and transmit kernels, so they are matrix products on the grid. S+ is no longer constant in (u, v), so the normalisation in T matters.
- Newton runs on (u, v) with a 2x2 Hessian (or LM, as in mlq.py).
- tau_th uses N = 256 bins (the beam grid is the 16x16 DFT grid, so the H0 bins are orthogonal).
- Peak AGC: the receiver knows A from its own output levels (acommon.adc_cells), so the cells are known, as GNOMP requires. The one sample that sets A sits exactly on a cell edge; GNOMP's model (a fixed, known gamma) ignores this data dependence.

## 4. Assumptions

| Item | What the paper assumes | Where |
|---|---|---|
| Noise variance | **Known** for the main GNOMP (the CFAR threshold and phi_B depend on sigma). For B >= 2, a noise power at least 2 dB above the assumed one raises P_FA by about 1.5-2 orders of magnitude. At 1 bit, P_FA stays close to nominal. The paper concludes sigma must be estimated in real time except at 1 bit. | (P) Sec. IV (paragraph after Algorithm 1), Sec. V-E, Fig. 9 |
| Unknown sigma variant | Results shown (Secs. V-I, V-K), algorithm not given. At 1 bit, sigma^2 is set to 1 (amplitudes ambiguous). | (P) Secs. V-E, V-I, V-K; v4 footnote 2 (p.5) |
| Number of targets | Unknown, estimated by the CFAR stopping rule. In the frequency-MSE study K = 8 is given to GNOMP. | (P) Sec. IV, Sec. V-F |
| Dithering | Not used; the word does not appear in (P). Sec. V-I uses **known time-varying thresholds** (drawn from 8 values uniformly spread over [-1, 1]) with 1-bit "signed measurements". OUR reading: this acts like known dither, which is what lets the amplitude scale (and sigma) be identified at 1 bit; the paper does not say this. | (P) Sec. V-I |
| AGC / full scale | No AGC. In simulations gamma = max of the true amplitudes and 3 sigma/sqrt2, an oracle rule; Secs. V-E and V-H use gamma = 3 sigma/sqrt2. The real data use gamma = 60. Quantiser design is declared out of scope. The outer cells are open, so clipping is part of the model. | (P) Sec. V (first paragraph), V-E, V-H, V-K |
| Snapshots | Single snapshot. Multisnapshot is only claimed as an extension (v4 only). | (P) eq. (1); v4 footnote 2 |
| Dimension | 1-D frequencies only. Multidimensional is only claimed (v4 only) and never demonstrated. The real-data test is range-only (fast time, N = 256). | v4 p.5; (P) Sec. V-K |
| Grid | N-point DFT grid for detection, 4x oversampled grid for the start, then off-grid Newton (gridless). | (P) eqs. (36), (50)-(55) |
| Statistics | Weak-signal, large-N asymptotics. Under unknown frequency the chi-square law is a conjecture checked by simulation. Bins are assumed independent. The bounds are oracle bounds (other targets known). | (P) eqs. (21), (33), (38), Figs. 1-3, Sec. II |
| Noise | White circular Gaussian. I and Q are quantised separately with the same quantiser. | (P) eqs. (1), (3) |

## 5. Reported results (exact numbers, figure, conditions)

Figure numbers are those of (P), with the v4 figure in brackets. "GNOMP (B = infinity)" means plain NOMP ((P) Sec. V, first paragraph).

| (P) section / figure | Conditions | Reported result |
|---|---|---|
| III-A, Fig. 1 (new) | B = 1, K = 3, N = 512, omega = [2, 3.8, 0.6], x = [-0.11-0.16j, -0.18-0.30j, 0.04+0.35j], integrated SNRs 13/18/18 dB, sigma^2 = 1 | The simulated Rao statistic follows chi-square_2 under H0 and the non-central law under H1 with lambda_R,1 = 23.32 |
| III-B, Fig. 2 (new) | as Fig. 1 but omega_1 = 0.1166 (between bins), g = 9 | beta_g = 0.405, lambda_R,B,g = 9.45; the conjectured law holds in simulation |
| III-B, Fig. 3 (new) | 1 bit, K = 3, N = 512, all three targets at 40 dB, sigma^2 = 1 | Correlations between bins approximately 0 |
| V-A, Fig. 5 [v4 Fig. 4] | zeta = a(0.15)(-0.96 - 1.75j) (about 6 dB time-domain), target omega = 2.34, sigma^2 = 1; x = -0.27 + 0.29j (-8 dB) or -0.68 + 0.73j (about 0 dB); 5000 MC; N up to 800 | Amplitude MSE approaches the CRB as N grows. Legend (checked on the figure image): GNOMP and CRB for B = 1, B = 2 and B = infinity |
| **V-B, Fig. 6 [v4 Fig. 5]** | N = 1024, zeta = 2 a(omega_1) (time-domain 6 dB), omega_1 = pi/2 (sc. 1) or pi/2 + 0.1 (sc. 2), P~FA = 0.01, 10^4 MC | Theory, eq. (31): **4.8 / 2.3 / 1 dB** (sc. 1), **6.4 / 2.2 / 0.9 dB** (sc. 2) for 1/2/3 bits. Measured Pd = 0.5 points: sc. 1 at 17.2 / 14.7 / 13.3 / 12.4 dB (1/2/3/infinity bits), i.e. losses **4.8 / 2.3 / 0.9 dB**; sc. 2 at 18.7 / 14.6 / 13.2 / 12.4 dB, i.e. **6.3 / 2.2 / 0.8 dB**. Target frequency, phase and sigma^2 are not stated |
| V-C, Fig. 7 [v4 Fig. 2] | K = 2, omega = 2.2 and 2.4, x1 = -1.505 - 0.497j, x2 = -0.164 - 0.609j (time-domain 4 and -4 dB), N = 128, B = 1, P_FA = 0.01 | Threshold 25.5 dB. First test 41.4 dB (start 2.197). Second 28.8 dB, 3.3 dB above threshold (start 2.405). Third 24 dB, so GNOMP stops. Estimates 2.198 / 2.401 and x^ = -1.506 - 0.680j / -0.333 - 0.591j. Theoretical SNR losses 2.58 / 5.21 dB vs 2.67 / 5.42 dB computed from the estimates |
| V-D, Fig. 8 [v4 Fig. 6] | K = 8, all at integrated 20 dB, min separation 2.5 DFT bins, 300 runs per point, nominal P_FA 1-10 %, B = 1, 2, 3, infinity (from the figure); N not stated (UNVERIFIED) | Paper: measured false-alarm and over-estimation probabilities closely follow the nominal value. Our reading of Fig. 8: close at 1-6 %, about 1-3 percentage points above nominal at 8-10 % for B = 1, 3, infinity (B = 2 stays near nominal) |
| V-E, Fig. 9 (new) | K = 3, omega = [2, 3.8, 0.6], x = [0.25 - 0.25j, -0.34 + 0.09j, -0.06 + 0.35j], nominal sigma0^2 = 1, gamma = 3 sigma0/sqrt2, N = 512, all 18 dB, P_FA0 = 0.01 | 1 bit: P_FA stays near nominal as the true noise power increases. B >= 2: +2 dB or more noise power gives P_FA about 1.5-2 orders of magnitude too high |
| V-F, Fig. 10 [v4 Fig. 8] | K = 8 (known), equal SNRs 14-40 dB, N = 512, 300 runs | Frequency MSE approaches the CRB from 14 to about 34 dB for 1-3 bits, then departs from it (except unquantised) |
| V-G, Fig. 11 [v4 Fig. 9] | 8 sinusoids, 7 at 30 dB, the 8th at 10-26 dB, N = 512, 1000 MC | P_FA near nominal, over-estimation below nominal, Pd of the weak target close to the oracle bound ((P) eq. (40)) |
| V-H, Fig. 12 (new) | K = 3, omega = [2, 3.8, 0.6], N = 512, target 1 at 12-15 dB, the others at 15 dB, sigma^2 = 1, gamma = 3 sigma/sqrt2, B = 1, 2, 3 | ROC close to the theoretical (oracle) ROC |
| **V-I, Fig. 13 [v4 Fig. 10]** | 1-bit signed data with time-varying thresholds (8 values in [-1, 1]), 8 sinusoids (7 at 25 dB, 1 at 14-24 dB), sigma^2 = 1, 500 MC, R_c = 3 (known sigma) / 4 (unknown); N not stated (UNVERIFIED). GNOMP and MVALSE-EP modified for this setting; MVALSE-EP and 1bMMRELAX estimate sigma | Pd = 0.5 at P_FA = 0.01 needs **15.7 dB** (GNOMP), **15.7 dB** (GNOMP, sigma unknown), **16.7 dB** (1bMMRELAX): a **1 dB** gain. Pd order: GNOMP (P_FA 0.05) > GNOMP (0.01) > 1bMMRELAX > MVALSE-EP. P_FA and over-estimation near nominal except the unknown-sigma variant at 25 dB. Run time (50 MC, i7-12700, 32 GB, Windows 11, mean +- 3 std): **1.70 +- 0.25 s** (P_FA 0.01), 1.74 +- 0.21 (0.05), 6.12 +- 1.56 and 6.18 +- 1.38 (sigma unknown), **7.20 +- 0.84** (1bMMRELAX), **12.84 +- 0.15 s** (MVALSE-EP). Unknown sigma costs 3.55-3.60x (`repro_checks.log`) |
| V-J, Fig. 14 (new) | B = 1, K = 3, omega = [2, 3.8, 0.6], N = 128, sigma^2 = 1; ANM = negative log-likelihood + atomic norm, solved by CVX, frequencies by MUSIC (Appendix B); lambda_A = 25 chosen with 500 MC at 16 dB; comparison 100 MC | Debiased NMSE: **ANM better below 17 dB** integrated SNR, GNOMP better above. Time 0.056 +- 0.02 / 0.055 +- 0.01 / 0.054 +- 0.01 s (GNOMP, P_FA 0.01/0.05/0.1) vs 13.78 +- 0.80 s (ANM). The caption says "AST" while the text says ANM |
| V-K, Table I, Figs. 15-17 | TI AWR1642 (2 Tx, 4 Rx, 77 GHz, slope printed as 29.982 MHz/s, sweep 60 us, PRI 160 us, bandwidth 1798.92 MHz, fs = 10 MHz, 128 pulses per CPI, N = 256 fast-time samples), dataset of NOMP-CFAR [26] (Xu et al., IEEE TAES 2023); sigma^2 = 250, P~FA = 0.01, gamma = 60, B = 1, 2, 3, infinity; range only | Exp. 1 (people at about 4.88 and 3.05 m): both found plus a leakage line; occasional false alarms at 2 and 3 bits. Time: NOMP 0.004 s; GNOMP 0.11 / 0.15 / 0.14 s (B = 1/2/3; 0.85 / 1.05 s with sigma unknown at B = 2/3); MVALSE-EP 1.18 / 1.50 / 1.47 s. Exp. 2 (people at 4.87 and 2.63 m, cyclist 7 to 2 m at about 2 m/s): at 1 bit GNOMP and MVALSE-EP both miss the cyclist and MVALSE-EP adds a false alarm; at B >= 2 GNOMP finds all three while MVALSE-EP misses the cyclist; at 3 bits GNOMP (both variants) and NOMP each give one false alarm |
| v4 only, Sec. V-D, v4 Fig. 7 (p.27) | two sinusoids, N = 512, 300 MC | Instantaneous dynamic range about 10 / 22 / 30 dB for 1 / 2 / 3 bits |
| v4 only, Sec. V-H (p.30) | random Gaussian Phi (M x N), N = 128, K = 2, omega = 1.5 and 3.2, time-domain SNR 0 dB, M = 200 or 100; bits not stated (UNVERIFIED) | GNOMP 1.5008 / 3.2002 (M = 200) and 1.4997 / 3.2015 (M = 100), stopping at the third iteration; AST fails in both |
| v4 only, p.13 | 1 bit, zero threshold, weak target | Minimal SNR loss = pi/2 (1.96 dB) |

### 5.1 Our reproduction of Sec. V-B (Fig. 6) — `repro_gnomp_vb.log`

Setting (as the paper states): N = 1024, zeta = 2 a(omega_1), time-domain SNR of x1 = 6 dB, quantiser rule gamma = max(|x_k|, 3 sigma/sqrt2), P~FA = 0.01, 10^4 trials per point, full Rao test ((P) eqs. (19)/(32)) on the 1024-bin grid, detection = maximum in the correct bin and above tau_th ((P) Sec. III-B).
Our assumptions (the paper does not state them): sigma^2 = 4/10^0.6 so that x1 is exactly 6 dB (sensitivity run: sigma^2 = 1), target on-grid (bin 300), uniform random target phase, test evaluated with the true zeta (the paper's oracle setting).

| Quantity | Paper (P) Sec. V-B | Ours |
|---|---|---|
| Sc. 1 theory loss, eq. (31), 1/2/3 bits | 4.8 / 2.3 / 1 dB | 4.82 / 2.33 / 1.03 dB |
| Sc. 1 SNR at Pd = 0.5, 1/2/3/inf bits | 17.2 / 14.7 / 13.3 / 12.4 dB (measured) | MC 15.77 / 12.74 / 11.45 / 10.43 dB; theory eq. (40) 15.24 / 12.75 / 11.45 / 10.42 dB |
| Sc. 1 measured loss | 4.8 / 2.3 / 0.9 dB | 5.35 / 2.31 / 1.02 dB |
| Sc. 2 theory loss | 6.4 / 2.2 / 0.9 dB | 6.41 / 2.20 / 0.87 dB |
| Sc. 2 SNR at Pd = 0.5 | 18.7 / 14.6 / 13.2 / 12.4 dB | MC 16.77 / 12.63 / 11.30 / 10.40 dB; theory 16.84 / 12.62 / 11.30 / 10.42 dB |
| Sc. 2 measured loss | 6.3 / 2.2 / 0.8 dB | 6.37 / 2.23 / 0.90 dB |
| Zero threshold, gamma = 3 sigma/sqrt2 | 1 bit: pi/2 = 1.96 dB (v4 p.13) | 1.96 / 0.77 / 0.21 / 0.05 dB for 1/2/3/4 bits |
| Sensitivity sigma^2 = 1 | — | all losses within 0.1 dB of the main run (1-bit sc. 1 MC loss 5.40 dB) |

Verdict:
- **The theory (eq. (31)) is reproduced** to within 0.05 dB in both scenarios.
- **The measured losses are reproduced** to within 0.1 dB, except 1-bit scenario 1 (ours 5.35 dB vs 4.8 dB). Our 1-bit Monte-Carlo curve rises more slowly than the weak-signal theory above Pd = 0.5 (Pd 0.88 at 19 dB, 0.99 at 24 dB), and the unstated target frequency and phase may matter at 1 bit. This gap is not resolved.
- **Absolute SNRs are 2.0 dB lower than the paper's at every bit depth**, including unquantised (10.4 vs 12.4 dB). For unquantised data the Pd = 0.5 point depends only on N, P~FA and the target's offset from the grid. On-grid it is 10.42 dB, averaged over a uniformly random offset 11.58 dB, at a half-bin offset 14.35 dB (`repro_checks.log`). The paper's 12.4 dB therefore implies an off-grid target (OUR inference; the paper does not state the target frequency). Losses are differences, so they are insensitive to this.
- **How to reproduce this result yourself:** run `python repro_gnomp_vb.py 10000` (2 threads; run time was not recorded). Every other result in Sec. 5 needs a full GNOMP implementation, which does not exist in this project yet.

### 5.2 The paper's theory applied to our ADC cells (OUR extension, not a result of the paper)

**(a) SNR loss per path** — `snrloss_our_cells.log`. For each path, the other two paths are treated as known (the paper's oracle view). The loss is ||a_k||^2 / (a_k^H diag(h_+(zeta_k)) a_k), with our beamspace atom, the peak-AGC thresholds of the actual noisy scene and open outer cells. 1000 scenes per cell, validation seeds 81,000,000+, L = 3. "Effective" = integrated SNR minus loss. The last column applies v4's "about 12 dB" rule of thumb (v4 p.12; a 1-D rule, used here only as a rough guide).

| Cell | Path (1 = strongest) | Median integrated SNR (dB) | SNR loss (dB): median [10 %, 90 %] | Median effective SNR (dB) | Fraction effective < 12 dB |
|---|---|---|---|---|---|
| 1-bit / 15 dB | 1 / 2 / 3 | 36.2 / 32.7 / 27.9 | 2.21 [1.98, 5.50] / 2.32 [1.99, 7.21] / 2.44 [2.00, 7.78] | 33.3 / 29.3 / 24.2 | 0.004 / 0.010 / 0.064 |
| 2-bit / 15 dB | 1 / 2 / 3 | 36.3 / 32.8 / 27.9 | 2.22 [1.99, 6.23] / 2.41 [2.00, 6.74] / 2.50 [2.00, 8.56] | 33.2 / 29.5 / 23.9 | 0.001 / 0.010 / 0.065 |
| 3-bit / 15 dB | 1 / 2 / 3 | 36.4 / 32.7 / 28.0 | 2.19 [1.98, 5.51] / 2.34 [1.99, 6.44] / 2.56 [2.00, 8.14] | 33.5 / 29.6 / 24.2 | 0.001 / 0.011 / 0.074 |
| 1-bit / 0 dB | 1 / 2 / 3 | 21.3 / 17.6 / 12.9 | 1.98 [1.96, 2.30] / 1.99 [1.96, 2.54] / 2.00 [1.96, 2.68] | 19.1 / 15.4 / 10.6 | 0.028 / 0.199 / 0.624 |
| 2-bit / 0 dB | 1 / 2 / 3 | 21.2 / 17.6 / 13.2 | 1.97 [1.95, 2.25] / 1.98 [1.96, 2.43] / 1.99 [1.96, 2.56] | 19.1 / 15.5 / 10.9 | 0.013 / 0.167 / 0.579 |
| 3-bit / 0 dB | 1 / 2 / 3 | 21.1 / 17.7 / 12.7 | 1.86 [1.03, 2.07] / 1.87 [1.03, 2.16] / 1.87 [1.03, 2.23] | 19.3 / 15.9 / 11.0 | 0.004 / 0.136 / 0.578 |

**(b) Oracle detection bound per path** — `gnomp_oracle_pd_our_cells.log`. Same scenes. Pd_k = Q1(sqrt(lambda_k), sqrt(tau_th)) with tau_th = 20.29 (N = 256, P~FA = 0.01) and lambda_k from (P) eq. (34). beta_k is the energy fraction of path k in its nearest beam (the 2-D analogue of (P) eq. (35); median about 0.61); "beta = 1" is the on-grid best case. "All 3" is the mean of prod_k Pd_k (v4 eq. (66)), an upper bound on P(all three paths detected). These are bounds for a CFAR **detector**, not bounds on our 1-degree Pd metric.

| Cell | beta | Pd path 1 / 2 / 3 | Mean over paths | All 3 detected (upper bound) |
|---|---|---|---|---|
| 1-bit / 15 dB | actual | 0.996 / 0.991 / 0.928 | 0.972 | 0.919 |
| 2-bit / 15 dB | actual | 0.999 / 0.992 / 0.934 | 0.975 | 0.927 |
| 1-bit / 0 dB | actual | 0.952 / 0.760 / 0.391 | 0.701 | 0.349 |
| 2-bit / 0 dB | actual | 0.964 / 0.781 / 0.418 | 0.721 | 0.381 |
| 1-bit / 0 dB | 1 (best case) | 0.985 / 0.885 / 0.558 | 0.809 | 0.537 |
| 2-bit / 0 dB | 1 (best case) | 0.994 / 0.915 / 0.584 | 0.831 | 0.572 |
| 3-bit / 15 dB and 0 dB | actual | 1.000 / 0.992 / 0.925 and 0.981 / 0.814 / 0.417 | 0.972 and 0.737 | 0.920 and 0.382 |

What these two tables say (our inference from our numbers):
1. **Under our per-scene peak AGC, extra bits buy almost nothing for weak-path detection.** The median loss is about 2 dB, which is the 1-bit zero-threshold minimum pi/2 (1.96 dB), at 1, 2 and 3 bits alike. The reason: A is set by the strongest beam, which is far above the noise, so the outer thresholds (+-A/2 at 2 bits) sit far from the noise-level samples and only the zero threshold matters for them. Only 3-bit/0 dB moves part of the scenes towards 1 dB (10 % quantile 1.03 dB).
2. **At 15 dB detection is not the bottleneck.** The oracle bound says all three paths are detectable in about 92 % of scenes at 1 and 2 bits. Our measured Pd at 1-bit/15 dB is 0.540 for multi-start ML and 0.626 even for ML started from the TRUE angles (RESULTS_SUMMARY.md). The loss is therefore in estimation accuracy (1-degree tolerance), resolution and the strong paths' own saturation, which the paper's weak-signal theory does not model (it evaluates h_B at the other paths only, not at the path's own signal).
3. **At 0 dB the third path is information-limited.** Its effective SNR is below about 12 dB in 58-62 % of scenes, and its oracle detection probability is about 0.39-0.42 (actual beta). This supports the low 0 dB ceilings we measured (ML from the true angles: 0.439 at 1 bit, 0.540 at 2 bits). The metrics differ, so this is supporting evidence, not a bound on our Pd.
4. **The theory predicts almost no 1-bit vs 2-bit difference** (mean oracle Pd 0.701 vs 0.721 at 0 dB, 0.972 vs 0.975 at 15 dB). Our measured 1-to-2-bit gains are much larger (multi-start ML 0.540 to 0.682 at 15 dB, 0.275 to 0.476 at 0 dB). So the benefit of the second bit lives in the non-weak-signal (estimation) regime that GNOMP's theory does not cover.

## 6. Citable claims (location given; paraphrased, one short quote)

1. **CFAR detection with a fixed, data-independent threshold.** GNOMP's stopping threshold depends only on the prescribed false-alarm probability and the number of DFT bins ((P) Sec. IV, "Main idea"; (P) eq. (39); v4 eq. (44), p.15). *Cite for:* GNOMP estimates the number of paths with a controlled false-alarm rate, whereas our decoders are given L.
2. **The quantisation SNR loss formula.** To keep the same Pd and P_FA, a B-bit system needs 10 log10(SNR_loss) dB more SNR, and the loss depends on sigma, the bit depth and the other sinusoids ((P) Sec. III-A, eq. (31) and the text after it; v4 eq. (30), p.11-12). *Cite for:* the theory behind "weak paths suffer most next to strong ones", and for our per-path loss analysis (Sec. 5.2, our extension). Our reproduction matches their theory to 0.05 dB (Sec. 5.1).
3. **The 4.8 / 2.3 / 0.9 dB numbers belong to one scenario.** They are the measured losses for N = 1024 with a known 6 dB interferer at pi/2 (theory 4.8 / 2.3 / 1 dB). A slightly shifted interferer (pi/2 + 0.1) gives 6.3 / 2.2 / 0.8 dB measured ((P) Sec. V-B, Fig. 6; v4 p.24-25). *Cite for:* quantisation loss is scenario-dependent. Do NOT cite these as generic per-bit losses. Under our peak AGC the median weak-path loss is about 2 dB at 1-3 bits (Sec. 5.2).
4. **Noise variance (the one verbatim quote).** Except at 1 bit, "the noise variance must be estimated in real time from the data" ((P) Sec. IV, paragraph after Algorithm 1; based on Fig. 9). Relatedly, for 1 bit with unknown variance the amplitudes are ambiguous and sigma^2 is set to 1 ((P) Sec. V-E). *Cite for:* our finding that the noise level is not identifiable at 1 bit, and that a quantiser-likelihood method is sensitive to the assumed SNR at >= 2 bits.
5. **Complexity.** GNOMP scales linearly with N, whereas 1bMMRELAX (O(KN^2)) and MVALSE-EP (O(N^2 + Iter N K^3)) scale quadratically ((P) Sec. IV, end; v4 p.21). Measured: 1.70 s vs 7.20 s and 12.84 s ((P) Sec. V-I). *Cite for:* GNOMP is the fast classical few-bit baseline.
6. **Detection gain over 1bMMRELAX.** At Pd = 0.5 and P_FA = 0.01, GNOMP (known or unknown sigma) needs about 1 dB less SNR than 1bMMRELAX in the time-varying-threshold 1-bit setting ((P) Sec. V-I, Fig. 13; v4 p.29). *Cite for:* GNOMP is a state-of-the-art classical 1-bit detector.
7. **GNOMP is not best everywhere.** For 1-bit data an atomic-norm (ANM) method gives a lower debiased NMSE below 17 dB integrated SNR ((P) Sec. V-J, Fig. 14), and GNOMP's frequency MSE departs from the CRB above about 34 dB for 1-3 bits ((P) Sec. V-F, Fig. 10). *Cite for:* the low-SNR and high-SNR gaps that a learned or hybrid method may target.
8. **The multidimensional extension was only claimed.** arXiv v4 footnote 2 (v4 p.5) lists multidimensional and multisnapshot LSE among settings GNOMP can be readily extended to, and says the details are not presented. The published version lists only unknown noise variance, time-varying thresholds and random linear measurements ((P) Sec. I-B). *Cite for:* a 2-D beamspace GNOMP is new work (ours), to be credited as "GNOMP (Zhu et al.) extended by us to 2-D".

## 7. Limitations / what the paper does not show

- **1-D only.** There is no 2-D (angle-angle or range-Doppler) algorithm or experiment. The real data are range-only. The multidimensional claim exists only in arXiv v4 and was dropped from (P).
- **Unknown-noise variant not specified.** Its results are reported but the algorithm is not described, so it is not reproducible. It is 3.6x slower in simulation (Sec. V-I) and 5.7-7.5x slower on real data (`repro_checks.log`).
- **Claimed but not shown in (P).** (P) Sec. I-B claims performance across oversampling and undersampling scenarios and adaptability to random linear measurements, but the only such experiment (vs AST) is in arXiv v4 Sec. V-H, not in (P).
- **No AGC.** In simulation the full scale uses the true amplitudes and sigma (an oracle rule), and quantiser design is out of scope. Real receivers set gamma from data, as our peak AGC does; the effect of data-driven AGC is not studied. Our Sec. 5.2 shows that peak AGC makes 2-3 bits behave almost like 1 bit for weak paths.
- **Asymptotic, oracle theory.** The chi-square laws assume weak signals and large N, and the law on the grid is a conjecture checked only by simulation. The SNR loss and the Pd bound assume the other targets are known exactly. With unknown sigma, the false-alarm rate departs from nominal at 25 dB ((P) Sec. V-I). The theory does not cover the strong-signal estimation regime where, by Sec. 5.2, our 1-to-2-bit gains appear.
- **Accuracy gaps.** GNOMP departs from the CRB above about 34 dB integrated SNR, and ANM beats it below 17 dB at 1 bit (Secs. V-F, V-J).
- **No comparison with multi-start quantised ML** (the same likelihood with a stronger search) or with learned methods. No confidence intervals. Monte Carlo sizes are 100-10^4 per point; Fig. 8 uses 300 runs, so a 1 % false-alarm rate is estimated from about 3 events.
- **Unstated settings:** the target frequency, phase and sigma^2 in Sec. V-B; N in Secs. V-D and V-I; the bit depth in v4's random-measurement test. These make exact reproduction of absolute SNR values impossible (Sec. 5.1).
- **Small inconsistencies:**
  - v4 Algorithm 1, line 3, writes zeta_B where phi_B is meant (checked on the rendered v4 p.22). (P) Algorithm 1 is correct.
  - The unquantised limit printed after (P) eq. (40), lambda = 2|x|^2/(N sigma^2) beta, disagrees with (P) eq. (25) (2N|x|^2/sigma^2) by a factor N^2. We use (25) times beta (our reading).
  - The score in (P) eq. (17) carries an overall minus sign relative to the derivative of (14) with phi_B defined by (18). It cancels in the quadratic Rao statistic (our check of the derivative).
  - Fig. 14's caption says "AST"; the text and Appendix B say ANM.
  - Table I prints the chirp slope as 29.982 MHz/s (presumably MHz/us; UNVERIFIED).
- **No code.**

## 8. Relevance to us

**8.1 Where GNOMP sits relative to what we already have (all checked in the code on 2026-10-16).**
- Our **quantiser-ML** (finetune/adversarial/mlq.py) maximises *the same likelihood* as GNOMP ((P) eq. (43)), with a known, kappa-inflated sigma. It uses a heavier search: Levenberg-Marquardt on all parameters from 13 classical starts plus EM. GNOMP instead does one greedy pass: Rao-test FFT start, single and cyclic Newton, CFAR stop.
- Our **"consistent NOMP"** (finetune/adc/acommon.py, `consistent_nomp`) is a POCS-style "clip the model into the known cell, then rerun NOMP" loop (its docstring says so). **It is not GNOMP.**
- **Corrections needed in unfolded_v2/lit/unfolding_survey.md, row [33]** (not edited here; outside this folder):
  - "the classical quantisation-consistent NOMP the project already uses as a baseline" is **wrong** (see above). GNOMP has never been run in this project.
  - "4.8 / 2.3 / 0.9 dB" is correct as reported but is the measured loss of **one** scenario ((P) Sec. V-B, scenario 1); it is not a generic per-bit loss.
  - "unknown sigma^2 costs about 3.5-4x the runtime": the paper's numbers give 3.55-3.60x in simulation and 5.7-7.5x on real data, and the variant's algorithm is not described.
- The docstrings of unfolded_v2/build/bcommon.py and ucore.py call the EM de-quantisation "GNOMP-style", and DESIGN.md / criteria.json say the design is unfolded from "quantisation-aware NOMP / ML (GNOMP)". GNOMP uses no EM, and the unfolded core uses NOMP's residual-FFT detection, not GNOMP's Rao test or CFAR. The accurate wording is "unfolded from NOMP; EM de-quantisation (our choice); exact quantised likelihood as in GNOMP and our quantiser-ML". The only link to GNOMP's detector is that phi_B equals a normalised E-step residual (Sec. 3.2).
- **GNOMP must become a named baseline.** It is the most-cited fast classical method for exactly this measurement model (few-bit, uniform quantiser, sinusoidal atoms), and a reviewer will ask for it.

**8.2 What to reuse.**
1. The **Rao-test detector and CFAR threshold** ((P) eqs. (19), (36), (39)) as a principled stop rule and order estimate. Our learned stop head did not beat its initialisation (unfolded_v2 findings). GNOMP's analytic threshold needs no training (tau_th = 20.29 for N = 256, P~FA = 0.01).
2. **phi_B (pseudo-measurements) and h_+ (per-sample information)** as network inputs or attention weights. They are the local sufficient statistics of the quantised likelihood around the current model: a "where is the unexplained energy" map and a "how reliable is each sample" map (our inference).
3. The **SNR-loss formula and the oracle Pd** ((P) eqs. (31), (40)) as per-path identifiability diagnostics: they predict which paths no detector can see (Sec. 5.2). Useful for stratifying our results ("detectable" vs "undetectable" third path) and for an honest ceiling at 0 dB.
4. The **cyclic Newton refinement** as a cheap polish after CNN seeds (a "hybrid-lite" ablation against our LM-based hybrid; ML with 1 start costs 93.8 ms vs 482.4 ms with all starts, s6_latency.json).

**8.3 How to compare fairly (protocol; planned, not run).**
- Implement **GNOMP-2D** as in the 2-D adaptation of Sec. 3.9, reusing acommon.adc_cells and mlq.nll_terms (log_ndtr, numerically stable). Name it "GNOMP (Zhu et al., IEEE TAES 2026) extended by us to 2-D beamspace". Unit-test it first in 1-D against Sec. 5.1 (theory curves) and the paper's Sec. V-C example (it should stop after two detections with estimates near 2.198 / 2.401).
- Use the same test banks (12000-12999), the same peak-AGC quantiser and cells, and the same known sigma as our ML. Report GNOMP with the paper's sigma (kappa = 1) and with our tuned kappa.
- Run two order modes:
  - (i) **GNOMP-CFAR**, P~FA = 0.01, N = 256, as in the paper.
  - (ii) **GNOMP-L**: exactly L = 3 detections, threshold ignored, as all our decoders.
- Report **strict Pd** in addition to paper-style Pd. Paper-style Pd drops scenes with fewer than L detections, which would flatter a CFAR method that stops early.
- Same Hungarian matching and 1-degree tolerance, same paired bootstrap (2000 resamples within SNR groups, seed 0), single-thread latency.
- Expected outcome: none yet. This is a planned experiment, not a result.

**8.4 What it implies for our network design.**
- The theory makes the information limit explicit: strong paths push samples far from the thresholds, and h_B then collapses for the weak path. At 0 dB, the third path is below the paper's detectability level in about 60 % of our scenes (Sec. 5.2), so no decoder (learned or classical) should be expected to find it reliably there.
- At 15 dB, detection is easy by the paper's theory (about 92 % of scenes have all three paths detectable), yet our best decoders reach 0.54-0.57 at 1 bit. The open headroom is therefore estimation and decision, not detection. This is consistent with our gap between ML from true angles and multi-start ML (0.626 vs 0.540 at 1-bit/15 dB) and with the survey lesson.
- A learned component should either (a) supply better starting points, where our hybrid already wins at 1-bit/15 dB (+0.032*), or (b) change the information content. Sec. 5.2 shows that peak AGC wastes the extra bits for weak paths, so the AGC choice or known dither / time-varying thresholds (the paper's Sec. V-I setting) are the levers that change h_B. This is a hypothesis to test, not a result.
- At 1 bit the noise level cannot be identified (the paper agrees). A sigma-free network is therefore not an advantage over GNOMP at 1 bit, because GNOMP simply sets sigma^2 = 1. At >= 2 bits, robustness to sigma mismatch is a legitimate advantage: both the paper (Fig. 9) and our ML (-0.04 to -0.07 Pd at 5 dB mismatch) show sensitivity.

## 9. "If a reviewer asks..."

**Q1. Why is GNOMP, the state-of-the-art classical method for few-bit line spectra, missing from your comparison?**
Honest answer today: it has not been run yet. Our strongest classical baseline maximises the same quantised likelihood ((P) eq. (43)) with a heavier multi-start LM + EM search, which we expect to be at least as accurate as one greedy pass. That is an expectation, not a result. GNOMP is 1-D; we will add GNOMP-2D (our 2-D extension, credited) in both CFAR and known-L modes, with strict Pd. Until then we cannot claim to beat GNOMP.

**Q2. GNOMP does not need to know the number of paths, but your methods are given L = 3. Is that fair?**
It favours our methods, not GNOMP. That is why both GNOMP-L (also given L = 3) and GNOMP-CFAR will be reported, and strict Pd counts missed paths. Any claim against GNOMP will be made against GNOMP-L, the stronger of the two under our metric.

**Q3. The paper reports only a few dB of quantisation loss (pi/2, about 2 dB, at 1 bit). Why does NOMP collapse to Pd 0.19 at 1-bit/15 dB in your setting?**
- The paper's loss is a weak-signal **detection** loss. Applied to our scenes (Sec. 5.2), it is indeed small (median about 2 dB) and says all three paths are detectable in about 92 % of 15 dB scenes.
- Plain NOMP fails for a different reason: it ignores quantisation, so it fits the distorted samples as if they were clean (the paper's motivation, (P) Sec. I).
- Even quantisation-aware ML started from the true angles reaches only 0.626 at 1-bit/15 dB, so the remaining loss is in estimation accuracy within 1 degree and strong-path saturation, which the paper's theory does not model. That is the honest explanation.

## 10. Glossary (plain English)

- **LSE&D (line spectral estimation and detection):** find how many sinusoids are in the data, and their frequencies and amplitudes.
- **Few-bit / low-resolution ADC:** a converter that records each real number with only 1-4 bits, so the receiver only knows which interval ("cell") the value fell in.
- **Full-scale range gamma:** the largest value the quantiser can represent. Beyond it, values are clipped into the end cells.
- **AGC (automatic gain control):** setting gamma from the data. The paper does not do this; we use the per-scene peak.
- **Probit / cell-probability likelihood:** the probability that a Gaussian value lands in its observed cell, Phi(upper) - Phi(lower). The paper's log-likelihood sums its logarithm.
- **Pseudo-measurements phi_B:** a per-sample number computed from the cell that acts like a "de-quantised residual". GNOMP runs its FFT on these instead of on the raw samples.
- **h_B (Fisher information per sample):** how much a quantised sample still tells us. 1 means as good as unquantised, 0 means useless.
- **Rao test (score test):** decides "is there one more sinusoid?" using only the slope of the log-likelihood at "no sinusoid", so the new sinusoid never has to be fitted.
- **GLRT / Wald test:** the two other classical tests; same asymptotic performance, but they need the ML estimate.
- **CFAR (constant false alarm rate):** the false-alarm probability stays at a chosen value (here 1 % per spectrum) regardless of the data.
- **P~FA vs per-cell P_FA:** the paper's false-alarm probability is for the whole spectrum of N bins, which is about N times the per-bin value.
- **Marcum Q-function Q1:** gives the detection probability when the statistic is non-central chi-square.
- **Non-centrality lambda:** how far the target pushes the statistic away from noise-only behaviour; twice the effective integrated SNR.
- **Integrated SNR:** SNR after coherently adding all N samples: 10 log10 N dB above the per-sample (time-domain) SNR.
- **SNR loss:** how many dB more SNR a B-bit receiver needs to detect as well as an ideal one.
- **Thresholds zeta (in the theory):** the already-known part of the signal (the other targets). It shifts each sample relative to the quantiser thresholds.
- **Oracle bound:** a bound computed as if the other targets were known exactly.
- **Straddle factor beta_g:** the fraction of a target's energy that falls into the nearest grid bin (1 if exactly on the grid).
- **Oversampling factor gamma_os:** how much finer the starting grid is than the DFT grid (4 here).
- **Newton step:** one update of a parameter using the first and second derivatives of the log-likelihood.
- **AM (alternating maximisation):** update the frequency with the amplitude fixed, then the amplitude with the frequency fixed.
- **Cyclic refinement / BCD (block coordinate descent):** revisit every detected path in turn, holding the others fixed.
- **Spurious-component suppression:** re-test a suspiciously small component and delete it if it no longer passes the CFAR test.
- **CRB (Cramer-Rao bound):** the smallest variance any unbiased estimator can reach.
- **NOMP:** Newtonized orthogonal matching pursuit (Mamandipoor, Ramasamy, Madhow, IEEE TSP 2016). GNOMP reduces to NOMP when B is infinite.
- **NOMP-CFAR:** the same group's CFAR version of NOMP for unquantised data (Xu et al., IEEE TAES 2023); its radar data are reused in (P) Sec. V-K.
- **1bMMRELAX, MVALSE-EP, ANM/AST:** competitor methods: a majorisation-minimisation RELAX for 1-bit signed data (Ren et al., IEEE TSP 2019, ref. [20] of (P)), a multisnapshot variational Bayesian method with expectation propagation (cited in (P) as [10], [22], [23]; [10] is Zhang, Zhu, Xu, IEEE TAES 2023), and atomic-norm convex methods.
- **Signed measurements with time-varying thresholds:** 1-bit data where each sample is compared with a known, changing threshold instead of zero.
- **NPI (noise power increase):** how much the true noise power exceeds the assumed one, in dB.
- **Dynamic range (DR):** how much weaker the second target can be than the first and still be detected (v4 only).
