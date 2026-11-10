# Hu, Sun & Zhang (2025): one-bit, single-snapshot DOA with an unrolled MM network (SBRI-Net / SBRI-X-Net). Explainer

Prepared 2026-10-16 for the ADC programme. Folder: `D:/ai_ml_project/paper/evidence/adc_program/papers/`.
Every number we computed ourselves comes from `check_hu2025_math.py`, with output in `check_hu2025_math.log` (sections A-K of that log are cited as "check A" ... "check K"), and from `check_hu2025_scale.py` / `check_hu2025_scale.log` ("check L", added on the resumed pass).
Resumed pass (2026-10-16, after the usage-limit cut-off): every section below was re-checked against the published text and the rendered figures (Fig. 6, Fig. 8(c), Fig. 9, Tables I-II re-rendered from the published PDF); the arXiv record was re-fetched. Corrections and additions from that pass are marked "[resumed pass]".
Final pass (same day, against the cached published text `hu2025_jstsp_published.txt` and `fig/page919_tables.png`, `fig/fig6_d_f.png`): corrected claim C6 (it overstated the 10-element and 3-target results), added the full author names, flagged the unprinted unit of the hit-rate threshold, corrected the Fig. 6(d) reading at b = 0.1, noted that the 3-target test does not say whether networks were retrained, and noted that the conclusion's convergence and hardware-efficiency claims are not backed by measured numbers.
Our project numbers come from `deep_research/RESULTS_SUMMARY.md` (verified, fresh seeds).
Copies of the paper text and figure renders are in `C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/hu2025/`.

## 0. Verified bibliographic entry and what was read

**Yunqiao Hu, Shunqiao Sun, and Yimin D. Zhang (cited in the text as Y. Hu, S. Sun, Y. D. Zhang), "Model-Based Learning for DOA Estimation With One-Bit Single-Snapshot Sparse Arrays," *IEEE Journal of Selected Topics in Signal Processing*, vol. 19, no. 6, pp. 909-925, Sept. 2025, doi: 10.1109/JSTSP.2025.3627141. arXiv:2502.17473 [eess.SP].**

- Verified on the first page of the published PDF (an author-hosted copy at sites.ua.edu that carries the IEEE Xplore stamp). The header reads vol. 19, no. 6, September 2025, and the pages run 909-925.
  - Dates printed there: received 9 Feb 2025; revised 23 Jun, 22 Aug and 2 Oct 2025; accepted 20 Oct 2025; published 31 Oct 2025; current version 31 Dec 2025. The issue is dated September 2025 even though the paper went online on 31 Oct 2025; that is how it is printed.
- The arXiv abstract page has two versions (v1 15 Feb 2025, v2 21 Oct 2025). Its comment field says the paper was accepted by IEEE JSTSP. Re-fetched on the resumed pass (2026-10-16): same two versions, same comment, no journal-ref field; the only DOI shown there is the arXiv DataCite DOI 10.48550/arXiv.2502.17473.
- The IEEE DOI 10.1109/JSTSP.2025.3627141 is printed on the published first page (Digital Object Identifier line).
- An earlier version of the work appeared at Asilomar 2024 (doi 10.1109/IEEECONF60004.2024.10942609). This is stated in the published footnote and is the paper's own ref. [1].
- **What we read in full:**
  - the arXiv v2 HTML: all sections, Appendices A-C and the references;
  - the published PDF: the text of all 17 pages, plus Figs. 1-10 viewed as rendered images.
  - Sections IV-B to VI of the published text match arXiv v2. Page numbers below are the published page numbers.
- **What we did not access:**
  - The supplementary material at the DOI. Its content is UNVERIFIED.
  - Source code: the paper has no code link, and a web search found none.
- **Copyright note:** this document contains exactly one verbatim quote, in claim C4. Every other statement about the paper is a paraphrase that points to its page, section and equation so it can be checked.

## 1. Plain-English summary

Picture an antenna array whose receivers keep only one bit for each antenna: is the real part positive or negative, and is the imaginary part positive or negative? They record this once, so there is no averaging over time.

From those plus/minus values the authors want the directions of two or three radio sources. They do it in four steps:
1. They write down how likely each pattern of signs is, given the source directions. Noise flips some of the signs.
2. They add the belief that only a few directions out of a fine list of candidates are active.
3. They solve the result with a loop. Each pass turns the signs into a "best guess" of the unquantised signal and then solves a weighted least-squares problem that pushes most candidate directions to zero. This loop is called SBRI. A variant, SBRI-X, replaces the Gaussian S-shaped curve with a logistic S-shaped curve that has two tuning knobs.
4. They turn a fixed number of passes into the layers of a neural network. Small convolutional networks replace the expensive matrix inversion, and small fully connected layers replace the step that corrects off-grid angles. The network also learns the steepness of the S-curve.

The simulations use two sparse arrays, with 18 and 10 antennas. They show:
- For sources exactly on the grid, the network is about as good as its loop.
- For sources between grid points, with three sources, and with the sparser array, the network is clearly better than its loop at low SNR.
- The network still works at SNRs above its training range (35-50 dB). No test below the training range (under 0 dB) is reported.

## 2. The problem and the system model

**Their setting.** The paper uses:
- an M-element sparse linear array (SLA);
- K far-field, narrowband sources;
- one time sample (a "single snapshot");
- the real and imaginary parts of every antenna quantised to one bit, with a zero threshold;
- complex white Gaussian noise.

The goal is to estimate the K directions of arrival (DOAs) in [-60 deg, 60 deg].

- Eq. (1) (p. 911): ybar = csgn(A(theta) s + n). The steering vector, eq. (2), has entries exp(j 2 pi d_m sin(theta)/lambda).
- Eq. (3) (p. 912), the on-grid form: ybar = csgn(Acal x + n). Acal is an M x N dictionary of steering vectors on a uniform angle grid, and x has only K non-zero entries.
- Eqs. (4)-(7) (p. 912), the off-grid form: a first-order Taylor expansion a(theta) ~ a(grid point) + b(grid point) (theta - grid point). The dictionary becomes C(beta) = Acal + Bcal diag(beta). Here beta_n is the gap between a true angle and its nearest grid point, and beta_n = 0 at inactive grid points.

**Symbol map (theirs -> ours).** Our forward model is in `deep_research/stress/stress_sim.py`.

| Paper | Meaning in the paper | Our project | Note |
|---|---|---|---|
| ybar in C^M | one-bit data csgn(.) of the M antenna outputs | Q_b(Y), 16 x 16 complex (256 complex = 512 real numbers) | At b = 1 our ADC output is exactly (A/2) csgn(Y), with the same sign(0) = +1 convention (check G: max difference 0.0). So the paper's model **is** our 1-bit case, up to the dictionary. |
| y = A(theta) s + n | unquantised snapshot | Y = W^H H F + Z | vec(Y) = (F^T kron W^H) vec(H) + vec(Z) |
| a(theta), entries e^{+j pi p_m sin theta} | SLA steering vector | atom vec(W^H a_r(u) a_t(v)^H F); a_r[n] = e^{-j pi u n}/4 | identity verified numerically (check K, error 1e-16) |
| Acal (M x N) | dictionary on a 1-D angle grid | 256 x (N_u N_v) dictionary on a 2-D (u, v) grid | Acal^H vec(Y) = a_r(u)^H Htil a_t(v), with Htil = W Y F^H (check K). Every matched-filter step is therefore one zero-padded 2-D FFT of Htil. |
| (no codebooks) | fully digital array | W, F: 16 x 16 DFT combiner / precoder | Their snapshot is one sample per antenna; ours is one sample per beam pair. Both are single-snapshot (no averaging over repeated snapshots). |
| (no MIMO channel) | | H = sqrt(256) sum_l alpha_l a_r(psi_l) a_t(phi_l)^H | |
| K | number of sources (2 or 3). **Also** the number of unrolled layers | L = 3 paths | name clash inside the paper |
| s, x | source amplitudes / sparse coefficients on the grid | alpha_l (path gains) | training draws Re, Im ~ U(0.5, 1) (Sec. IV-B) |
| theta_k (enters via sin theta) | 1-D DOA, broadside convention | (u_l, v_l) = (cos psi_l, cos phi_l): AoA **and** AoD | 1-D versus our joint 2-D problem; the exponents have opposite sign conventions |
| n ~ CN(0, sigma^2 I) | white noise | Z, per-component std sqrt(10^(-SNR/10)/2) | |
| csgn (threshold 0) | one-bit ADC | b-bit mid-rise quantiser, b in {1, 2}, step 2A/2^b | **2-bit is not covered by the paper** |
| (no AGC) | the sign does not depend on scale | A = per-scene peak max(abs Re, abs Im) | At 1 bit, A only rescales the output |
| "input SNR" | training 0-30 dB, test 0-50 dB | -10..25 dB | **The paper never defines SNR** (UNVERIFIED convention) |
| a, b | shape parameters of sig(x) = 1/(1 + a e^{-b x}) | **our b = ADC bits** | name clash with our notation |
| beta, r | off-grid gaps and grid step (r = 2 deg in the off-grid runs) | continuous refinement (Newton in NOMP, Levenberg-Marquardt in the quantiser ML) | |
| eps | per-measurement noise offsets estimated by SBRI-X | none: our quantiser ML integrates the noise out | eps_0 is also the stopping tolerance |
| xtil = x/sigma, xhat = sqrt(2) x/sigma | amplitudes in noise units | alpha/sigma | the paper never needs sigma explicitly |
| hit rate | trial counts as a success only if all K errors are <= 2 deg | strict Pd (1 deg per angle component, Hungarian matching) | see Sec. 8.4 |

Other name clashes to warn the audience about:
- Phi is both the normal CDF and the Lyapunov function of Theorem 1.
- eta is both the smoothing constant and the function -phi/Phi in Lemma 3.
- beta is both the grid gap and a Lyapunov weight.
- The calligraphic A is their dictionary, not our AGC level A.

## 3. The method step by step

### 3.1 Big picture

There are two classical loops, SBRI and SBRI-X ("Sparse Bayesian Reweighted Iterative"). Each comes in an on-grid and an off-grid version (Algorithms 1-4, pp. 913-917). Each loop is then unrolled into a network: SBRI-Net and SBRI-X-Net (Sec. IV, pp. 916-919).

### 3.2 SBRI, on-grid (Sec. III-A, pp. 912-914; Algorithm 1)

1. **Likelihood (8)-(9).** Every observed sign is modelled as Phi(sign x prediction / noise std), separately for the real and imaginary parts.
   - In words: each bit is a coin whose bias depends on how far the noiseless prediction is from zero, measured in noise standard deviations.
   - Rescaling to xhat = sqrt(2) x / sigma removes sigma from the formula. One-bit data only carry the ratio of signal to noise, not the absolute scale.
2. **Prior (10).** p(xhat) is proportional to prod_i exp(-gamma abs(xhat_i)^alpha / alpha), with 0 < alpha <= 1.
   - In words: almost every grid point carries no source. alpha = 1 gives a Laplace prior; a smaller alpha is stricter about sparsity.
3. **Objective (12).** The negative log-likelihood plus the sparsity penalty. It is not convex.
4. **Majorisation (13)-(14).** The likelihood part is replaced by a quadratic, 1/2 ||Acal xhat - v||^2, around the current estimate.
   - The "pseudo-measurement" is v = ybar (.) (d + phi(d)/Phi(d)), where d = ybar (.) (current prediction) and (.) is the element-wise product. The paper writes this as d - I'(d), with I'(x) = -phi(x)/Phi(x).
   - In words: v is the best guess of the unquantised measurement, given its observed sign and the current prediction.
   - **Our check I:** v equals the mean of a Gaussian truncated to the observed sign (max difference 1.5e-14). That is exactly the E-step of EM for one-bit data, the same quantity our `mlq.em` E-step computes ("truncated-normal mean", `finetune/adversarial/mlq.py` docstring, line 10).
5. **Smoothing (15).** abs(x)^alpha is approximated by (abs(x)^2 + eta)^(alpha/2), with eta = 1e-6. This makes the penalty differentiable.
6. **IRLS step (16)-(19).** xhat <- (Acal^H Acal + gamma Lambda(xhat))^-1 Acal^H v, with Lambda = diag((abs(xhat_i)^2 + eta)^(alpha/2 - 1)).
   - In words: a ridge regression whose penalty is very large for grid points that are currently small, so those points are pushed to zero.
7. **Regulariser update (20).** gamma <- gamma0 ||xhat||_2, with gamma0 = 1.
   - Initialisation: xhat0 = Acal^H ybar / ||Acal^H ybar|| (a normalised beam scan).
   - Read-out: the paper never says how the DOAs are taken from xhat. The most likely rule is the K largest peaks (UNVERIFIED).

### 3.3 SBRI, off-grid (Sec. III-B, pp. 913-915; Algorithm 2)

- The gaps get a uniform prior, beta ~ U(-r/2, r/2).
- Each pass alternates three steps:
  - pseudo-measurement w, eq. (25), computed with C(beta);
  - coefficient update with C(beta), eq. (26);
  - least-squares gap update, eq. (27): beta = [Re((Bcal^H Bcal)^* o xhat xhat^H)]^-1 Re(diag(xhat^*) Bcal^H (w - Acal xhat)).
- The paper prints the two factors in (27) side by side. The derivation (30)-(31) only works if "o" is the element-wise (Hadamard) product; that is our reading.
- After the update, the gaps are clipped to [-r/2, r/2], eq. (54).

### 3.4 SBRI-X, the "augmented" loop (Sec. III-C, pp. 914-917; Algorithms 3-4)

**Change 1: a logistic likelihood, eqs. (32)-(34).** The probability that a bit agrees with the prediction y is sig(ybar y), where sig(x) = 1/(1 + a e^{-b x}).
- In words: the Gaussian S-curve is replaced by a logistic S-curve with an adjustable slope b and offset a. Fig. 1 (p. 915) compares b = 0.1-2 with the normal CDF.
- **Our check E:** with a = 1, the logistic curve fits Phi best at b = 1.702 (sup error 0.0095). A smaller b gives a flatter curve, which acts like more noise. *If* x stays in the noise-normalised units x/sigma of eq. (35), the learned b of about 0.19 behaves like a Gaussian likelihood whose noise std is about 8.9 times the nominal one.
- **Our check L [resumed pass]:** that "if" matters. The data term of (39) depends on b only through b x and b eps: (b, x, eps) and (1, b x, b eps) give identical values (difference 0.0 on a paper-like instance, `check_hu2025_scale.log`). Only the sparsity prior changes. So b is confounded with the amplitude scale. In SBRI-X-Net the spectrum labels are |s_k| in amplitude units (eq. (74)), not x/sigma, and the CNNs can rescale x freely. The learned b = 0.19 is therefore a scale convention of the network as much as a noise level. Do not present it as "the network learned that the noise is 9x larger".
- **Our check D:** identity (33), and the requirement that the two outcome probabilities add up to 1, hold **only when a = 1**. For a = 0.5, b = 1.7 their sum ranges from 1.0003 to 1.3333. In words: a != 1 shifts the S-curve sideways by ln(a)/b.

**Change 2: noise offsets, eqs. (35)-(38).** SBRI-X estimates xtil = x/sigma **and** the noise realisation eps of each measurement jointly. As printed, (38) puts no prior on eps.

**Objective (39).** As printed, the data terms are indexed i = 1..N and the same term appears twice. The intended meaning is a sum over the M real parts and the M imaginary parts.

**Majoriser (40)-(44).** With f(s) = ln(1 + a e^{-b s}), each log-loss term is replaced by a parabola with curvature a b^2/(a+1)^2. This gives the "target" g(.) of eq. (44).
- In words: a parabola that touches the curve at the current point and lies above it everywhere, so minimising the parabola can only lower the loss.
- **Our checks B and C:** the true maximum of f'' is b^2/4, reached at s = ln(a)/b. It equals the printed bound only when a = 1.
  - For a != 1 the parabola dips **below** the curve: the minimum gap is -0.031 for (a, b) = (0.5, 1.7) and -0.177 for (3, 1).
  - So the MM descent guarantee holds for a = 1, which is what SBRI-X-Net uses. It does not hold for the a != 1 runs of Fig. 6(e)-(f) and Fig. 7, whose best setting is a of about 0.5.

**Updates (48)-(49).**
- xtil <- (Acal^H Acal + gamma (a+1)^2/(a b^2) Lambda)^-1 Acal^H (Acal xtil + g)
- eps <- eps - Acal (xtil_new - xtil) + g

The off-grid version is eqs. (50)-(54) (Algorithm 4).

### 3.5 The unrolled networks (Sec. IV-A, pp. 916-918; Figs. 2-5)

All complex quantities are first stacked into real vectors and matrices, eqs. (55)-(59).

- **SBRI-Net (60)-(61).**
  - The pseudo-measurement layer is kept as fixed physics, with no weights.
  - The matrix inverse is dropped. Instead, xhat^k = F_conv^k(Acal_r^T vhat^k): a CNN applied to the back-projection.
  - Each stage has J = 4 one-dimensional conv layers with (in, out, length) = (1,21,21), (21,11,11), (11,5,5), (5,1,3) (Sec. IV-E, p. 919). "Length" is most likely the kernel size (UNVERIFIED).
  - Off-grid version, eqs. (62)-(65): beta^k = F_FC^k(xhat^k). The FLOP count on p. 919 implies three FC layers per stage.
- **SBRI-X-Net (66)-(69).**
  - uhat^k = Acal_r xhat^{k-1} + g_r(xhat^{k-1}, epshat^{k-1}), with learnable a_k and b_k, eq. (67).
  - xhat^k = F_conv^k(Acal_r^T uhat^k).
  - The eps update, eq. (69), is unchanged from the loop.
  - Off-grid version: eqs. (70)-(73).
- **What is learned:**
  - In each stage, the CNN replaces the regularised inverse (Acal^H Acal + gamma Lambda)^-1. It learns to sharpen the blurry beam scan and make it sparse.
  - The FC layers replace the least-squares gap update.
  - The logistic slope b is learned. Letting a and b move together gave numerical instabilities (NaN), so a is fixed at 1.0. Over 100 epochs b fell from 1.0 to about 0.19 (Fig. 9, p. 921).
- **What stays fixed (physics):** Acal, Acal^T and Bcal, the pseudo-measurement formulas, and the eps update.
- **Not stated in the paper (all UNVERIFIED):** the number of stages K, the activation functions, the output nonlinearity (the BCE loss needs outputs in (0, 1)), whether a_k and b_k are shared across stages (Fig. 9 plots a single b), whether the BCE loss is applied to the last stage only, and the DOA read-out rule. [resumed pass] Also not stated: the value of the prior exponent alpha used in any experiment (only 0 < alpha <= 1 is given), and which (a, b) the iterative SBRI-X uses in Figs. 8 and 10.
- **Table II (p. 919), re-checked [resumed pass]:** it lists SBRI-Net and SBRI-X-Net as handling single or multiple snapshots, and all four iterative variants as single-snapshot only. Table I lists the costs of SBRI and SBRI-Net only. The SBRI-X-Net formulas are given in the text of Sec. IV-E.

### 3.6 Training (Secs. IV-B to IV-D, pp. 918-919)

- **Arrays:** an 18-element SLA at (lambda/2)[0,1,2,3,4,7,8,...,19] and a 10-element SLA at (lambda/2)[0,3,4,5,6,7,11,16,18,19].
- **Sources:** 2, in a field of view of [-60, 60] deg.
  - On-grid: random integer-degree DOAs.
  - Off-grid: a 2-deg grid with gaps ~ U(-1, 1) deg.
  - Amplitudes: Re and Im ~ U(0.5, 1).
- **Labels, eqs. (74)-(75):** the spectrum label is |s_k| at the true grid point; the gap label is |beta_k|.
- **Data:** 1,000,000 samples at SNR 0, 5, ..., 30 dB, split 90 % training / 10 % validation.
- **Losses:** BCE on the spectrum, eq. (76). Off-grid stage 2 adds an MSE on beta, eq. (77).
- **Schedule:**
  - On-grid: 100 epochs, end to end.
  - Off-grid: 100 epochs with the gap layers frozen, then 100 epochs with all layers trained.
  - Batch size 64, Adam, learning rate 1e-3.
  - Hardware: Intel i9-9820X CPU with 2 x RTX 2080 GPUs.
- **Puzzles in the printed text:**
  - |s_k| lies in [0.71, 1.41], so it is not a valid BCE target unless it is normalised, and normalisation is not stated.
  - |beta_k| throws away the sign of the gap, and the paper does not say how the sign is learned.
  - Both are UNVERIFIED without the code.

### 3.7 Pseudo-code, detailed enough to re-implement

Items marked [ASSUMED] are not specified in the paper.

```
# ---------- (1) iterative SBRI-X, off-grid (Algorithm 4, eqs. 50-54, 20) ----------
input : ybar (M complex, entries +-1 +-1j), Acal (M x N), Bcal = dAcal/dtheta (M x N), grid theta_grid (step r)
        a, b (sigmoid shape), alpha in (0,1], eta = 1e-6, gamma0 = 1, Tmax, tol, K (number of sources)
x     = Acal^H ybar / ||Acal^H ybar||_2 ;  beta = 0 ;  eps = 0 [ASSUMED: eps init not stated] ; gamma = gamma0
for k in 0 .. Tmax-1:
    C    = Acal + Bcal diag(beta)
    z    = C x + eps
    g    = (a+1)^2 Re(ybar) / (b exp(b Re(ybar) Re(z)) + a b)  +  1j (a+1)^2 Im(ybar) / (b exp(b Im(ybar) Im(z)) + a b)   # (51)
    Lam  = diag((|x|^2 + eta)^(alpha/2 - 1))                                                                         # (19)
    x_new = solve(C^H C + gamma (a+1)^2/(a b^2) Lam ,  C^H (C x + g))                                              # (50)
    eps  = eps - C (x_new - x) + g                                                                                  # (52)
    dx   = x_new - x
    beta = solve(Re(conj(Bcal^H Bcal) o (dx dx^H)) , Re(diag(conj(dx)) Bcal^H (g - Acal dx)))   # (53), o = Hadamard (our reading)
    beta = clip(beta, -r/2, r/2)                                                                                    # (54)
    gamma = gamma0 ||x_new||_2                                                                                      # (20)
    stop if ||x_new - x|| small (tol) ;  x = x_new
DOA = theta_grid[n] + beta[n] for the K largest |x[n]|   [ASSUMED read-out]
# SBRI (Algorithms 1-2) is the same loop with g replaced by the probit pseudo-measurement v = ybar (.) (d + phi(d)/Phi(d)),
# no eps, the x-update (C^H C + gamma Lam)^-1 C^H v, and the gap update (27).

# ---------- (2) SBRI-X-Net forward (eqs. 55-59, 66-73) ----------
real-stack: yr = [Re ybar; Im ybar] (2M), Ar = [[Re A, -Im A],[Im A, Re A]] (2M x 2N), Br likewise
x = Ar^T yr ; eps = 0 [ASSUMED] ; beta = 0
for k in 1..K_layers [K_layers not stated]:
    Ck  = Ar + Br diag(beta)                                     # on-grid version: Ck = Ar, no beta
    g   = (a_k+1)^2 yr / (b_k exp(b_k yr (.) (Ck x + eps)) + a_k b_k)   # (67)/(70); in practice a_k = 1 fixed, b learnable
    u   = Ck x + g                                               # (66)/(70)
    x_n = Fconv_k(Ck^T u)    # 4 x Conv1d: (1->21, k=21), (21->11, k=11), (11->5, k=5), (5->1, k=3) [activations ASSUMED]
    eps = eps - Ck (x_n - x) + g                                 # (69)/(72)
    beta = FFC_k(x_n)        # 3 FC layers (off-grid only); paper's (73) feeds x^{k-1}, (65) feeds x^k: inconsistent
    x = x_n
output: spectrum x (2N real; magnitude per grid point [ASSUMED]) and beta

# ---------- (3) training ----------
data: 1e6 scenes, 2 sources, SNR in {0,5,...,30} dB ; targets s* (eq. 74), beta* (eq. 75)
on-grid : 100 epochs, loss BCE(x, s*)
off-grid: 100 epochs BCE with FFC frozen ; then 100 epochs BCE + MSE(beta, beta*) with all layers ; Adam 1e-3, batch 64
```

## 4. Assumptions

| Question | What the paper assumes | Where |
|---|---|---|
| Known noise variance? | **No.** The amplitudes are estimated in noise units (x/sigma, eqs. (9) and (35)). However, the likelihood shape is fixed: probit for SBRI, logistic with a hand-tuned (a, b) for SBRI-X, and a learned b for SBRI-X-Net. A fixed shape works like an implicit assumed noise level (check E). | pp. 912, 915, 921 |
| Known number of sources? | Effectively **yes**. The data are built with 2 sources, the hit rate needs all K angles, and no model-order step is described. The paper states that training data use 2 signals (p. 918). For the 3-target test (p. 922) it does not say whether the networks were retrained (UNVERIFIED). | pp. 918, 922 |
| Dithering / thresholds | **None.** The threshold is zero (csgn), and the noise itself acts as a dither. Time-varying thresholds are only cited ([26], [50]). | eq. (1) |
| AGC | Not applicable: the sign does not depend on scale, so the amplitude scale is lost. | eq. (1) |
| Snapshots | **One.** Table II says the networks also support multiple snapshots, but no multi-snapshot result is shown (UNVERIFIED). | p. 919 |
| Grid | On-grid tests: targets exactly on a 1-deg integer grid. Off-grid: a 2-deg grid plus a first-order Taylor correction; training gaps ~ U(-1, 1) deg. | pp. 912, 918 |
| Array | Ideal SLA manifold with half-wavelength base spacing. No calibration errors, coupling or other impairments are simulated. | p. 918 |
| Amplitudes | Re and Im ~ U(0.5, 1), so every phase lies between 26.6 and 63.4 deg (first quadrant only). This is a narrow training distribution. | p. 918 |
| Noise | White complex Gaussian. The "input SNR" is never defined. | pp. 912, 918 |
| Test geometry | **Fixed** target positions: (-30, 30), (-10, 20), (-10.28, 20.56) and (-10.28, 0.25, 20.56) deg. Trials vary only the random draws (noise, presumably amplitudes). | pp. 920-922 |
| Theory | Theorems 1-2 and Corollary 1 assume RIP of order 2s with delta_2s < 1. **Our check J:** for the paper's own 18-element dictionary, delta_4 >= 2.86 (complex x) or 2.51 (real x) on the 1-deg grid, and >= 2.45 or 1.47 on the 2-deg grid. The assumption is therefore **violated**, and the bounds do not apply to the reported experiments. | pp. 913, 916 |

## 5. Reported results

### 5.1 Exact numbers stated in the text

- 1,000,000 training samples; SNR 0-30 dB in 5-dB steps; 90/10 split (p. 918).
- Batch 64, Adam with learning rate 1e-3; 100 epochs on-grid, 100 + 100 epochs off-grid; J = 4 conv layers per stage (pp. 917, 919).
- Hit-rate threshold: the text gives a numeric threshold of 2 on **all** angles of a trial (p. 919). The unit is not printed in the text layer; we read it as degrees, which fits the 1-deg grid (UNVERIFIED unit). RMSE is taken over successful trials only (p. 919).
- Trials: 1,000 for Fig. 6 and 5,000 for Figs. 8 and 10 (pp. 919-922).
- SBRI-X tuning at 10 dB: the best RMSE is near (a, b) = (0.5, 1.7) (Fig. 7). The gain from a saturates for a >= 3 (Fig. 6(e)). Small b (0.1 or 0.5) converges within about 10 iterations (Fig. 7(c)) (pp. 920-921).
- Training b with a fixed: a is fixed at 1.0 and b starts at 1.0 (p. 921).
- Complexity (Table I, Sec. IV-E, p. 919):
  - SBRI: (2/3) N_iter N^3 + 2 N_iter M N^2 (on-grid).
  - SBRI-Net: K(8NM + 13,088N) on-grid and K(16NM + 13,088N + 12N^2) off-grid.
  - SBRI-X-Net: K(12NM + 13,088N) on-grid and K(20NM + 13,088N + 12N^2) off-grid.
  - **No measured run times are reported**, only FLOP formulas.

### 5.2 Numbers read from the figures

The paper has **no results table**; every accuracy number exists only as a plot. The values below were read by eye from the published figures, rendered at 280-500 dpi. Expect about +-0.01-0.02 error on hit rates and about +-5 % on RMSE. They are approximate, not exact.

**(a) On-grid, 18-element SLA, targets (-10, 20) deg, 5,000 trials: hit rate (Fig. 8(c), p. 921)**

| SNR (dB) | SBRI | **SBRI-Net** | SBRI-X | **SBRI-X-Net** | 1-bit SLIM | 1-bit LIKES | 1-bit SPICE |
|---|---|---|---|---|---|---|---|
| 0 | 0.30 | 0.56 | 0.71 | 0.70 | 0.55 | 0.29 | 0.35 |
| 5 | 0.59 | 0.88 | 0.96 | 0.94 | 0.86 | 0.57 | 0.53 |
| 10 | 0.82 | 0.97 | 1.00 | 0.99 | 0.96 | 0.80 | 0.70 |
| 15 | 0.92 | 0.99 | 1.00 | 1.00 | 0.98 | 0.92 | 0.75 |
| 20-50 | 0.96-0.97 | 0.99 | 1.00 | 1.00 | 0.99 | 0.96-0.98 | 0.78 |

RMSE in Fig. 8(b) (log axis labelled "rad"):

| SNR | SBRI-X | SBRI-X-Net | SBRI-Net | SLIM | LIKES | SBRI | SPICE |
|---|---|---|---|---|---|---|---|
| 0 dB | 0.104 | 0.106 | 0.113 | 0.127 | 0.136 | 0.138 | 0.185 |
| 15 dB | 0.049 | 0.046 | 0.047 | 0.061 | 0.061 | 0.064 | — |
| 25-50 dB | 0.039-0.042 | 0.037-0.042 | 0.048-0.050 | — | — | — | — |

At 25-50 dB, SBRI-X-Net is at or below SBRI-X; the two are about equal at 35 dB.

**(b) The learned slope (Fig. 9, p. 921).** a = 1.0 throughout. b is about 0.92 (epoch 1), 0.71 (epoch 10), 0.26 (20), 0.22 (30), 0.20 (40-60) and 0.19 (100).

**(c) Off-grid, 5,000 trials: hit rate (Fig. 10, p. 922)**

| Setting | SNR | OGIR | SLIM | SBRI | SBRI-X | LIKES | SPICE | CNN-DNN | **SBRI-Net** | **SBRI-X-Net** |
|---|---|---|---|---|---|---|---|---|---|---|
| 18-el, 2 targets (10(b)) | 0 | 0.48 | **0.56** | 0.40 | 0.36 | 0.29 | 0.02 | ~0.53 (overlapping markers) | 0.525 | 0.535 |
| | 5 | 0.795 | **0.85** | 0.83 | 0.795 | 0.56 | 0.07 | ~0.82 | 0.815 | 0.835 |
| | 20-50 | 0.99-1.0 | 0.99-1.0 | 0.99-1.0 | 0.99-1.0 | 0.98 | 0.14-0.15 | 0.94-0.95 | 0.96-0.97 | 0.99-1.0 |
| 10-el, 2 targets (10(d)) | 0 | 0.365 | 0.25 | 0.135 | 0.22 | 0.35 | **0.465** | ~0.38 | 0.37 | 0.375 |
| | 5 | ~0.52 | 0.50 | 0.37 | 0.48 | ~0.50 | 0.52 | **0.59** | 0.565 | 0.565 |
| | 10 | 0.675 | 0.67 | 0.565 | 0.68 | 0.665 | 0.58 | ~0.70 | ~0.70 | **0.705** |
| | 20-50 | 0.77-0.785 | 0.745-0.76 | 0.67-0.68 | **0.80-0.81** | 0.76-0.78 | 0.62-0.64 | 0.76-0.78 | 0.76-0.77 | 0.79-0.80 |
| 10-el, 3 targets (10(f)) | 0 | 0.25 | 0.285 | 0.065 | 0.245 | 0.10 | 0.00 | 0.34 | **0.57** | 0.56 |
| | 10 | 0.69 | 0.68 | 0.37 | 0.67 | 0.48 | 0.01 | 0.58 | 0.78 | 0.78 |
| | 50 | 0.80 | 0.79 | 0.54 | 0.78 | 0.645 | 0.01 | 0.66 | 0.83 | **0.84** |

**(d) Off-grid RMSE (Fig. 10(a), (c), (e))**

| Setting | 0 dB | 15 dB | high SNR |
|---|---|---|---|
| 18-el, 2 targets (10(a)) | Parents lowest: SBRI-X 0.103, SBRI 0.106. Networks: SBRI-X-Net 0.129, SBRI-Net 0.137; OGIR 0.135. | SBRI-X-Net 0.024, OGIR 0.030, SBRI-Net 0.033, CNN-DNN 0.042, SBRI-X 0.069, SBRI 0.072 | 20-50 dB: SBRI-X-Net 0.022-0.031, OGIR 0.028-0.032 (they cross; OGIR is lower at 30 dB), SBRI-Net 0.029-0.035, SBRI-X about 0.060 |
| 10-el, 2 targets (10(c)) | SBRI-X-Net 0.102, SLIM 0.117, SBRI-X 0.118, SBRI-Net 0.122 | SBRI-Net 0.054, SBRI-X-Net 0.065, SLIM 0.095, SBRI-X 0.101 | 50 dB: SBRI-X-Net 0.074, SBRI-Net 0.079, SLIM 0.090, SBRI-X 0.096, OGIR 0.143 |
| 10-el, 3 targets (10(e)) | SBRI-Net 0.089, SBRI-X-Net 0.091, OGIR 0.119, SBRI-X 0.122 | — | 50 dB: SBRI-X-Net 0.074, SBRI-Net 0.079, OGIR 0.079, SBRI-X 0.081 |

**(e) Fig. 6(b), on-grid, 18-element SLA, targets (+-30) deg, 1,000 trials.** At 0 dB the hit rates are about 0.89 (SBRI-X), 0.565 (SBRI) and 0.45 (1-bit SLIM). By 15 dB all three are at or above about 0.97.

**(f) Fig. 6(d) and 6(f), the (a, b) ablations of SBRI-X (p. 920) [resumed pass, read from a 4x render].**

| Panel (legend) | 0 dB | 5 dB | plateau |
|---|---|---|---|
| 6(d): a = 1.0; b = 1.0 / 0.5 / 0.1 / 1.5 / 2.0 | 0.90 / 0.89 / not separable (curves overlap, about 0.8-0.9) / 0.87 / 0.83 | 0.97-0.99 | 1.00 from 15 dB |
| 6(f): b = 1.0; a = 7.0 / 5.0 / 3.0 / 1.0 / 0.5 | 0.87 / 0.85 / 0.77 / 0.39 / 0.18 | 0.97 / 0.97 / 0.95 / 0.77 / 0.45 | about 0.97 (a >= 3), 0.93 (a = 1), 0.73 (a = 0.5) |

- **The two panels disagree on the same setting.** (a, b) = (1.0, 1.0) appears in both: hit rate 0.90 at 0 dB in 6(d), but 0.39 in 6(f); plateau 1.00 versus about 0.93. The text gives no different scenario for 6(e)-(f). We cannot resolve this without the code; treat the a-ablation curves with caution.
- In 6(e)-(f), a = 0.5 is the **worst** setting at b = 1, yet the 2-D grid search of Fig. 7 (10 dB) puts the best RMSE near (a, b) = (0.5, 1.7). Both use a != 1, where the printed MM bound does not hold (checks B-C).

### 5.3 Results against the iterative parent, across SNR (the focus of this note)

- **On-grid (Fig. 8):** SBRI-X-Net roughly ties SBRI-X. Hit rate is 0.70 vs 0.71 at 0 dB and 0.94 vs 0.96 at 5 dB, then about 1.0 for both. RMSE is equal at 0-15 dB and slightly lower for the network at 20-50 dB.
  - SBRI-Net beats its parent SBRI by a wide margin at low SNR: 0.56 vs 0.30 at 0 dB and 0.88 vs 0.59 at 5 dB.
- **Off-grid, 18 elements (Fig. 10(a)-(b)):**
  - Both networks have a higher hit rate than their parents at 0 dB: 0.535 vs 0.36 for the X pair, 0.525 vs 0.40 for the SBRI pair.
  - From 5 dB upward their RMSE is much lower (15 dB: 0.024 vs 0.069).
  - At 0 dB the parents' RMSE is lower (0.103 vs 0.129). But RMSE only counts successful trials, so methods with different hit rates cannot be compared on RMSE.
- **Sparser array and 3 targets (Fig. 10(d)-(f)):** the networks are clearly above their parents at low SNR (3 targets, 0 dB: 0.56 vs 0.245 and 0.57 vs 0.065).
  - At high SNR on the 10-element array, the parent SBRI-X has the highest hit rate (about 0.81).
- **The text overstates in places.** It says the proposed methods have the highest hit rate across SNR, but the plots show exceptions:
  - 1-bit SLIM, an on-grid method, is slightly higher than both networks in 10(b) at 0-5 dB (0.56 vs 0.535; 0.85 vs 0.835).
  - 1-bit SPICE is highest at 0 dB in 10(d) (0.465 vs 0.375), and CNN-DNN is highest at 5 dB (0.59 vs 0.565).
  - The smallest of these gaps are within our read-off error. The 0.09 gap at 0 dB in 10(d) is not.
- **Generalisation outside the training SNR range.** The training range is 0-30 dB. Tests go **upward only**, to 35-50 dB (Figs. 8 and 10), and the networks' curves stay flat there.
  - No test below 0 dB.
  - No test on an array geometry the network was not trained on: one network per array is implied but not stated (UNVERIFIED).
  - No test with a number of sources different from training, unless the 3-target network was not retrained (UNVERIFIED).

### 5.4 An inconsistency in the RMSE units (our arithmetic)

The text defines RMSE over **successful** trials, i.e. trials where every error is within the threshold of 2 (p. 919). If that threshold is in degrees (see 5.1), then RMSE <= 2 deg = 0.0349 rad by definition.

Many plotted values are larger than this, even though the axes are labelled "rad":
- Fig. 8(b): 0.037-0.185.
- Fig. 6(a): up to about 1.
- Fig. 7(a): colour scale up to about 4.
- Fig. 10(c): up to 0.234.

So either the unit label or the definition used for the plots differs from the text. **Use only the ordering of the RMSE curves, never their absolute values.** Hit rates are not affected.

## 6. Citable claims

Only C4 is a verbatim quote (the copyright limit allows one per document). The others are faithful paraphrases with exact pinpoints.

- **C1 (Sec. III-A, eqs. (8)-(20), pp. 912-913).** One-bit single-snapshot DOA is posed as a sparse MAP problem: a probit likelihood plus a Laplacian-type prior, solved by MM-IRLS.
  - *Cite for:* the classical sparse-MAP lineage of one-bit DOA; the probit one-bit likelihood (the 1-bit case of our quantiser likelihood).
- **C2 (Sec. III-C, eqs. (32)-(34); Sec. V-A, Fig. 6(a)-(b), p. 920).** The logistic (Bernoulli) likelihood variant, SBRI-X, gives the lowest RMSE of SBRI-X, SBRI and 1-bit SLIM at all SNRs, and the highest hit rate at 0-15 dB, for on-grid targets.
  - *Cite for:* a tunable-slope likelihood helping at low SNR.
- **C3 (Sec. IV-A, eqs. (61), (64)-(65), (68), p. 917).** The unrolled networks replace the matrix inversion with per-layer 1-D CNNs and the grid-gap update with FC layers. The pseudo-measurement physics layers are kept.
  - *Cite for:* an unrolled one-bit DOA estimator "unfolded from MM/IRLS"; the closest prior unrolled one-bit DOA network.
- **C4 (Sec. V-A, p. 921, Fig. 8(b)-(c)).** "SBRI-X-Net achieves comparable RMSE to SBRI-X at SNRs from 0 dB to 15 dB". The same sentence adds slightly lower RMSE at 20-50 dB, and the next sentence says the hit rate is maintained at 35-50 dB, outside the training SNRs.
  - *Cite for:* an unrolled network matching its parent on-grid; upward SNR generalisation.
- **C5 (Sec. V-A, p. 921, Fig. 9).** Training both sigmoid parameters together caused numerical instabilities (NaN). The authors fixed a and learned only b, which decreased steadily during training.
  - *Cite for:* the stability lesson (learn few physics scalars at a time). Do **not** cite it for "learned likelihoods are flatter than the nominal one": b is confounded with the amplitude scale (check L).
- **C6 (Sec. V-B, pp. 921-922, Fig. 10).** Off-grid, the model-based networks beat a purely data-driven CNN-DNN [59]. The text says SBRI-X-Net has the lowest RMSE on the 18-element array with 2 targets (Fig. 10(a)). For the 10-element array it says all methods degrade, and that the proposed methods still lead. For 3 targets (5,000 trials, p. 922) it says SBRI-X-Net has the lowest RMSE over the whole SNR range and both networks the highest hit rate. It does not say the 3-target networks were retrained (UNVERIFIED). Our reading of the 3-target hit-rate and RMSE curves matches this, but only in the 0-50 dB range the paper tested.
  - *Cite for:* model-based beating purely data-driven in one-bit DOA, with our caveats from Sec. 5.3. Do not cite it as "the gains hold for every array and every number of sources": the paper's own text does not claim that.
- **C7 (Sec. IV-E, Table I, p. 919).** The iterative algorithms cost O(N_iter N^3), dominated by an N x N inversion. The networks cost O(KN) on-grid and O(KN^2) off-grid.
  - *Cite for:* the complexity motivation for unrolling. Note that these are FLOPs, not measured run times.
- **C8 (Sec. IV-B, p. 918).** Training data: 1,000,000 samples over 0-30 dB in 5-dB steps, 2 sources, a field of view of +-60 deg, with 18- and 10-element SLAs.
  - *Cite for:* the training range when discussing generalisation.

## 7. Limitations: what the paper does not show

1. **No numeric results**, only plots. The RMSE axes are inconsistent with the stated definition (Sec. 5.4). The same SBRI-X setting (a, b) = (1, 1) has different hit rates in Fig. 6(d) and Fig. 6(f) (0.90 vs 0.39 at 0 dB; Sec. 5.2(f)) [resumed pass].
2. **Few test geometries.** Every test uses fixed target positions; random-DOA test sets are not reported. Amplitude phases are limited to the first quadrant in training (and presumably in testing).
3. **No bound to compare with.** The one-bit CRB is not plotted, although they cite a conservative CRB ([2]).
4. **No measured latency, and no convergence count.** The conclusion (Sec. VI, p. 922) says the networks give faster convergence and that the results validate hardware efficiency. No wall-clock time, iteration count or convergence curve for the networks is shown. Our check H evaluates their FLOP formulas:
   - On-grid (N = 121 on a 1-deg grid over +-60 deg, M = 18): one SBRI-Net layer costs about 0.94 times one SBRI iteration (1.60e6 vs 1.71e6 FLOPs).
   - Off-grid (N = 61, M = 18): one SBRI-X-Net layer costs about 2.0 times one off-grid SBRI iteration (8.65e5 vs 4.28e5 FLOPs).
   - So at the paper's own sizes, the network is only faster if it uses fewer layers than the loop needs iterations. K is not reported.
5. **SNR generalisation only upward** (35-50 dB). Nothing below 0 dB, which is where one-bit data are hardest. The SNR definition is missing.
6. **Mathematical gaps (our checks):**
   - The MM majoriser (42) and identity (33) are valid only for a = 1 (checks B, C, D). The a != 1 ablations and the grid-search optimum (a of about 0.5) therefore run outside the stated MM guarantee.
   - As printed, the joint MAP (38)-(39) has no prior on eps. Its infimum is reached by x = 0 (no source) with eps -> +-infinity times ybar. On a paper-like instance (check F) the objective falls from 23.8 at the true x to the lower bound 7.65 at x = 0.
   - The method therefore works because of its initialisation and the finite number of iterations, not as a global MAP estimator.
   - The theorems assume RIP with delta_2s < 1, which fails for the paper's own dictionaries (check J). The constant C1 in (86) does not shrink with SNR, so the error bound is loose even where it applies.
7. **Notation slips a reader will trip over:**
   - x is stated to be real in (3) but treated as complex in (58).
   - Theorem 2 cites Algorithm 2 instead of Algorithm 3.
   - Algorithm 4 returns xhat instead of xtil.
   - Eq. (65) feeds xhat^k to the FC block but (73) feeds xhat^{k-1}.
   - Eq. (71) uses shat^{k-1} where (70) defines shat^k.
   - The caption of Fig. 6 swaps "vary a" and "vary b" relative to the plotted legends. The text mostly follows the legends (6(c)-(d) vary b; 6(e)-(f) vary a), except p. 921 which cites Fig. 6(e)-(f) for the sensitivity to b.
8. **Missing implementation details** (Sec. 3.5): number of layers, activations, output layer, label normalisation, sign of beta, DOA read-out, the sharing of a_k and b_k, the alpha used in the experiments, and the (a, b) of the iterative SBRI-X in Figs. 8 and 10. There is no code. The supplementary material was not checked by us.
9. **Narrow scope:** 1-bit only (no 2-4-bit case), no AGC, ideal arrays, 1-D DOA with K <= 3, and no unknown model order.

## 8. Relevance to us

### 8.1 Exact correspondence (1-bit case)

- Our 1-bit ADC output is (A/2) csgn(Y) exactly (check G). So our 1-bit problem is the paper's model with the dictionary atom vec(W^H a_r(u) a_t(v)^H F) on a 2-D (u, v) grid (check K).
- The paper's probit likelihood (8) is the b = 1 case of our quantiser cell likelihood. `mlq.nll_terms` computes the log-probability of each cell from log_ndtr of the cell edges (`mlq.py`, lines 60-66).
- Their SBRI pseudo-measurement (17) is exactly our EM E-step at 1 bit (check I).
- Differences:
  - Our quantiser ML is **gridless**: continuous Levenberg-Marquardt / Newton refinement with 13 classical starts plus EM. SBRI is grid-plus-Taylor.
  - Our ML needs sigma; SBRI and SBRI-X do not.
  - Our likelihood also covers b = 2 and the AGC cells.

### 8.2 What to reuse

1. **A likelihood "temperature": already tried by us, and it does not help when sigma is known [resumed pass, corrected].** SBRI-X-Net learns the slope b, and it settles at about 0.19 (Fig. 9). Check L shows this b is confounded with the amplitude scale, so it is not a clean noise estimate.
   - Our quantiser ML already has the same knob: the noise-std inflation kappa (s = kappa sigma; `finetune/adversarial/mlq.py`, docstring lines 6 and 12).
   - It was tuned on tuning banks (seeds 8730 / 8745, 300 scenes per cell, no CI; `finetune/adversarial/s2_tune.log`). **kappa = 1 (no inflation) was best in all four ADC cells.** At 1-bit / 15 dB, ML (all starts) scored 0.529 at kappa 1.0, 0.508 at 1.5, 0.432 at 2.5 and 0.516 at 0.7. Mean over the four cells: 0.495 / 0.461 / 0.389 / 0.475 for kappa 1.0 / 1.5 / 2.5 / 0.7.
   - In unfolded_v2 the hypernetwork could learn a per-layer kappa_t. Its outputs stayed at their initial values (kappa 1.000; `unfolded_v2/build/deviations.txt`, item 8).
   - So with the **true** sigma, a flatter likelihood is not a gain for our exact-likelihood ML. SBRI-X's flatter logistic most likely compensates for things our ML does not have: the grid, the first-order Taylor model and an unknown amplitude scale. This explanation is our inference, not tested.
   - What remains open is the **unknown-sigma** case: ML loses 0.04-0.07 Pd when its assumed SNR is 5 dB off, and at 1 bit the noise level cannot be identified. There the SBRI-X idea (work in noise units, fix the likelihood shape) corresponds to our "ML nominal 10 dB" baseline (`deviations.txt`, item 9; defined, but no result log found, so no number yet). Any learned or per-scene kappa must be tuned on validation seeds 81,000,000+ and tested on banks 12000-12999 before anything is claimed.
2. **Physics layers built from Acal and Acal^T only.** In our setting Acal^T is one 2-D FFT of Htil (check K). So a pseudo-measurement layer plus a CNN costs about one FFT and one CNN per stage, not an explicit inverse.
   - An explicit inverse is impractical: one SBRI iteration on a 64 x 64 grid costs 5.4e10 FLOPs, versus 6.2e7 for one SBRI-Net layer (check H).
3. **A stability recipe.**
   - Learn one physics scalar at a time; they fixed a to stop NaNs.
   - Train in stages; they froze the gap head for 100 epochs and then unfroze everything.
   - These match two problems in our `unfolded_v2/build/deviations.txt`: front-end drift (item 8) and the zero-gradient saddle of a gated output (item 10).
4. **The multi-bit extension is ours to make.** Their pseudo-measurement is the 1-bit truncated-normal mean. For b bits it becomes the cell-truncated mean, which `mlq.em` already computes. A "b-bit SBRI-X-Net" would be described as "unfolded from EM / IRLS with the quantiser cell likelihood (Hu et al. 2025 for the 1-bit case)".

### 8.3 Must it be a baseline?

**It must be cited, but it is not a mandatory experimental baseline.**
- It solves a different problem: 1-D DOA on sparse arrays, 2-3 sources at fixed positions, 1 bit only.
- There is no code.
- At 1 bit its likelihood is the same probit model our exact quantiser ML already uses, without the grid and the Taylor approximation. Our ML is gridless and multi-start, so the iterative SBRI is expected to be at most as good. **This expectation is untested and must not be claimed.**
- A reviewer of the ADC section could still ask for it (see Q1). If time allows, add **our re-implementation of iterative SBRI-X (Algorithms 3-4)** as a 1-bit classical baseline. A retrained SBRI-X-Net would be an optional learned baseline.

### 8.4 How to make the comparison fair

- **Same scenes:** test banks 12000-12999 at 1 bit; SNR groups -10..25 dB.
- **Same metric:** paper-style Pd (1 deg, Hungarian matching, mean over SNR groups), strict Pd and RMSE in degrees, with paired bootstrap CIs. Also report their 2-deg scene-level hit rate, so readers can relate to their plots.
- **Same knowledge:** the true L for every method. For SBRI-X, take the L largest peaks of |x| plus the gaps beta.
  - SBRI-X does not use sigma, so also report our ML in its SNR-blind form ("ML nominal 10 dB", `deviations.txt` item 9) next to ML with the true sigma.
- **Same tuning budget:** (a, b, gamma0, alpha, eta, T_max, grid step) tuned on validation seeds 81,000,000+ only.
- **Disclosed deviations:**
  - 2-D (u, v) grid with an oversampling factor of our choice.
  - Matrix-free conjugate-gradient solves with FFT operators instead of explicit inverses.
  - 2-D Taylor gaps (beta_u, beta_v).
- **At 2 bits:** SBRI-X can only use the sign bits. Report it as "not designed for b > 1" or leave it out; do not present it as a fair 2-bit baseline.
- **Latency:** the same 1-thread CPU timing as `s6_latency.json`.

### 8.5 Corrections to our own documents

- `unfolded_v2/lit/unfolding_survey.md` says unfolded one-bit methods "match their iterative parents and do not beat them [35]" (line 26; also lines 106, 111 and 117, "parity"). Its line 206 notes that the [35] figures were not readable at the time. That statement holds **only for the on-grid SBRI-X pair** (Fig. 8). Elsewhere the networks beat their parents:
  - SBRI-Net vs SBRI on-grid at 0-10 dB (0.56 vs 0.30 at 0 dB);
  - off-grid at low SNR (0.535 vs 0.36);
  - 3 targets, 0 dB, 10-element array (SBRI-X-Net 0.56 vs SBRI-X 0.245; SBRI-Net 0.57 vs SBRI 0.065).
  - Readers should also be told that these are read off plots at fixed target positions.
- The DOI is now verified: 10.1109/JSTSP.2025.3627141, vol. 19, no. 6, pp. 909-925.

### 8.6 What it implies for our network design (inference, to be tested)

- In this paper, learning helps most where the iterative parent is weakest: the probit SBRI at low SNR, the 2-deg grid, the sparser array and 3 sources. Where the parent is already good (on-grid SBRI-X), learning only matches it.
- Our parent is strong: 13-start exact-likelihood ML reaches 0.540 at 1-bit / 15 dB, against 0.626 when started from the true angles. In unfolded_v2 the learned refinement parts did not beat their initialisation.
- Both findings point the same way: put the learning where the classical method is weak. That means generating candidates or seeds and choosing among them. Per the task brief, the oracle pick in the candidate pool is +0.036 to +0.060 above 13-start ML; lowest-NLL selection only reaches about ML's level.
- Keep the refinement itself physics-exact. With the true sigma, do not add a likelihood temperature: our own tuning and the unfolded_v2 hypernet both point to kappa = 1 (8.2.1). A sigma-free variant is a separate question, for the unknown-SNR case only.

## 9. "If a reviewer asks..."

**Q1. "Why don't you compare with SBRI-X-Net (Hu et al., JSTSP 2025), a recent unrolled one-bit DOA network?"**

Answer honestly:
- Its problem is 1-D DOA with a sparse array and 2-3 sources at fixed positions, at 1 bit only. Ours is joint 2-D AoA/AoD in an analog beamspace, with 1-2-bit ADCs and AGC. No code is available.
- Our classical bar is already an exact-likelihood quantiser ML. At 1 bit it uses the same probit model as their SBRI (eq. (8)), but gridless, multi-start and with EM. We also report quantisation-consistent NOMP.
- If we have re-implemented iterative SBRI-X on our 2-D dictionary (Sec. 8.4), we show its numbers. If we have not, we **cannot** claim to beat it and should say so.

**Q2. "Hu et al. avoid needing the noise variance and generalise outside the training SNR. Does your method?"**

- Their method estimates amplitudes in noise units and fixes or learns the likelihood slope. In effect that is an assumed noise level, and they test only upward (35-50 dB) on fixed targets.
- Our verified results:
  - The quantiser ML needs sigma and loses 0.04-0.07 when its SNR is 5 dB off (2-bit / 15 dB: 0.682 -> 0.606-0.615).
  - The network read-out needs no sigma but is weaker (0.593 at 2-bit / 15 dB).
  - At 1 bit the noise level is not identifiable from the data.
- An out-of-range test (above 25 dB and below -10 dB) has **not** been run for our system. Say so, and report it once it is run.
- Do not offer a "learned temperature" as the fix. With the true sigma, our tuning gave kappa = 1 as best (0.529 vs 0.508 at kappa 1.5, 1-bit / 15 dB, `s2_tune.log`), and the learned kappa in unfolded_v2 did not move from its initial value (8.2.1). The honest answer is: report the measured SNR-mismatch loss of ML (verified above), the sigma-free network read-out, and, once it has been run, the SNR-blind "ML nominal 10 dB" baseline. That baseline is defined in `deviations.txt` item 9, but we found no result log for it, so no number for it exists yet.

**Q3. "Isn't your unfolded decoder just SBRI-Net applied to beamspace?"**

No. The parent algorithm is different:
- Theirs: a grid MAP with an MM-IRLS inverse step and a first-order Taylor gap.
- Ours: NOMP / quantiser-likelihood ML with continuous Newton / Levenberg-Marquardt refinement over (u, v) and complex gains, plus the b-bit cell likelihood and AGC.

The general idea of unrolling a one-bit sparse-recovery loop is **not** new. Credit Hu et al. 2025 as the closest one-bit DOA instance, and describe ours as "unfolded from NOMP / quantiser ML". The LISTA-based one-bit DOA paper they cite ([48], Yeganegi et al., SAM 2024) was seen only as a citation: UNVERIFIED by us.

## 10. Glossary

| Term | Plain English |
|---|---|
| DOA (direction of arrival) | The angle a signal comes from. |
| One-bit quantisation, csgn | The receiver keeps only the sign (+ or -) of the real part and of the imaginary part. |
| Single snapshot | Only one time sample per antenna; nothing to average over time. |
| SLA (sparse linear array) | Antennas on a line with some positions left empty, for a wide aperture with fewer elements. |
| On-grid / off-grid | Whether the true angle sits exactly on one of the candidate angles the algorithm searches (on-grid) or between two of them (off-grid). |
| Grid gap beta | How far the true angle is from its nearest candidate angle. |
| First-order (Taylor) approximation | Approximating a curve near a point by its tangent line, here for the steering vector between grid points. |
| Dictionary Acal / atom | The table of steering vectors for every candidate angle; one column is an "atom". |
| Sparsity, Laplacian-type prior | The assumption that only a few candidate angles are active; the prior is the formula that rewards having few non-zero entries. |
| MAP (maximum a posteriori) | Choose the answer that is most probable given both the data and the prior. |
| Likelihood | How probable the observed bits are if a given answer were true. |
| Probit / Gaussian CDF Phi | The S-shaped curve giving the probability that a Gaussian-noisy value is positive. |
| Logistic / sigmoid link sig(x) | Another S-shaped curve, 1/(1 + a e^{-bx}). b sets the steepness; a != 1 shifts the curve sideways. |
| Bernoulli likelihood / binary classification | Treating every bit as a yes/no outcome with some probability. |
| MM (majorisation-minimisation) | Repeatedly replace a hard function by a simpler one that touches it at the current point and lies above it, then minimise the simpler one. |
| Majoriser / surrogate | That simpler upper function. |
| IRLS (iteratively reweighted least squares) | Solve a weighted least-squares problem, recompute the weights from the answer, and repeat. |
| Pseudo-measurement v | The best guess of the unquantised value, given its sign and the current prediction (= the truncated-normal mean; check I). |
| Regularisation gamma | How strongly the sparsity penalty is enforced. |
| Noise offsets eps (SBRI-X) | Extra unknowns standing for the noise added to each measurement. |
| Deep unrolling / unfolding | Turn a fixed number of iterations of an algorithm into network layers and learn some of the parts. |
| SBRI / SBRI-X | The paper's classical loops: Gaussian likelihood / logistic likelihood with noise offsets. |
| SBRI-Net / SBRI-X-Net | Their unrolled networks. |
| 1-bit SLIM / LIKES / SPICE | One-bit versions of classical sparse spectral estimators (from their ref. [28]). |
| OGIR | Off-grid iterative reweighted one-bit DOA method (their ref. [37]). |
| CNN-DNN | A purely data-driven two-stage network (their ref. [59]). |
| Hit rate | The fraction of trials in which all estimated angles are within 2 deg. |
| RMSE | Root-mean-square angle error, here over successful trials only. |
| RIP (restricted isometry property) | A condition that every small set of dictionary columns behaves almost like orthogonal vectors. It fails for fine DOA grids (check J). |
| BCE / MSE | Binary cross-entropy (a loss for "is this grid point active?") and mean squared error. |
| FOV | Field of view, the angle range considered (+-60 deg). |
| Epoch | One pass over the whole training set. |
