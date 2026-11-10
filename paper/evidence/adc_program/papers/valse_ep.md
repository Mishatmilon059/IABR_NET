# VALSE-EP explained: grid-less variational Bayesian line spectral estimation from quantized samples

Prepared 2026-10-16 for the analog mmWave 16x16 joint AoA/AoD project. The paper was read in full from the arXiv v3 PDF text (33 printed pages; page numbers below are the printed page numbers). Figures were not digitised and the paper's text has no results table. Anything I could not check is marked UNVERIFIED (list in section 11).

Sources: arXiv abstract page (WebFetch summary); arXiv PDF text (pdftotext and pymupdf, cached at C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/valse_ep_1811.05680.pdf); IEEE Xplore journal page (browser: title, authors, volume, issue, publisher confirmed); authors' GitHub repository page (WebFetch summary; code not read); numeric check valse_dft_structure_check.py with log valse_dft_structure_check.log (same cache folder).

Copyright note: this file paraphrases the paper and gives section and page locators so the wording can be checked. It contains one short verbatim phrase.

---

## 0. Bibliographic entry

- Authors: Jiang Zhu, Qi Zhang, Xiangming Meng (Zhu and Zhang: Ocean College, Zhejiang University; Meng: Huawei Technologies, Shanghai). Verified on the PDF title block and on IEEE Xplore.
- Journal version (verified on the IEEE Xplore page, document 9597637): "Gridless variational Bayesian inference of line spectral from quantized samples", China Communications, Volume 18, Issue 10, October 2021, publisher IEEE. Verified: venue, volume, issue, authors, publisher, title. The page numbers 77-95 come from a search-result snippet of the same record and were not on the fetched page, so they are UNVERIFIED. A DOI string appeared in an earlier draft of this file, but I did not see it in any source I opened, so it is UNVERIFIED and not used here.
- arXiv: arXiv:1811.05680 [cs.IT]. The v3 PDF (3 Sep 2019) is titled "Grid-less Variational Bayesian Inference of Line Spectral from Quantized Samples". The title in the task ("...Line Spectral Estimation from Quantized Samples") does not match the v3 PDF. The first-version date (14 Nov 2018) comes from a summarised abstract page (UNVERIFIED).
- The arXiv PDF is a dated draft ("September 4, 2019 DRAFT"). The journal version may differ. No journal-version numbers are claimed.
- Code: github.com/RiverZhu/VALSE-EP (MATLAB files: VALSE_EP.m, VALSE.m, NMSEvsIter.m, GaussianMomentsComputation_MJH.m, README). The README gives the main call VALSE_EP(y_q, m, ha, x, Iter_max, B, yy_min, alpha, method_EP), with method_EP = 'diag_EP' recommended and B = inf for unquantised data. The code was not read, so these details are UNVERIFIED until the files are opened.
- References the paper relies on: VALSE (Badiu, Hansen, Fleury, IEEE TSP 65(9), 2017, pp. 2247-2261, ref [20]); EP for generalised linear models (Meng, Wu, Zhu, IEEE SPL 25(3), 2018, ref [41]); moments of quantised measurements (Zhu, Han, Meng, IEEE Access 2018, ref [42]); NOMP (Mamandipoor, Ramasamy, Madhow, IEEE TSP 64(19), 2016, ref [10]); CRB for quantised spectral CS (Fu and Chi, IEEE TSP 66(12), 2018, ref [38]).

---

## 1. Plain-English summary

Some sensors record only coarse, few-level values of a signal that is a sum of a few pure tones. Recovering the tone frequencies and strengths from such coarse data is hard, because rounding creates many false tones. The authors build an estimator that never uses a fixed list of candidate frequencies (no grid). It alternates between two parts. Part A treats the coarse data as if they were fine data with a known noise level per sample, and runs an existing Bayesian tone finder (VALSE). Part B looks at each coarse reading, works out what the fine value probably was given the cell it fell into, and turns that into a fine-looking reading with its own noise level. The two parts pass information back and forth until they agree. The method also estimates the number of tones and the noise level. Using plots only, the authors show it beats their earlier method on coarse data and gets close to a theoretical lower bound on error. They do not compare it with the classical methods that our project uses.

---

## 2. Problem and system model

### 2.1 The paper's model

- Signal (eqs. 1-2, section II, p.5): z = sum_k a(theta_k) w_k, with a(theta) = [1, e^{j theta}, ..., e^{j(N-1) theta}]^T. This is a Vandermonde vector in the sample index. Frequencies lie in [-pi, pi).
- Candidate form (eq. 5, p.6): z = A(theta) w, where A = [a(theta_1), ..., a(theta_N)] and N > K. A binary support s says which of the N components are active (w_i = 0 when s_i = 0).
- Noise (section II, p.5): eps ~ CN(0, sigma^2 I_N).
- Measurement (eq. 3, p.5): y = Q(Re{z + eps}) + j Q(Im{z + eps}). Real and imaginary parts are quantised separately by a b-bit uniform quantiser with 2^b cells (eq. 4).
- Quantiser scale (section VII-A, p.21): uniform on [-3 sigma_z/sqrt(2), 3 sigma_z/sqrt(2)], where sigma_z^2 is the variance of z, which the authors say is about K in their normalisation. For one bit the threshold is zero (sign quantiser).
- SNR definition (section VII-A, p.21): SNR = 20 log10(||A(theta) w||_2 / ||eps||_2). This is a total-energy ratio over the whole vector, not a per-element SNR.
- Prior (eqs. 6-7, p.6): each theta_i is uniform on [-pi, pi) with density 1/(2 pi). Each support indicator s_i is Bernoulli(rho). Given s_i = 1, w_i is complex Gaussian with variance gamma. The model order is K_hat = |S_hat| (section IV, p.10).

### 2.2 Mapping to our project

Our symbols come from the task and from RESULTS_SUMMARY.md. Where the summary does not define a symbol, this is marked.

| Paper | Meaning in paper | Our project | Note |
|---|---|---|---|
| y (N complex) | quantised measurements | vec(Y), Y = W^H H F + Z, 16x16 = 256 complex beam samples, quantised per Re and Im | Same per-component quantiser idea. Our ADC acts on beam-domain Y. |
| z | noiseless 1-D line spectrum | W^H H F (noiseless beam-domain signal) | Structure differs (section 2.3). |
| eps, sigma^2 | additive complex white noise, variance sigma^2 | noise on vec(Z); variance set by SNR (-10..25 dB) | Paper's SNR is total-energy; ours must be matched before comparison. |
| N | number of measurements and number of candidate components (their setting) | 256 measurements | Our problem is 2-D, so N does not carry over directly. |
| K, K_hat | number of sinusoids; estimated order | L = 3 paths (known in our benchmark) | The paper estimates K; we fix L = 3. |
| theta_k | one frequency per tone | two angles per path (AoA, AoD); mapping to 1-D frequencies needs a model change (section 2.3) | UNVERIFIED mapping. |
| w_k | complex amplitude | path gain g_l | Direct analogue. |
| A(theta) | Vandermonde dictionary | beam-domain dictionary built from W^H a(.) and a(.)^H F (our derivation) | Not Vandermonde (section 2.3). |
| b | bit depth (1, 2, 3 bits in their figures) | b (1-bit, 2-bit in our tables) | Same meaning. |
| quantiser range +-3 sigma_z/sqrt(2) | scale from the true sigma_z | our peak AGC A = max(abs Re, abs Im) over 256 samples | The paper's scale is an oracle; ours is data-derived. |
| u, v | not in the paper | not defined in RESULTS_SUMMARY.md | UNVERIFIED. If u, v are the two direction variables (sines of the angles), they map to the theta row. Confirm against the 2-D model file. |
| SNR | total-energy ratio, 20 log10 of norm ratio | SNR -10..25 dB (definition to confirm in code) | Must be matched before any comparison. |

### 2.3 Structural mismatch with our 2-D beam-domain model

The paper's signal is a Vandermonde sum in the measurement index (eqs. 2, 5). Our beam-domain entry Y_ij = w_i^H H f_j is a product of two beam responses, one per dimension. With a DFT codebook, the beam response of one ULA direction is a Dirichlet-type kernel in the beam index, not a complex exponential. I checked this with valse_dft_structure_check.py (log: valse_dft_structure_check.log, cache folder): for a 16-point DFT codebook and one ULA response, successive ratios r[i+1]/r[i] are not constant (std of |ratio| between 0.50 and 1.10 at three test angles). So vec(Y) is not a pure 1-D line spectrum in the paper's sense. This is my derivation and check for one DFT codebook, not a statement from the paper. General codebooks were not checked.

Consequence: VALSE-EP cannot be applied to beam-domain Y as published. A 2-D extension (two frequencies per path, a 2-D dictionary, a 2-D von Mises posterior) would be needed. The paper has no 2-D method, so any such extension is our own derivation and must be labelled that way.

---

## 3. The method step by step

### 3.1 Why expectation propagation (EP) is used

The quantiser makes the likelihood of each measurement a difference of two Gaussian CDFs: the probability that the fine value lies in the observed cell (eqs. 15-17, p.7). This likelihood is not Gaussian, so the linear-model Bayesian tool (VALSE) cannot use it directly. EP replaces each non-Gaussian factor with a Gaussian message that has its own variance. In plain words, for each measurement EP asks: given the cell I observed and what the rest of the model says, which Gaussian pseudo-measurement would explain this? That pseudo-measurement has a per-measurement noise variance, which is why the noise is heteroscedastic. The paper argues that diagonal EP (one variance per measurement) is better than scalar EP (one variance for all) (section IV, p.9).

### 3.2 The two modules

- Module B (componentwise MMSE, section V-A, pp.14-16): for each measurement, combine the Gaussian message from module A with the exact quantiser likelihood. The result is a truncated Gaussian on the observed cell, with a closed-form mean and variance (eqs. 47-48, from ref [42]). Dividing out the module-A message gives the extrinsic pseudo-measurement z_ext_B and its variance v_ext_B (eqs. 51a, 51b, p.15). These are the pseudo-observation y_tilde and noise variance v_tilde that module A uses.
- Module A (VALSE with known heteroscedastic noise, sections IV and V-B, pp.10-13 and p.16): treat y_tilde = A(theta) w + eps_tilde with eps_tilde ~ CN(0, diag(v_tilde)), and run one VALSE step. It returns the support, the weight posterior, the frequency estimates, and the posterior mean and variance of z (eqs. 63-64, p.17). The extrinsic values are passed back to module B (eqs. 67-68, p.17).
- Noise variance (eq. 58, p.16): updated with an EM-type rule that the authors call approximate for quantised data: sigma^2 = (||y_tilde - z_post_B||^2 + sum(v_post_B)) / N.

### 3.3 Key equations

Transcribed from the text layer; some glyphs are garbled, so check against the PDF before reuse.

- Eq. (38a), p.11: the matrix J has entries J_ij = a_i^H Sigma^{-1} a_j for i != j; the diagonal is printed as tr(Sigma^{-1}) (UNVERIFIED as printed).
- Eq. (38b), p.11: h = A_hat^H Sigma^{-1} y_tilde.
- Eq. (40), p.12: C_S = (J_S + gamma^{-1} I)^{-1} and w_S = C_S h_S (weight posterior for the active set S).
- Eq. (45), p.13: rho_hat = ||s_hat||_0 / N and gamma_hat = (w_S^H w_S + tr(C_S)) / ||s_hat||_0 (closed-form updates for the prior parameters).
- Eqs. (30a, 30b), p.10: theta_hat_i = arg E[e^{j theta_i}] and a_hat_i = E[a(theta_i)], with each frequency posterior approximated by a von Mises distribution (eq. 37, heuristic from VALSE [20]).
- Eqs. (82), (86), (92), appendix X-A, p.28: the change in log evidence when a component is switched on or off. The greedy support search uses these.

### 3.4 Algorithm 1 in pseudo-code (re-implementation sketch)

Inputs: quantised y_q (N complex values), the quantiser cells for the real and imaginary parts, maximum iterations T_max = 1000, tolerance 1e-6.

```
INITIALISE (section V-D, p.18)
  sigma^2: from the Toeplitz estimate of E[y y^H] built with the AQNM model (eq. 69; the scale term is garbled in the text layer), averaging the lower quarter of its eigenvalues
  z_ext_A = 0 ; v_ext_A = 10 (componentwise)
  rho = 0.5 ; gamma from its closed form (section V-D)
  theta_i: for each new component, project exp(|r^H diag(v^-1) a(theta)|^2 / N) onto a von Mises distribution and take its mean direction

FOR t = 1..T_max:
  # Module B: componentwise MMSE with the exact cell likelihood (diagonal EP)
  for each measurement n and each of Re and Im:
      tilt = truncated Gaussian on the observed cell, cavity N(z_ext_A, v_ext_A)
      z_post_B, v_post_B = mean and variance of tilt                     # eqs. 47-48
  v_ext_B = 1 / (1/v_post_B - 1/v_ext_A)                                 # eq. 51a
  z_ext_B = v_ext_B * (z_post_B/v_post_B - z_ext_A/v_ext_A)              # eq. 51b
  y_tilde = z_ext_B ; v_tilde = v_ext_B                                  # pseudo measurement
  sigma^2 = (||y_tilde - z_post_B||^2 + sum(v_post_B)) / N               # eq. 58 (approximate)
  # Module A: VALSE with heteroscedastic noise diag(v_tilde)
  J (eq. 38a), h = A_hat^H Sigma^-1 y_tilde (eq. 38b)
  support: greedy flip search maximising ln Z(s) (eqs. 42-43; appendix X-A updates), stop when no single flip improves
  C_S = (J_S + gamma^-1 I)^-1 ; w_S = C_S h_S                            # eq. 40
  rho = |S| / N ; gamma = (||w_S||^2 + tr C_S) / |S|                     # eq. 45
  for each active i: update theta_i (von Mises), a_i = E[a(theta_i)]     # eqs. 30, 36-37
  z_post_A = A_S w_S ; v_post_A from eq. 64                              # eqs. 63-64
  v_ext_A = 1 / (1/v_post_A - 1/v_tilde)                                 # eq. 67
  z_ext_A = v_ext_A * (z_post_A/v_post_A - y_tilde/v_tilde)              # eq. 68
  if ||z_hat(t) - z_hat(t-1)||^2 / ||z_hat(t)||^2 < 1e-6: break

OUTPUT: K_hat = |S| ; theta_hat (active) ; w_hat ; z_hat = sum_i a_hat_i w_hat_i
```

Notes for re-implementation: (a) the greedy search is a local search with no global guarantee; the paper says O(K_hat) steps are often enough (section IV-B, p.13). (b) The paper gives no pseudo-code for the von Mises step and refers to Heuristic 2 of VALSE [20]. (c) The authors' code is the authoritative reference for the details; it was not read.

### 3.5 Computational cost (the paper's claim; no timings)

- Module B: O(N K_hat) per iteration.
- Module A: dominated by the support search, O(N K_hat^3), and the von Mises step, O(N^2) (section V-E, p.19).
- The paper says the cost is comparable to VALSE and low compared with atomic-norm minimisation (section I-B, p.4). No runtime numbers are given, so the cost claim is qualitative.

---

## 4. Assumptions

| Item | Paper | Our project | Consequence for comparison |
|---|---|---|---|
| Noise | additive, circular complex Gaussian, i.i.d. | additive complex noise on Z (SNR -10..25 dB) | Same family. |
| Noise variance | not given to the algorithm; learned by EM (eq. 58); initialised from covariance eigenvalues | known to our quantiser-ML baseline (summary: ML loses 0.04-0.07 Pd when its SNR is off by 5 dB) | The paper's method is more forgiving about the noise level. Not analysed at 1 bit. |
| Number of components | unknown; K_hat estimated from the support | L = 3, known | Our baselines get the true L. The paper's K_hat is not tested against our L. |
| Dither | none | none | Same. |
| Quantiser scale | range +-3 sigma_z/sqrt(2) with sigma_z^2 about K (true signal scale) | peak AGC over 256 beam samples | The paper's scale is an oracle; our AGC is data-derived. Matching needed. |
| Snapshots | one snapshot (N samples) | one snapshot (256 samples) | Same. No multi-snapshot version here. |
| Grid | none; N candidate components with continuous frequencies (N is also the number of measurements in their setting) | none | Grid-free in both. |
| Quantiser type | uniform, Re and Im separately; 1-bit = sign | b-bit mid-rise, Re and Im separately; 1-bit = sign | Same structure. At 1 bit the scale does not matter. |
| Model structure | 1-D Vandermonde | 2-D beam-domain, not Vandermonde | Major mismatch (section 2.3). |
| Termination | relative change below 1e-6, or 1000 iterations (p.21) | not in our summary | Report it. |
| Simulation weights | magnitude N(1, 0.04), phase uniform (p.21) | not in our summary | Nearly constant magnitudes in their simulations. |
| Frequency separation | wrap-around distance greater than 2 pi / N (p.21) | not in our summary | Their scenes are well separated. Our bank may not be (UNVERIFIED). |

---

## 5. Reported results (numbers and conditions)

The paper's text contains no results table and no NMSE, MSE or probability values. All quantitative results are in plots (Figs. 4-9). I did not digitise the plots, so no curve values are quoted. The exact numbers in the text are the simulation settings.

Settings (section VII, pp.21-27):
- Synthetic: N = 100 (Figs. 4-6), N = 160 (Fig. 8); K = 2 (Fig. 4) or K = 3 (Figs. 5-7); K from 3 to 8 (Fig. 8). SNR 0, 20, 40 dB (Figs. 4-5); sweep (Fig. 6); 10 dB (Figs. 7-8). 500 Monte Carlo trials unless stated (p.21). Stopping rule: relative change below 1e-6 or 1000 iterations (p.21).
- Metrics: NMSE of z (multi-bit); debiased NMSE for one bit, because magnitude cannot be recovered from one-bit data (p.21); MSE of frequencies in dB, averaged only over trials where all algorithms get the model order right (p.21); the probability that K_hat = K.
- Baselines: VALSE with the additive quantisation noise model (AQNM) input (p.21), and VALSE on unquantised data. Also the CRB for the quantised model (section III, p.7), with K known.
- Real data (section VII-F, pp.25-26): 32-element horizontal line array (SW06 experiment), element spacing 15 m, design frequency 50 Hz (30 m wavelength). The text says "The first 1000 snapshots (8494 snapshots in total) are used" (ambiguous, UNVERIFIED). Data variance 0.0136 was used to design the 2-bit quantiser.

Qualitative results (paraphrased, with locators):

| Claim | Where | Figure | Conditions |
|---|---|---|---|
| Both VALSE-EP and VALSE converge in tens of iterations; at 0 dB they are similar and the gap grows with SNR. | section VII-B, pp.22-23 | Fig. 5 (p.23) | N = 100, K = 3, 1-3 bits, SNR 0/20/40 dB |
| At 1 bit, VALSE overestimates the model order at 20 and 40 dB; VALSE-EP gets the order right at 20 and 40 dB. | pp.21-22 | Fig. 4 (p.22) | N = 100, K = 2, 1 bit, SNR 0/20/40 dB |
| For 1 and 2 bits, VALSE first improves and then degrades with SNR; VALSE-EP keeps improving. | section VII-C, p.23 | Fig. 6 (p.24) | N = 100, K = 3, SNR sweep |
| The authors report that 2-bit VALSE-EP has the highest model-order success rate and reaches the CRB first; at high SNR it moves away from the CRB at 1 and 2 bits. | p.23 | Fig. 6 (p.24) | as above |
| VALSE-EP beats VALSE at 1 and 2 bits as N grows; its frequency error approaches the CRB faster. | section VII-D, p.24 | Fig. 7 (p.25) | SNR 10 dB, K = 3, N from 40 to 120 |
| All methods degrade as K grows, except VALSE at 1 bit, which first improves and then degrades. The sentence on K >= 4 and K >= 6 is ambiguous as printed (UNVERIFIED reading). | section VII-E, pp.24-25 | Fig. 8 (p.26) | N = 160, SNR 10 dB, K from 3 to 8 |
| Real data: at 1 and 2 bits, VALSE-EP removes the interferer source that VALSE keeps. | section VII-F, pp.25-26 | Fig. 9 (p.27) | 32-element array; qualitative only |
| The authors summarise the result as "near CRB". | abstract (p.1), conclusion (p.26) | Figs. 6-8 | CRB for known K (section III) |

What the paper does not give: any NMSE number, any detection-type metric (Pd), any comparison with NOMP, quantiser-ML, atomic-norm, SBL-type or other grid-based methods, any confidence interval, or any runtime.

---

## 6. Citable claims (paraphrased, with locators)

Only one short verbatim phrase is used in this file, in item 2.

1. Core idea (abstract, p.1): the quantised problem is replaced by a sequence of simple pseudo unquantised problems, each solved with VALSE. Cite for the method's overall structure.
2. Diagonal versus scalar EP (section IV, p.9): the authors state that "Since diagonal EP performs better than scalar EP" and use per-measurement variances. Cite for the choice of heteroscedastic pseudo-noise.
3. Heteroscedastic pseudo-noise (abstract, p.1; section I-B, p.4): the pseudo-measurement noise has a different variance for each component. Cite for the pseudo-observation model.
4. Model order inside the iteration (section I-B, p.4; section IV-B, p.13): the order is estimated within the iteration, via a greedy search on the evidence with no threshold. Cite as the authors' claim; the decision rule is local.
5. Relation to VALSE (section VI, p.20): VALSE-EP is not exactly equivalent to VALSE even without quantisation, because the noise variance is updated in a different module. Cite for a fair description of the relation.
6. Complexity (section V-E, p.19): the cost is said to be comparable to VALSE, with O(N K_hat^3) for the support search. Cite as the authors' statement; no runtime numbers exist.
7. Protocol (section VII-A, p.21): 500 trials; frequency MSE uses only trials where all algorithms get the order right. Cite for how the numbers were produced and as a caveat (selection effect).
8. CRB as benchmark (section III, p.7): the CRB is for deterministic unknown parameters with K known. Cite for what "near CRB" means.

---

## 7. Limitations and what the paper does not show

1. One dimension only. No 2-D (AoA/AoD) model, no separable dictionary, no beam-domain sampling model. Our problem is 2-D.
2. No comparison with NOMP, quantiser-ML, consistent NOMP, atomic norm, SBL or other grid-based methods. The only comparisons are VALSE with AQNM input and VALSE on unquantised data.
3. No numbers in the text; all results are in plots. No confidence intervals, no paired tests, and only 500 trials per point.
4. Frequency and model-order metrics are conditional on correct order (selection bias).
5. One-bit results use a debiased NMSE because magnitude is not identifiable (p.21). Our Pd metric does not need this, so the two metrics are not directly comparable.
6. The noise variance comes from an approximate EM rule (eq. 58). The paper does not analyse its behaviour at 1 bit, where the noise level is weakly identifiable (our summary reports the same for our data).
7. Initialisation uses the AQNM model, itself an approximation (eq. 69). The quantiser scale is an oracle (sigma_z is known).
8. The greedy support search is local; there is no global guarantee (appendix X-A, p.28).
9. "Near CRB" appears only in plots, and the authors note deviation from the CRB at high SNR for 1 and 2 bits (p.23).
10. The real-data test is qualitative (Fig. 9) with no ground truth beyond the cited reference [58].
11. No runtime measurements.
12. The arXiv PDF is a 2019 draft; the 2021 journal version may differ. Not checked.
13. The paper's SNR is a total-energy ratio; mapping it to our per-element SNR needs care.

---

## 8. Relevance to us

### 8.1 What to reuse (ideas, not code)

- The EP step that turns a quantised measurement into a Gaussian pseudo-measurement with its own variance (eqs. 46-51). It is a principled, closed-form way to use the exact cell likelihood that our quantiser-ML already uses. It does not add a new likelihood; what it adds is information passed between modules and a learned noise level.
- Per-measurement variances as a model input. For a network, the cell width (or the posterior variance of each measurement) could be an extra input channel, not only the cell midpoint. This is my design suggestion, not a result of the paper, and its benefit is UNVERIFIED.
- Learning the noise variance inside the loop (eq. 58). Our ML baseline assumes the noise variance, so a learned-noise variant could be tested as an extra baseline.

### 8.2 Must it be a baseline?

Not as published. The method is one-dimensional, has no published numbers that we can match, and has no 2-D version. Making it a baseline for the 16x16 case means building a 2-D extension ourselves. That would be our own derivation, unverified by the paper, and must be labelled as such. Recommendation: cite it as the closest grid-less quantised Bayesian method; do not present our own 2-D extension as "VALSE-EP" unless the derivation is written down and checked. The baselines our summary already supports (NOMP, consistent NOMP, quantiser-ML with the true L and known noise, and the ML variant under a 5 dB SNR mismatch) are the defensible comparison set. If time allows, the authors' generalised SBL method for one-bit DOA (ref [33], Meng and Zhu, IEEE Commun. Lett. 2018; not read) would be a grid-based Bayesian comparison.

### 8.3 Fair comparison rules if it is ever run

- Same metric: our paper-style Pd (1 degree, Hungarian matching), strict Pd and RMSE. The paper's metrics (NMSE, frequency MSE, order probability) are reported in addition, not instead.
- Same SNR definition, stated explicitly (their total-energy ratio versus our per-element ratio).
- Same quantiser: b bits, Re and Im separately, same scale rule. Their scale is an oracle and ours is peak AGC. Run both or state which.
- Same knowledge: VALSE-EP learns the noise and the order, while our ML gets the true L and noise. Run ML under noise mismatch and VALSE-EP with learned noise, and report both.
- Same scenes: test seeds 12000-12999 (project rule); paired bootstrap with 2000 resamples and seed 0 within SNR groups; claim only when the 95% CI excludes zero.
- Same hardware: CPU, same thread count, time per scene and memory reported.

### 8.4 Implications for network design

- The EP view suggests the network should output, or be trained with, a per-measurement pseudo-observation and variance rather than only a point estimate. This fits our finding that the network read-out is fast and close to ML, and that ML refinement from network seeds is best at 1-bit / 15 dB. The paper gives no evidence that EP-style refinement beats a good starting point; any such comparison would be ours.
- The paper's one-bit magnitude ambiguity (p.21) is a warning: a network trained for amplitude at 1 bit may learn an identifiability artefact. Keep the loss on angles only, as our Pd does.

---

## 9. "If a reviewer asks..."

1. "Why not compare with VALSE-EP, the grid-less quantised Bayesian method?"
   Honest answer: the published method is one-dimensional, has no published numbers we can match, and has no 2-D version. A 2-D extension would be our own derivation and would need checking before it could count as a baseline. We cite it, explain the mismatch (the beam-domain Y is not a Vandermonde sum, checked numerically for one DFT codebook), and compare with the classical methods we already have.

2. "VALSE-EP learns the noise and the model order, but your ML is given the true L and the noise level. Isn't that unfair to VALSE-EP?"
   Honest answer: the asymmetry is real and runs both ways. Our ML is an oracle for L and the noise level, which makes it stronger than a practical estimator. VALSE-EP is not given them, which makes it harder. A fair report shows both: ML with the true noise and with a 5 dB mismatch (our summary: loses 0.04-0.07 Pd), and a learned-noise variant. At 1 bit the noise level is not identifiable from the data (our summary), so learned noise may not help there. The paper does not analyse this.

3. "The paper claims near-CRB performance. Does that mean your detection rate is near optimal?"
   Honest answer: no. The paper's CRB is the Cramer-Rao bound for unbiased estimates of frequencies and weights when K is known and the parameters are deterministic (section III, p.7). It concerns estimation error, computed over trials with the right order. Our Pd (within 1 degree after matching) is a detection-type metric, and our summary contains no CRB for Pd. Our CRB results (IABC-v2) are for phase and gain errors, a different quantity. "Near CRB" from this paper cannot be transferred to our Pd.

---

## 10. Glossary (plain English)

- Line spectrum: a signal that is a sum of a few pure tones (sinusoids), each with one frequency and one strength.
- Frequency (theta): how fast a tone oscillates; here it encodes an angle through the array geometry.
- Model order: how many tones there are (K). The paper estimates it; we fix L = 3.
- On-grid / off-grid / grid-less: on-grid methods allow frequencies only from a fixed list; off-grid methods refine them continuously; grid-less methods never use a list.
- Quantiser and bit depth b: maps a fine value to one of 2^b levels. One bit means sign only.
- Cell: the interval of fine values that map to one observed level.
- AGC (automatic gain control): scales the signal before quantising so the levels fit it. The paper's scale is an oracle; ours is peak-based.
- Dither: random noise added before quantising to linearise it. Not used in the paper or in our project.
- Prior and posterior: the prior is the belief before the data; the posterior is the belief after.
- Bernoulli-Gaussian prior: each tone is on with probability rho; if on, its strength is Gaussian with variance gamma.
- Support: the set of tones that are on.
- Variational Bayes (VALSE): approximates the posterior with a simpler family, optimised iteratively.
- Expectation propagation (EP): approximates each non-Gaussian likelihood factor with a Gaussian message, refined iteratively.
- Extrinsic message: the part of a message that comes from the other module, with its own contribution removed (the cavity idea).
- Heteroscedastic noise: noise whose variance differs from measurement to measurement.
- Diagonal EP / scalar EP: diagonal EP keeps one variance per measurement; scalar EP uses one for all.
- AQNM (additive quantisation noise model): treats quantisation as scaled signal plus independent noise. Used here only for initialisation and as the VALSE baseline input.
- von Mises distribution: the circular analogue of a Gaussian, used as each frequency's posterior.
- MMSE: minimum mean squared error estimate, here per component.
- EM (expectation maximisation): iterative method for parameters such as the noise variance.
- NMSE, dNMSE, MSE: normalised mean squared error of the signal; debiased version (removes the unknown scale); mean squared error of frequencies (in dB here).
- CRB (Cramer-Rao bound): lower bound on the variance of any unbiased estimator.
- NOMP (Newtonized OMP, ref [10]): classical off-grid sparse method with Newton refinement; our classical reference.
- Atomic-norm minimisation: a convex grid-less method that solves a semidefinite program; the paper calls it expensive.
- Greedy search: at each step flip the one component that improves the objective most; stop when nothing improves.

---

## 11. Unverified items

1. Journal pages 77-95 and the DOI of the China Communications version (pages only from a search snippet; DOI not seen in any source I opened).
2. Content of the journal version (only the arXiv v3 draft was read).
3. First arXiv version date (14 Nov 2018), from a summarised page.
4. Printed forms of eq. (36), eq. (38a) (J_ii), eq. (69) (AQNM scale), and the denominator of eq. (58): the text layer is garbled, so check against the PDF before reuse.
5. Values in all figures (not digitised; no table in the text).
6. The meaning of the sentence on K >= 4 and K >= 6 (section VII-E, p.25), ambiguous as printed.
7. Behaviour of the authors' MATLAB code (only the README and file list were seen, through a summary).
8. Runtime (the paper gives none).
9. The "u, v" symbols in our project (not defined in RESULTS_SUMMARY.md).
10. Beam-domain non-Vandermonde result (section 2.3): verified only for a 16-point DFT codebook and one ULA response (valse_dft_structure_check.log), not for general W, F or the project's own codebook files.
11. Whether our bank's frequency separations resemble the paper's (section 4).
12. Any 2-D extension of VALSE-EP is not in the paper; it would be our own unverified derivation.

---

## 12. Bibliographic summary (for citation)

J. Zhu, Q. Zhang, and X. Meng, "Gridless variational Bayesian inference of line spectral from quantized samples," China Communications, vol. 18, no. 10, pp. 77-95, Oct. 2021 (IEEE Xplore document 9597637; pages and DOI UNVERIFIED). Preprint: arXiv:1811.05680v3 [cs.IT], 3 Sep 2019, titled "Grid-less Variational Bayesian Inference of Line Spectral from Quantized Samples."
