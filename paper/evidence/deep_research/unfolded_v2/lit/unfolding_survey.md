# How can an unfolded / learned decoder beat a converged ML-type greedy estimator (NOMP)? Verified literature survey

Saved 2026-10-16 (unfolded_v2/lit). Scope: the IABR-Net decoder, which is **unfolded from NOMP (Mamandipoor et al. 2016)**. The current version (lurd.pt, 4,803 parameters) learned a step that correlates 0.91-0.92 with the clipped Newton step, and it ties or loses to NOMP on the official bank (-0.0073*).

**How sources were checked.** Every source in Section 9 was opened with WebFetch on 2026-10-16. I read either the bibliographic record (Crossref, OpenAlex or the publisher/repository page) or the arXiv abstract, and usually also the arXiv/ar5iv full text. Column "Read" in the table says which. Numbers below are quoted from those pages; I recomputed nothing. Statements marked **INFERENCE** are mine, not the papers'. Project numbers come from `../../verify/REPORT.md`, `../../RESULTS_SUMMARY.md` and `../../finetune/adversarial/` (s3_adc_test.log, s6_latency.json), and are quoted, not re-run. `oracle_select.*` in this folder comes from an earlier, interrupted attempt. This survey does not use it.

---

## 0. Bottom line

1. **No source I found shows a learned or unfolded estimator beating a converged ML estimator on correctly specified data unless it uses extra information.** That information is either a prior (the training distribution) or a different decision rule. Where a learned method is compared with a converged, near-optimal classical method, it matches it:
   - learned VAMP recovers VAMP's own parameters [15];
   - DetNet never beats the sphere decoder [30];
   - SBRI-X-Net is comparable to its iterative algorithm at low SNR [35];
   - the DNN of Bialer et al. "attains" ML [6];
   - ALISTA's analytic weights are as good as learned ones [16].

   The lurd finding (the MLP re-learned Newton) is the expected result, not a failure of training.
2. **The low-SNR gap to NOMP is a threshold (outlier) effect of the ML objective** [1-3]. Local analysis (CRB, Taylor expansion) does not capture it. The project's own diagnosis agrees. At -10 dB, NOMP misses at least one component in 998 scenes. The oracle-seeded solution is all-correct in 55 of them, and in **0** of those 55 does it fit the data better than NOMP's solution. A better step rule or a better start cannot fix this; only a different objective or decision rule can.
3. **Four mechanisms in the literature give a real gain over converged ML/greedy methods:**
   - (a) a **Bayesian objective**: a prior, or the amplitude prior marginalised, as in VALSE [11], GMM-CME [12], score-based posterior sampling [13] and MAP chains [8]. The gain comes from the training prior and disappears when the prior is wrong [10].
   - (b) **Misspecification correction**: learning the dictionary or the impairments (mpNet [22], SubspaceNet [23]).
   - (c) **Search and selection over several hypotheses**, including a lower-than-ML threshold from a biased re-estimation plus a likelihood check [4], multipath matching pursuit [24], and learned model-order selection [7, 26].
   - (d) **The same accuracy for less compute**: LISTA's 10x [14], NeuPLSE versus VALSE (VALSE needs 137x the multiplications) [21], LDAMP versus BM3D-AMP (up to over 50x faster in the abstract, 7-60x in the tables) [18].
4. **The metric matters.** Paper-style Pd counts components with |error| <= 1 deg, so it is a per-component **hit-or-miss cost**. The Bayes rule for a hit-or-miss cost maximises the posterior mass in the window. For a vector parameter scored component by component, the rule maximises each **marginal** posterior, not the joint likelihood [37]. NOMP maximises the deterministic (conditional) likelihood. That is a different rule, and at low SNR the two can disagree.
5. **For 1-2-bit ADCs the strongest classical baseline is already a quantisation-aware NOMP (GNOMP) [33]** or the project's multi-start quantiser-likelihood ML. Unfolded one-bit methods match their iterative parents and do not beat them [35]. LoRD-Net beats a 700-iteration nML only by also learning a surrogate system model, without CSI [36]. The open headroom is the gap between ML from the true angles and multi-start ML: 0.626 vs 0.540 at 1-bit/15 dB and 0.439 vs 0.275 at 1-bit/0 dB. That headroom is again in the objective and the decision rule, not the search.

---

## 1. Threshold effect and estimators that beat ML there

| # | Work | What is learned / what changes | Classical comparison | Gain and conditions | Converged ML baseline? |
|---|---|---|---|---|---|
| [1] | Athley, TSP 2005 | Analysis, nothing learned | Deterministic and stochastic ML, single and multiple sources | Outliers cause the threshold. The CRB and asymptotic analysis miss it. For one source and one snapshot, ML is **not efficient even at high SNR** because of outliers. | n/a (analyses ML itself) |
| [2] | Athley, Asilomar 2002 | Analysis | Deterministic ML, multiple sources | A global, non-local analysis predicts ML MSE in the threshold region. | n/a |
| [3] | Richmond, TIT 2006 | Analysis (method of interval errors) | ML with estimated coloured noise | Pairwise error probabilities with a union bound predict the threshold-region MSE "with remarkable accuracy". | n/a |
| [4] | Vishnu & Ramalingam, arXiv 2020 | Classical. ESPRIT, then zero-padded ESPRIT-AC (biased, lower variance), a SNR-minus-threshold test, re-estimation of 2 sinusoids and gradient descent, **accepted only if the likelihood cost decreases** | MLE (coarse grid plus local refinement) | Threshold lower than MLE by about 10 dB (2 sinusoids, N=25, spacing 0.02 < 1/N), about 4 dB (3 sinusoids) and about 2 dB (DOA, M=10, 10 snapshots). | **No**. For p=4-5 the "ML" reference was shown not to be the global optimum (they found lower-cost solutions). Part of the gain is ML *search* failure. |
| [5] | Zhang, Shi, Gu, TSP 2023 | A Ziv-Zakai (Bayesian) bound for several DOAs, with an order-statistics prior | CRB | The ZZB stays valid in the a-priori (low-SNR) region, where the CRB is loose. | n/a (bound) |
| [6] | Bialer, Garnett, Tirer, ICASSP 2019 | A fully connected DNN (16-element ULA, one snapshot) that outputs the source count (1-4) and angles | ML only for 2 equal-gain sources over a 20 deg field of view; AP/MUSIC/OMP for 4 sources | DNN "attains" ML RMSE over a broad SNR range. **Parity, no gain over ML anywhere.** | Grid ML (2 sources only) |
| [7] | Barthelme & Utschick, TSP 2021 | MCENet (DoAs) and CovNet (model order, cross-entropy). Hybrid = NN start, then block-coordinate ascent on the **stochastic ML** likelihood until the change is < 1e-6 | GLS, SSR, SSR-iter, MVDR, MDL; "genie ML" (local SML from the true DoAs) as reference | Hybrid MCENet has **fewer outliers** than the model-based methods and "almost achieves" genie ML at >= 10 dB on the best 90% of realisations. Outliers dominate the gap over all realisations. CovNet beats MDL at few snapshots (2.6 s vs 505 s). | No converged global ML. The genie is local ML from the truth, like the project's oracle-seeded decoder. |
| [8] | Barthelme & Utschick, ChainNet, arXiv 2021/22 | A chain of grid classifiers (cross-entropy), one stage per source, interpreted as an approximation of **successive MAP** | SPICE, MUSIC, MCENet, BRClaNet, GramNet, each with or without likelihood refinement; genie ML | With refinement it is the only method that reaches genie ML (L=2). For L=3 it reaches genie ML in about 99% of realisations (N=10, 20 dB). About 1% outliers vs MUSIC's 10% (fully sampled, L=3, >= 10 dB). | Genie ML only |
| [9] | Lima de Oliveira & Bekooij, ICASSP 2022 (Deep-MLE) | A 1-D ResNet gives estimates for up to 4 targets, then MLE partially corrects them (single snapshot, MRA radar) | Pure MLE, pure NN | "Better" than either (repository abstract, no numbers). | Not stated |
| [10] | Diskin, Eldar, Wiesel, TSP 2023 | MSE-trained networks approximate the **MMSE estimator under the training prior** and are biased for individual parameter values. A bias-constrained loss (BCE) recovers an MVUE. | MLE, CRB | In distribution, the MSE-trained net has the best MSE. **Under a shifted prior it "significantly deteriorates"**, while BCE stays near the CRB. In SNR estimation the MSE net is unstable at low SNR, while BCE and the MLE are robust. | MLE where available |
| [11] | Badiu, Hansen, Fleury, TSP 2017 (VALSE) | Not learned. Variational Bayes with a Bernoulli-Gaussian amplitude prior and von Mises frequency posteriors (a von Mises prior can encode prior knowledge) | AST, GLS, EMaC, RAM | Beats them "at all SNR values". Near the CRB at >= 10 dB. Estimates the model order. | NOMP not compared |
| [12] | Koller, Fesl, Turan, Utschick, TSP 2022 | A GMM fitted to channel samples gives a closed-form conditional-mean estimator | Optimal CME (provably approached as the number of components grows) | MSE-optimal in the limit. A learned *prior*, not learned steps. | n/a |

**Lessons for this decoder (INFERENCE):**
- The only verified ways to beat ML at low SNR are a prior/Bayesian objective ([10], [11], [12], [8]) and multi-hypothesis checks against the likelihood ([4]).
- [10] warns that a gain obtained through the prior is a gain *on the benchmark's prior*. It must be reported as such and stress-tested off-prior (the project's sep_* and L != 3 banks).
- [7] and [8] show that hybrids reduce outliers relative to *practical* model-based methods. Their reference, like ours, is local ML from the truth, so they never claim to beat a converged ML.

## 2. Unfolding that learns the objective, prior or score (not only the step)

| # | Work | What is learned | Beats | By how much / conditions | Baseline converged? |
|---|---|---|---|---|---|
| [14] | Gregor & LeCun, ICML 2010 (LISTA) | Matrices and thresholds of ISTA / coordinate descent | ISTA / CoD at equal depth | Same error with about 1/10 the computation | Speed gain, not accuracy at convergence |
| [15] | Borgerding, Schniter, Rangan, TSP 2017 (LAMP / LVAMP) | Linear transforms and shrinkage functions, with Onsager correction kept | LISTA; AMP/VAMP | LAMP improves on LISTA. **For i.i.d. signals the parameters LVAMP learns coincide with VAMP's prescribed ones.** | Yes, and learning adds nothing once the classical algorithm is already Bayes-matched |
| [16] | Liu, Chen, Wang, Yin, ICLR 2019 (ALISTA) | Only step sizes and thresholds; weights computed analytically by coherence minimisation | LISTA | Comparable to LISTA, so the learned matrices were unnecessary | n/a |
| [18] | Metzler, Mousavi, Baraniuk, NeurIPS 2017 (LDAMP) | A DnCNN **denoiser = learned prior** inside D-AMP. The noise level sigma_hat = norm(z)/sqrt(m) selects the denoiser trained for that noise range. | BM3D-AMP, NLR-CS | +0.6 dB PSNR (Gaussian measurements), +1.4-1.5 dB (coded diffraction), 7-60x faster | The classical algorithm uses a hand-made prior (BM3D), so the gain is a better prior |
| [19] | Venkatakrishnan, Bouman, Wohlberg, GlobalSIP 2013 (Plug-and-Play) | Not learned. The proximal step of the prior inside ADMM is replaced by any denoiser | (framework) | Lets denoisers with no explicit optimisation formulation act as priors | n/a |
| [20] | Zhang et al., TPAMI 2022 (DPIR) | DRUNet denoiser with a **noise-level map input** (one network for all noise levels). In HQS, sigma_k decreases log-uniformly from 49 to the true noise level. | Model-based methods (EPLL etc.) | Example: 30.11 vs 20.82 dB (EPLL), deblurring on *Leaves* | The prior is the difference |
| [13] | Arvinte & Tamir, TWC 2023 | A score-based generative prior on MIMO channels, trained once; **posterior sampling** by annealed Langevin dynamics | Lasso, EM-GM-AMP, fsAD (atomic norm), WGAN, L-DAMP, regularised-pseudo-inverse "ML" | At least 3 dB NMSE better in distribution (-5 to 30 dB). At very low SNR, ML/Lasso/EM-GM-AMP are competitive. Under distribution shift, CS methods are more robust (error floor when trained on CDL-B and tested on CDL-A). | Gain comes from the prior and is fragile off-prior |
| [21] | Liu, Jiang, Yuan, TSP 2026 (NeuPLSE) | Fixed DFT initialiser; T unfolded gradient layers with learned step sizes and learned residual-mixing scalars; selection, regeneration and fusion for model order | OMP, VALSE, Superfast, ADMM-Net, ANM-Net, UNet | Close to the CRLB like VALSE, "more robust at low SNR". VALSE needs 137x the multiplications (1-D sparse; 5.15 vs 273.18 ms) | **NOMP not compared**; minimum spacing 15 deg in the sparse scenes |
| [22] | Yassine & Le Magoarou, TWC 2022 (mpNet) | The **dictionary** (initialised with nominal steering vectors), online and unsupervised | OMP with the nominal dictionary, LS | Approaches OMP with the *ideal* dictionary. Recovers after antenna damage, where OMP does not. | Gain comes only from model mismatch |
| [23] | Shmuel et al., TVT 2025 (SubspaceNet) | A surrogate covariance (a learned *input to* MUSIC / root-MUSIC) | MUSIC family, CNN, DA-MUSIC | Coherent sources, few snapshots, low SNR, array mismatch | Changes the objective's input, not the step |

**Lessons (INFERENCE):** unfolding beats its parent algorithm in three cases: (i) the parent's prior/denoiser was hand-made and weaker ([18], [20], [13]); (ii) the parent's model was wrong ([22], [23]); (iii) the comparison is at equal compute ([14], [21]). Case (i) is the only one available on the ideal official bank. It requires a prior over path configurations: the separation, the gain distribution, L=3. [13] and [10] show the gain shrinks or reverses under distribution shift.

## 3. Multiple hypotheses, learned model selection, stopping

| # | Work | Mechanism | Result |
|---|---|---|---|
| [24] | Kwon, Wang, Shim, TIT 2014 (MMP) | Tree search. Each candidate support spawns L children; the **final candidate is chosen by smallest residual**; a depth-first variant caps the number of candidates (N_max = 1 gives OMP). | MSE approaches oracle LS as SNR grows while single-path greedy methods saturate. Lower miss and false-alarm rates. MMP-DF is about 100x faster than breadth-first MMP. |
| [4] | Vishnu & Ramalingam 2020 | Alternative hypothesis (biased re-estimate) **accepted only if the likelihood improves** | 2-10 dB lower threshold than their (not always converged) MLE |
| [25] | Xu, Zhu, Fang, Zhang, Xu, arXiv 2022 (NOMP-CFAR) | NOMP proposes candidates; a CFAR test prunes the least likely; Newton refinement | CFAR without knowing the noise variance; about 1 dB loss vs NOMP with known variance (Pfa 1e-2, Pd 0.8) |
| [7] | Barthelme & Utschick 2021 | CovNet chooses the model order (cross-entropy) | Clearly more accurate than MDL at few snapshots; about 200x faster |
| [26] | Yang, Gao, Qian, Liao, SPL 2020 | Networks on covariance eigenvalues (regression ERNet, classification ECNet) | Beat AIC/MDL at low SNR and few snapshots |
| [27] | Izacard, Mohan, Fernandez-Granda, NeurIPS 2019 (DeepFreq) | Learned frequency representation plus a counting CNN | Lowest false-negative rate at low-to-middle SNR; CBLasso equal only at high SNR (1.71 s vs 42 ms). Training data had minimum separation >= 1/N; the paper does not test whether the net depends on it. |
| [8] | ChainNet | Successive MAP classification over a grid (posterior-like softmax) | See Section 1 |
| [28] | Chen, Dai, Li, Gao, Song, ICML 2020 | Unfolded network plus a **learned stopping policy** (variational; oracle stage, then imitation stage) | Better accuracy and inference efficiency than fixed-depth unfolding (the PMLR abstract gives no numbers) |

**Lessons (INFERENCE):** NOMP already keeps a single hypothesis with a likelihood (residual) acceptance rule. Its decision rule is "smallest residual". The project's min-residual oracle shows that selecting by residual gains only +0.0051. A multi-hypothesis decoder therefore helps only if **its selection score differs from the residual**: a posterior score with a prior or Occam/curvature terms, or a learned classifier as in [8]. On the official bank L=3 is given to every decoder, so model-order learning ([7], [26], [27], [25]) is not an accuracy lever there. It matters only for the L != 3 / L-unknown banks, and as a latency lever ([28], [22]).

## 4. Adaptive unfolding (SNR, impairment, bits, depth)

| # | Work | Conditioning | Result |
|---|---|---|---|
| [17] | Chen, Liu, Wang, Yin, NeurIPS 2021 (HyperLISTA) | Layer parameters computed **adaptively from earlier layers**, plus momentum; training is reduced to 3 hyperparameters | About the same as state-of-the-art LISTA in distribution; **better on unseen distributions** (sparsity, magnitudes) |
| [29] | Goutay, Ait Aoudia, Hoydis, SPAWC 2020 (HyperMIMO) | A hypernetwork takes R from the QR decomposition of H **and the noise std sigma**, and outputs the weights and per-iteration noise-variance estimates of an unfolded MMNet | About 0.65 dB behind MMNet retrained per channel; 1.85 dB ahead of OAMPNet and 2.85 dB ahead of LMMSE (SER 1e-3, 12x6, QPSK). The ML detector is shown as a reference and is not beaten. |
| [18] | LDAMP | sigma_hat from the residual selects the denoiser for that noise range | The authors attribute a small gap to the discretisation |
| [20] | DPIR | A noise-level map input plus a decreasing sigma_k schedule (annealing) | One denoiser for all noise levels and all iterations |
| [22] | mpNet | **Variable depth**: stop when norm(r)^2 <= sigma_tilde^2 * N (SC2; noise variance assumed known) | Under varying SNR, the adaptive version clearly beats fixed-depth versions |
| [28] | Learning to stop | Learned per-input depth | See Section 3 |
| [30] | Samuel, Diskin, Wiesel, TSP 2019 (DetNet) | Unfolded projected gradient, **one network for a whole distribution of channels** (and SNR) | Comparable to SDR/AMP, better than both for 16-QAM/8-PSK, **never better than the sphere decoder (ML)**. Fastest when batched. |

**Lessons (INFERENCE):** conditioning on the noise level is standard and cheap ([18], [20], [29]). It fits two known weaknesses in this project: (a) quantiser-likelihood ML loses 0.04-0.07 when its assumed SNR is off by 5 dB; (b) the default self-calibration gate shuts off under large impairments. Adaptivity gives robustness across conditions. None of these papers shows it beating a matched, converged ML at a fixed condition.

## 5. Quantisation-aware estimation and unfolding

| # | Work | Likelihood / what is learned | Result | Converged baseline? |
|---|---|---|---|---|
| [31] | Rangan, ISIT 2011 (GAMP) | Arbitrary componentwise output channel (quantisation fits this), approximate BP, state evolution | General framework | n/a |
| [32] | Mo, Schniter, Heath, TSP 2018 | EM-GM-GAMP / EM-VAMP with an exact cell-probability likelihood; **EM learns the Bernoulli-Gaussian-mixture prior** | Better than LS, ALMMSE, SPGL1 and QIHT with few bits; SPGL1 catches up only above about 7 bits | Prior learned by EM, not by a network |
| [34] | Zhu, Zhang, Meng, China Commun. 2021 (VALSE-EP) | VALSE plus expectation propagation: the quantised model becomes a sequence of pseudo-unquantised models with heteroscedastic noise | Close to the CRB; gridless; estimates the model order | n/a |
| [33] | Zhu, Zhang, Zhang, Fang, Qu, IEEE TAES 2026 (**GNOMP**) | NOMP with a **probit cell-probability log-likelihood**, Rao-test detection on an FFT grid, Newton on the quantised likelihood, CFAR stopping; a variant handles unknown sigma^2 | Detection SNR loss vs unquantised: 4.8 dB (1-bit), 2.3 dB (2-bit), 0.9 dB (3-bit). About 1 dB better than 1bMMRELAX; better detection than MVALSE-EP; 1.7 s vs 12.8 s (MVALSE-EP). Unknown sigma^2 costs about 3.5-4x the runtime. | This is the *classical* quantisation-consistent NOMP the project already uses as a baseline |
| [35] | Hu, Sun, Zhang, IEEE JSTSP (accepted per arXiv 2502.17473) | MAP with a Laplacian sparsity prior and a logistic one-bit likelihood; MM iterations unrolled (CNN layers replace the matrix inversion; FC layers predict the off-grid gaps) | SBRI-X-Net: RMSE **comparable to SBRI-X at 0-15 dB**, slightly lower at 20-50 dB; generalises to 35-50 dB outside the 0-30 dB training range | Matches its parent at low SNR |
| [36] | Khobahi, Shlezinger, Soltanalian, Eldar, TSP 2021 (LoRD-Net) | Gradient steps on the relaxed one-bit ML (Q-function) with a learned **positive-definite preconditioner** per layer and a learned surrogate system model; about 2.5k parameters | Beats nML **(given perfect CSI, 700 iterations) with no CSI and 30 layers**; BER 1e-2 at about 3 dB vs about 5 dB for DeepSIC (128x16) | The gain over nML shows that a learned preconditioner plus a learned "competitive" model can beat the plain relaxed-ML gradient on a non-convex quantised objective |
| [38] | Dreifuerst & Heath, TSP 2022 (SignalNet) | NN for 1-3-bit sinusoids with reconstruction inside the network | Beats periodogram/AIC/MDL/EM-VAMP in MSE. Its "worst-case learning threshold" is **often not exceeded for 1-bit data**: the net mostly learns the output distribution (the prior), not features of the data. | No ML baseline |


**Lessons (INFERENCE):** for 1-2-bit data, (i) the classical bar is GNOMP / multi-start quantiser-likelihood ML; (ii) unfolded one-bit algorithms match their parents ([35]), and LoRD-Net's win over nML is the only verified case of an unfolded quantised-likelihood method beating its non-learned counterpart; (iii) SignalNet [38] shows that at 1 bit a network can fall back on the prior. The project's ADC-fine-tuned CNN (0.419 at 1-bit/15 dB, vs 0.540 for multi-start ML) should be checked for the same behaviour.

## 6. When does learned beat converged ML? (synthesis, INFERENCE)

| Mechanism | Available on the official ideal bank? | Available under ADC / impairments? | Literature |
|---|---|---|---|
| Better step / search (learned Newton) | **No.** NOMP is converged; search headroom is +0.0051 | Small (multi-start ML already has 13 starts) | [14]-[16], [30], [35]: parity |
| Prior / Bayesian objective (MAP, MMSE, posterior mass) | **Yes, but only by using the benchmark prior** (separation pi/6, CN(0,1/L) gains, L=3) | Yes | [8], [10]-[13], [18], [20] |
| Misspecification correction | No (model is exact) | **Yes** | [22], [23], [36] |
| Decision rule matched to the metric (hit-or-miss gives marginal-posterior mass) | Yes (needs a posterior, so it includes the prior) | Yes | [37] |
| Same accuracy, less compute | Yes (latency) | **Yes**: 482 ms multi-start ML vs about 100 ms hybrid | [14], [21], [18], [33] |

---

## 7. Design recommendations for the IABR-Net decoder (unfolded from NOMP, Mamandipoor et al. 2016)

Each recommendation below gives what to build, the mechanism by which it could beat NOMP, the supporting literature, and a falsification test. The order is cheapest and most informative first.

### R0 (diagnostic, do first, CPU-light): compute the Bayes ceiling before building anything
- **What:** on a few hundred official-bank scenes per SNR, approximate the posterior p(u,v | Ht) under the generator's prior. The gains are CN(0, 1/L) and linear in the data, so they can be **integrated out in closed form**. The marginal likelihood is CN(vec Ht; 0, A A^H / L + sigma^2 I), multiplied by the separation prior. Sample it by importance sampling or MCMC seeded from NOMP, the CNN heat peaks and the truth. Report two numbers:
  - (a) the Pd of the per-component **marginal-posterior-mass-in-+-1-deg** estimator, i.e. the Bayes rule for the Pd metric [37];
  - (b) the Pd of the posterior mode.
- **Why:** the oracle-seeded 0.8261 is *not* an achievable target (at -10 dB, 0 of 55 all-correct oracle solutions fit better). (a) is the true ceiling of any decoder that uses this prior. If (a) is about 0.76, no design below can beat NOMP on the official bank and the paper should say so.
- **Literature:** [1]-[3], [5], [10], [11], [37].

### R1: multi-hypothesis decoder with a learned posterior score (the main bet for ideal hardware)
- **What:** generate K candidates per scene:
  - NOMP;
  - NOMP os8 rc6;
  - the CNN-seeded heat decoder;
  - MMP-style branches on the 2nd-3rd strongest residual peaks at the first detection [24];
  - the min-residual solution;
  - optionally posterior samples from R2.

  Score each candidate with a small network. Its inputs are:
  - the residual energy (deterministic log-likelihood);
  - the **gain-marginalised log-evidence** (R0);
  - per-path local curvature (a Laplace / posterior-mass-in-window term);
  - pairwise separations;
  - |gains|;
  - the CNN heat at the candidate angles;
  - sigma_hat.

  Train with listwise cross-entropy towards the candidate with the most hits (as ChainNet trains towards MAP [8]). Alternatively, select per path: the Bayes rule for Pd is component-wise.
- **Mechanism vs NOMP:** NOMP outputs the residual minimiser. This decoder approximates the Pd-optimal Bayes decision. Its score can prefer a candidate that fits slightly worse but has much more prior and posterior mass. That is exactly the low-SNR situation in verify/REPORT.md.
- **Honesty:** any gain over the residual rule comes from the benchmark prior (separation pi/6, the gain law, L=3) [10]. Report it as such. Evaluate on the sep_* banks and the L != 3 banks, with and without the prior features. Include "oracle selection among the K candidates" as the ceiling of this design, and "residual selection" (= +0.0051) as the classical control.

### R2: refinement on a learned-prior objective (MAP / plug-and-play) instead of learned Newton steps
- **What:** keep Newton (NOMP's step is already optimal for the likelihood), but change the objective to log p(Ht | u, v, x) + lambda(sigma_hat) * log p_phi(u, v, |x|). Options for the prior term:
  - (i) the closed-form gain prior (ridge-regularised LS gains plus the log-det Occam term from R0); this needs no learning;
  - (ii) a small learned configuration prior or score: a separation and ordering potential, or the CNN heat used as a log-prior, in the spirit of plug-and-play [19], LDAMP [18] and DPIR [20].

  Anneal the noise level used in the objective from large to nominal (DPIR schedule [20], annealed Langevin [13]). This smooths the likelihood and reduces outlier lock-in. Run a few stochastic chains and pass them to R1 as candidates.
- **Mechanism:** the fixed point moves from ML to MAP or posterior mass. This is the only route in the literature that beats a matched classical estimator at low SNR on correctly specified data.
- **Risk:** LVAMP = VAMP [15]. With an uninformative prior, nothing is gained. Off-prior, a learned prior hurts ([10], [13]), so gate lambda on a validation check and report the sep_*/L != 3 results.

### R3: condition the unfolded decoder on SNR, bits and impairment level (hypernetwork)
- **What:** estimate sigma_hat from the residual (as in LDAMP [18]) and append the ADC bit count / impairment flags. A hypernetwork [29] or per-layer adaptive rule [17] outputs:
  - step damping;
  - lambda for R2;
  - the number of candidates K and the stop threshold (mpNet-style SC2 [22]);
  - the self-calibration strength (exp5 StagedSC already has a learnable strength; make it conditional);
  - for ADC data, **the noise variance assumed inside the quantiser likelihood**.
- **Mechanism vs NOMP and ML:** classical NOMP and quantised ML use fixed thresholds and an assumed SNR. ML loses 0.04-0.07 Pd when its SNR assumption is off by 5 dB (adversarial s3). A learned sigma estimate and per-scene hyperparameters address this directly.
- **Evidence it helps:** robustness on unseen distributions [17]; near per-channel-trained performance without retraining [29].
- **Test:** mismatched-SNR ADC banks and the impairment-severity sweeps, against ML with the true SNR and with a +-5 dB SNR error.

### R4: unfolded GNOMP with learned preconditioning and annealing for 1-2-bit ADCs
- **What:** start from the ADC-fine-tuned CNN seeds. Unfold the GNOMP / quantiser-likelihood Newton iterations [33], with:
  - a learned per-layer positive-definite (or diagonal) preconditioner (LoRD-Net [36]);
  - a learned per-layer "temperature" (effective noise variance) that decreases to the nominal value, which smooths the probit likelihood (graduated non-convexity, [20]);
  - R1 selection over a few annealed runs instead of 13 random starts.
- **Mechanism vs the strongest classical method** (multi-start quantised ML: 0.540 / 0.682 / 0.275 / 0.476 at 482 ms): equal or better accuracy from fewer, better-placed starts plus smoothing, at less than 100 ms. The selector targets the remaining objective headroom: ML from the true angles reaches 0.626 / 0.719 / 0.439 / 0.540. The hybrid (ML from network seeds) already shows +0.032* at 1-bit/15 dB, so the seeds carry information the 13 starts lack.
- **Baselines to keep named:** GNOMP [33], multi-start quantised ML, the hybrid, and the CNN read-out alone.

### R5: train for the metric, with bias and prior-shift control
- **What:** replace MSE or Newton-imitation losses with a smoothed hit-rate loss, e.g. mean of sigmoid((1 deg - |err|) / tau) under Hungarian matching. Add the bias-constrained term of [10], or train on a mixture of separations and L values, so the learned score is not tied to sep = pi/6 and L = 3.
- **Mechanism:** the Pd metric is a hit-or-miss cost, so its Bayes rule is mode-seeking (posterior mass in the window) [37]. MSE training pulls towards the posterior mean, which can lie between two hypotheses and miss both. Bias control keeps the decoder honest on off-prior banks.

### R6 (latency, not accuracy, on the official bank): variable depth and learned stopping
- **What:** stop the cyclic rounds per scene when the residual meets a noise-scaled threshold (mpNet SC2 [22]), or with a learned stop policy [28]. Skip R1/R2 when sigma_hat is high (at 15-25 dB NOMP is already within 0.004-0.008 of the oracle). For L-unknown banks, add a learned order selector ([7], [26]) against NOMP-CFAR [25].
- **Mechanism:** the decoder runs at about 98 ms per scene, against a NOMP that is fast at high SNR. Spending compute only on low-SNR scenes can make the decoder match NOMP's accuracy at lower average latency.
- **Not an accuracy claim on the official bank,** because L is given there.

**Expected outcome (INFERENCE):**
- On the ideal official bank, any significant gain over NOMP will come from R1/R2 using the benchmark prior. It should be largest at -10 to 0 dB and should shrink or vanish on prior-violating banks.
- Under 1-2-bit ADCs and impairments, R3 + R4 compete with the strongest classical methods without exploiting the prior, through misspecification handling and SNR adaptation.
- If R0's Bayes ceiling is close to NOMP, report that as a null result.

---

## 8. Not found / caveats
- I found **no paper that unfolds NOMP itself** into a learned network. NeuPLSE [21] is the closest; it does not cite or run NOMP. Searches on "deep unfolded NOMP" returned only NOMP variants (NOMP-CFAR [25], GNOMP [33]).
- I found **no verified paper in which a neural estimator beats a converged ML estimator in the threshold region on prior-free, correctly specified data.** [6] reports parity. [7] and [8] compare against genie (truth-started) local ML. [4] beats an ML whose search was shown not to be global.
- Not verified (search hits only, not opened, not cited): Andrieu & Doucet 1999 (RJ-MCMC sinusoids); Johnston & Djuric 1994 (MMSE frequency estimation by importance sampling; repository record opened, but the abstract has no ML comparison, so it is not cited); a 2026 ICASSP "denoising diffusion model for DOA" (citation only).
- [35] numbers come from the arXiv HTML text only; its figures are not readable here. [9] and [28]: abstract-level only, no numbers.

## 9. References (all opened 2026-10-16)

| # | Title | Authors | Venue, year | DOI / URL | Verified | Read |
|---|---|---|---|---|---|---|
| 1 | Threshold region performance of maximum likelihood direction of arrival estimators | F. Athley | IEEE TSP 53(4):1359-1373, 2005 | 10.1109/TSP.2005.843717 | yes | Crossref + OpenAlex abstract |
| 2 | Threshold region performance of deterministic maximum likelihood DOA estimation of multiple sources | F. Athley | Asilomar 2002, vol. 2, pp. 1283-1287 | 10.1109/ACSSC.2002.1196987 | yes | Crossref + Chalmers record |
| 3 | Mean-squared error and threshold SNR prediction of maximum-likelihood signal parameter estimation with estimated colored noise covariances | C. D. Richmond | IEEE TIT 52(5):2146-2164, 2006 | 10.1109/TIT.2006.872975 | yes | Crossref + OpenAlex abstract |
| 4 | A Method with Lower-than-ML Threshold for Frequency Estimation of Multiple Sinusoids | P. Vishnu, C. S. Ramalingam | arXiv 2008.12608, 2020 (no venue listed) | arxiv.org/abs/2008.12608 | yes (preprint) | ar5iv full text |
| 5 | Ziv-Zakai Bound for DOAs Estimation | Z. Zhang, Z. Shi, Y. Gu | IEEE TSP 71:136-149, 2023 | 10.1109/TSP.2022.3229946 | yes | Crossref + arXiv abstract |
| 6 | Performance Advantages of Deep Neural Networks for Angle of Arrival Estimation | O. Bialer, N. Garnett, T. Tirer | ICASSP 2019 (per arXiv) | arXiv 1902.03569 | yes (journal DOI not checked) | ar5iv full text |
| 7 | A Machine Learning Approach to DoA Estimation and Model Order Selection for Antenna Arrays With Subarray Sampling | A. Barthelme, W. Utschick | IEEE TSP 69:3075-3087, 2021 | 10.1109/TSP.2021.3081047; arXiv 2009.12858 | yes | ar5iv full text |
| 8 | ChainNet: Neural Network-Based Successive Spectral Analysis | A. Barthelme, W. Utschick | arXiv 2105.03742, 2021/22 (no venue listed) | arxiv.org/abs/2105.03742 | yes (preprint) | ar5iv full text |
| 9 | Deep-MLE: Fusion between a Neural Network and MLE for a Single Snapshot DOA Estimation | M. L. Lima de Oliveira, M. J. G. Bekooij | ICASSP 2022, pp. 3673-3677 | 10.1109/ICASSP43922.2022.9747692 | yes | repository abstract |
| 10 | Learning to Estimate Without Bias | T. Diskin, Y. C. Eldar, A. Wiesel | IEEE TSP 71:2162-2171, 2023 (per DBLP listing) | arXiv 2110.12403 | yes (journal pages via DBLP snippet; DOI not checked) | ar5iv full text |
| 11 | Variational Bayesian Inference of Line Spectra (VALSE) | M.-A. Badiu, T. L. Hansen, B. H. Fleury | IEEE TSP, 2017 (DOI per arXiv record) | 10.1109/TSP.2017.2655489; arXiv 1604.03744 | yes | arXiv abstract + ar5iv full text (first 100k chars) |
| 12 | An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models | M. Koller, B. Fesl, N. Turan, W. Utschick | IEEE TSP 70, 2022 (DOI per arXiv record; pages 4109-4123 per author list, not opened) | 10.1109/TSP.2022.3194348; arXiv 2112.12499 | yes | arXiv abstract |
| 13 | MIMO Channel Estimation Using Score-Based Generative Models | M. Arvinte, J. I. Tamir | IEEE TWC 22(6):3698-3713, 2023 | 10.1109/TWC.2022.3220784 | yes | Crossref + ar5iv full text |
| 14 | Learning Fast Approximations of Sparse Coding (LISTA) | K. Gregor, Y. LeCun | ICML 2010, pp. 399-406 | mlanthology.org/icml/2010/gregor2010icml-learning | yes | proceedings abstract |
| 15 | AMP-Inspired Deep Networks for Sparse Linear Inverse Problems | M. Borgerding, P. Schniter, S. Rangan | IEEE TSP 65(16):4293-4308, 2017 | 10.1109/TSP.2017.2708040; arXiv 1612.01183 | yes | arXiv abstract |
| 16 | ALISTA: Analytic Weights Are as Good as Learned Weights in LISTA | J. Liu, X. Chen, Z. Wang, W. Yin | ICLR 2019 | mlanthology.org/iclr/2019/liu2019iclr-alista | yes | proceedings abstract |
| 17 | Hyperparameter Tuning is All You Need for LISTA | X. Chen, J. Liu, Z. Wang, W. Yin | NeurIPS 2021 | arXiv 2110.15900 | yes | arXiv abstract |
| 18 | Learned D-AMP: Principled Neural Network Based Compressive Image Recovery | C. A. Metzler, A. Mousavi, R. G. Baraniuk | NeurIPS 2017 (NIPS 30) | arXiv 1704.06625; proceedings.neurips.cc (2017 hash 8597a6cf...) | yes | ar5iv full text |
| 19 | Plug-and-Play Priors for Model Based Reconstruction | S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg | IEEE GlobalSIP 2013, pp. 945-948 | 10.1109/GlobalSIP.2013.6737048 | yes | Crossref + ORNL abstract |
| 20 | Plug-and-Play Image Restoration With Deep Denoiser Prior (DPIR) | K. Zhang, Y. Li, W. Zuo, L. Zhang, L. Van Gool, R. Timofte | IEEE TPAMI 44(10):6360-6376, 2022 | 10.1109/TPAMI.2021.3088914; arXiv 2008.13751 | yes | Crossref + ar5iv full text |
| 21 | Low-Complexity Neural Pursuit for Line Spectral Estimation (NeuPLSE) | C. Liu, W. Jiang, X. Yuan | IEEE TSP 74:3317-3332, 2026 | 10.1109/TSP.2026.3715966 (as printed in arXiv 2609.22731) | yes (DOI from the paper's own text, not Crossref) | arXiv HTML full text |
| 22 | mpNet: Variable Depth Unfolded Neural Network for Massive MIMO Channel Estimation | T. Yassine, L. Le Magoarou | IEEE TWC 21(7):5703-5714, 2022 | 10.1109/TWC.2022.3142737 | yes | Crossref + arXiv HTML full text |
| 23 | SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation | D. H. Shmuel, J. P. Merkofer, G. Revach, R. J. G. van Sloun, N. Shlezinger | IEEE TVT 74(3):4962-4976, 2025 | 10.1109/TVT.2024.3496119 | yes | OpenAlex abstract |
| 24 | Multipath Matching Pursuit | S. Kwon, J. Wang, B. Shim | IEEE TIT 60(5):2986-3001, 2014 | 10.1109/TIT.2014.2310482; arXiv 1308.4791 | yes (DOI from the search listing; full text read) | ar5iv full text |
| 25 | CFAR based NOMP for Line Spectral Estimation and Detection | M. Xu, J. Zhu, J. Fang, N. Zhang, Z. Xu | arXiv 2210.10302, 2022 | arxiv.org/abs/2210.10302 | yes (preprint) | arXiv abstract |
| 26 | Model-Aided Deep Neural Network for Source Number Detection | Y. Yang, F. Gao, C. Qian, G. Liao | IEEE SPL 27:91-95, 2020 | 10.1109/LSP.2019.2957673 | yes | OpenAlex abstract |
| 27 | Data-driven Estimation of Sinusoid Frequencies (DeepFreq) | G. Izacard, S. Mohan, C. Fernandez-Granda | NeurIPS 32, 2019 | arXiv 1906.00823 | yes | ar5iv full text |
| 28 | Learning To Stop While Learning To Predict | X. Chen, H. Dai, Y. Li, X. Gao, L. Song | ICML 2020, PMLR 119:1520-1530 | proceedings.mlr.press/v119/chen20c.html | yes | PMLR abstract |
| 29 | Deep HyperNetwork-Based MIMO Detection | M. Goutay, F. Ait Aoudia, J. Hoydis | IEEE SPAWC 2020, pp. 1-5 | 10.1109/SPAWC48557.2020.9154283; arXiv 2002.02750 | yes | Crossref + ar5iv full text |
| 30 | Learning to Detect (DetNet) | N. Samuel, T. Diskin, A. Wiesel | IEEE TSP 67(10):2554-2564, 2019 | arXiv 1805.07631 (journal DOI not checked) | yes | ar5iv full text |
| 31 | Generalized Approximate Message Passing for Estimation with Random Linear Mixing | S. Rangan | IEEE ISIT 2011, pp. 2168-2172 | 10.1109/ISIT.2011.6033942; arXiv 1010.5141 | yes | Crossref + arXiv abstract |
| 32 | Channel Estimation in Broadband Millimeter Wave MIMO Systems With Few-Bit ADCs | J. Mo, P. Schniter, R. W. Heath | IEEE TSP 66(5):1141-1154, 2018 | 10.1109/TSP.2017.2781644 | yes | Crossref + ar5iv full text |
| 33 | Line Spectrum Estimation and Detection With Few-Bit ADCs: Theoretical Analysis and Generalized NOMP Algorithm (GNOMP) | J. Zhu, H. Zhang, N. Zhang, J. Fang, F. Qu | IEEE TAES 62:3933-3951, 2026 | 10.1109/TAES.2025.3649136; arXiv 2307.00491 | yes | Crossref + arXiv HTML full text |
| 34 | Gridless Variational Bayesian Inference of Line Spectral from Quantized Samples (VALSE-EP) | J. Zhu, Q. Zhang, X. Meng | China Communications 18(10):77-95, 2021 | 10.23919/JCC.2021.10.005; arXiv 1811.05680 | yes | Crossref + arXiv abstract |
| 35 | Model-Based Learning for DOA Estimation with One-Bit Single-Snapshot Sparse Arrays | Y. Hu, S. Sun, Y. D. Zhang | IEEE JSTSP (accepted per arXiv), 2025 | arXiv 2502.17473 | partial (journal DOI not found) | arXiv HTML full text |
| 36 | LoRD-Net: Unfolded Deep Detection Network With Low-Resolution Receivers | S. Khobahi, N. Shlezinger, M. Soltanalian, Y. C. Eldar | IEEE TSP 69:5651-5664, 2021 | 10.1109/TSP.2021.3117503; arXiv 2102.02993 | yes | Crossref + ar5iv full text |
| 37 | EECE 522 lecture notes, Ch. 11B (MAP; hit-or-miss cost gives MAP; a per-component cost gives maximisation of the 1-D marginal posterior), following S. M. Kay, Fundamentals of Statistical Signal Processing: Estimation Theory, Prentice Hall 1993, Ch. 11 | M. Fowler (Binghamton) | lecture notes | ws2.binghamton.edu/fowler/.../EECE 522 Notes_26 Ch_11B.pdf | yes (notes read; the Kay book itself not opened) | PDF text extracted |
| 38 | SignalNet: A Low Resolution Sinusoid Decomposition and Estimation Network | R. Dreifuerst, R. W. Heath Jr. | IEEE TSP 70:4454-4467, 2022 | 10.1109/TSP.2022.3201336; arXiv 2106.05490 | yes | arXiv abstract |
| 39 | Newtonized Orthogonal Matching Pursuit: Frequency Estimation Over the Continuum | B. Mamandipoor, D. Ramasamy, U. Madhow | IEEE TSP 64(19):5066-5081, 2016 | 10.1109/TSP.2016.2580523; arXiv 1509.01942 | yes | arXiv abstract |
| 40 | Wireless Propagation Parameter Estimation with Convolutional Neural Networks | S. Schieler, S. Semper, R. Thomä | Int. J. Microw. Wirel. Technol. 17:892-899, 2025 | 10.1017/S1759078725000431; arXiv 2506.23621 | yes | arXiv abstract (details in ../../lit/baselines_survey.md R22) |
| 41 | Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing | V. Monga, Y. Li, Y. C. Eldar | IEEE SPM 38(2):18-44, 2021 | 10.1109/MSP.2020.3016905; arXiv 1912.10557 | yes | arXiv abstract |

Earlier verified surveys in `../../lit/when_dl_helps.md` and `../../lit/baselines_survey.md` cover more of the DL-vs-classical DOA literature. Of those papers, YOLO+Newton vs NOMP (Han et al., JSAC 2020) and CKNet+NOMP (Li et al., TWC 2025) are the only ones that run NOMP. Both keep Newton/NOMP refinement inside their own pipelines.
