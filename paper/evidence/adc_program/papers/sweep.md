# Sweep: learned angle (DOA/AoA/AoD) and channel-parameter estimators with 1-4-bit ADCs (2021-2026)

Prepared 2026-10-16 for the ADC programme. Folder: `D:/ai_ml_project/paper/evidence/adc_program/papers/`. Cached sources (arXiv HTML text, abstracts, OpenAlex and Crossref records): `C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/sweep/`. Earlier explainers in this folder that this sweep reuses: `onebit_unrolled.md` (Hu, Sun, Zhang 2025), `gnomp.md` (Zhu et al. 2026), `lordnet.md` (Khobahi et al. 2021), `mo2018.md` (Mo, Schniter, Heath 2018), `valse_ep.md` (Zhu et al. 2021), `foundations.md`. The literature survey of learned baselines is `D:/ai_ml_project/paper/evidence/deep_research/lit/baselines_survey.md`.

**Status legend.** VERIFIED = read in the paper's full text (arXiv HTML or the author-hosted PDF), in this session or in a cached copy. ABSTRACT = only the abstract (arXiv page, or a search snippet) was read. BIB-ONLY = only an index record (OpenAlex or search) exists; content not read. UNVERIFIED = a claim or number I could not check in the source. FIGURE-ONLY = the result exists only in a plot; I did not digitise it, so no plot value is quoted here as a fact.

**Honesty note on the search.** One requested title could not be confirmed: "One-bit DOA Estimation with Array Imperfections: A Learning-based Approach" (Azhdari, Farhang; OpenAlex record only, DOI 10.1109/AREE63378.2025.10880272 per the OpenAlex file name). The search returned no matching paper, so it is BIB-ONLY and UNVERIFIED. The same holds for "Multiple Measurement Vector-Based Deep Unfolded Network for One-bit DOA Estimation" (Zhan, Xi, Chen, ICICSP 2022): not found in any search, so it is UNVERIFIED and not used below.

---

## 1. Plain-English summary

Several recent papers use neural networks, or networks built by "unrolling" a classical algorithm, to find where signals arrive at an antenna array when the receiver keeps only one or a few bits per measurement. The most relevant ones are for direction-of-arrival (DOA) estimation with one-bit ADCs from a single time sample. A few are for channel estimation with 1-3 bit ADCs. None of them handles our exact problem: a 16 x 16 beam-domain matrix with two angles per path (AoA and AoD), 1-2 bits, and three paths. None of them compares against NOMP or against maximum-likelihood with the exact quantiser likelihood, which are our two strongest references. The learned papers compare against older baselines: MUSIC on unquantised data, one-bit sparse-MAP or SBL methods, OGIR, a plain CNN or DNN, BG-GAMP, EM-GM-GAMP, or Bussgang linear estimators. The strongest and closest learned comparator is Hu, Sun and Zhang (2025, JSTSP), a model-based unrolled network for one-bit single-snapshot DOA. It should be a baseline. The other learned papers mostly help us design the network; they are weaker as baselines.

---

## 2. The problem and the system model

Our system (project notation): Y = W^H H F + Z is a 16 x 16 complex beam-sample matrix, where H = sum over L = 3 paths of alpha_l a_r(u_l) a_t(v_l)^H, W and F are 16-point DFT codebooks, and Z is complex Gaussian noise. The ADC is a b-bit uniform mid-rise quantiser applied to Re and Im separately, with peak AGC A = max(|Re|,|Im|) over the 256 samples and step 2A/2^b.

Symbol mapping for the learned papers (VERIFIED from the cited text unless marked):

| Our symbol | Meaning in ours | Hu, Sun, Zhang 2025 (arXiv 2502.17473) | TransMUSIC (arXiv 2309.08174) | Fesl et al. 2023 (arXiv 2309.04014) | Nguyen et al. 2021 (arXiv 2107.11958) | Doshi & Andrews 2022 (arXiv 2211.08635) | Yeganegi et al. 2024 (arXiv 2405.09712) |
|---|---|---|---|---|---|---|---|
| Y (16x16) / vec(Y) | beam samples | y in C^N, one snapshot, N sensors | z(t), t = 1..L snapshots, M sensors | r in C^N (quantised), N antennas, P pilots | r, N antennas, K users | one-bit pilot measurements | one-bit dithered samples |
| W, F | beam codebooks | none (no beamspace) | none (antenna domain) | none (antenna domain) | none | hybrid precoder (pilots) | none |
| H | channel | none (DOA target only) | none | h in C^N channel | H in C^{N x K} | H in C^{Nr x Nt} | none (source covariance) |
| L | number of paths | K sources (sparse model) | estimated by a source-number head | not applicable | not applicable | not applicable | target count (2 or 3) |
| b | ADC bits | 1 only (sign), with a logistic variant | 1 only (sign of Re and Im, scaled by 1/sqrt 2) | B in {1,2,3} | b in {1,2,3} | 1 bit | 1 bit (dither) |
| A | AGC / scale | not used: one-bit data only see the ratio to noise | not used | not used (SNR = 1/sigma^2 normalisation) | not stated | genie-aided scale in NMSE (VERIFIED, Sec. V-C1) | dither scale |
| SNR | noise | "input SNR" (never defined; UNVERIFIED convention) | 0, 5, 10 dB (training); tested over a range | SNR = 1/sigma^2 | rho = 1/N0 | genie-aided | not stated in cached text |
| u, v | AoA, AoD | theta (one angle) | theta (one angle) | angular channel parameters | none | angular domain | theta (one angle) |

Key structural differences from ours: (i) every learned paper above is a single-angle (1-D) or channel-vector problem; (ii) none has a beamspace DFT codebook with a 2-D angle pair; (iii) only one-bit quantisers are used in the DOA papers; the channel papers stretch to 2-3 bits.

---

## 3. Method, step by step, per work

General recipe shared by the unrolled papers: each network layer equals one iteration of a classical algorithm, with learned step sizes or thresholds; training uses supervised pairs (quantised data, true angles) or, in some channel papers, quantised pilots only.

**3.1 Hu, Sun, Zhang (2025, JSTSP): SBRI-Net and SBRI-X-Net.** Full explainer: `onebit_unrolled.md`, checked against the published text, figures and pages 909-925. In plain words: a sparse Bayesian MAP estimator for one-bit single-snapshot data (probit likelihood plus a Laplacian-type prior, solved with MM/IRLS). The network unrolls that loop with learned parameters. The augmented variant SBRI-X replaces the probit with a logistic likelihood with shape parameters (a, b). The off-grid variant adds grid-gap updates. Training: 1,000,000 samples at SNR 0, 5, ..., 30 dB, 90/10 split. Key equations (from the explainer): the probit one-bit likelihood (each bit is a coin whose bias depends on the noiseless prediction measured in noise standard deviations); the logistic likelihood sig(x) = 1/(1 + a e^{-b x}) (eqs. 32-34).

**3.2 Hu, Sun, Zhang (2024, Asilomar; arXiv 2412.10976).** The earlier single-snapshot off-grid one-bit model. Model: sign of Re and Im of A(theta) s + n, single snapshot, first-order off-grid approximation. Method: sparse Bayesian MAP over a fixed grid, with Unrolled Block 1 (fixed-grid refinement; matrix inversion replaced by convolutional layers) and Unrolled Block 2 (updates amplitudes and grid gaps). Training: 100,000 samples, SNR 0-30 dB in 5 dB steps, two stages (block 1 first, then block 2 with MSE plus BCE). Evaluation: 1,024 Monte Carlo trials, two targets fixed at -10.28 and 20.56 degrees; detection = every error within 0.5 degrees; 91-point grid of 2-degree spacing; sparse linear arrays of 18 and 10 elements. Baselines: OGIR (off-grid iterative reweighting for one-bit DOA, Feng et al., IEEE TVT 2023, their ref. [25]) and CNN-DNN (Chung et al., Energies 2021, their ref. [34]).

**3.3 TransMUSIC (Ji, Mao, Xi, Chen; arXiv 2309.08174 v2).** VERIFIED, full text. Steps: (1) quantise each snapshot with the complex one-bit operator Q_1(.) = (1/sqrt 2)(sign Re + j sign Im); (2) a Transformer encoder processes the L snapshots in parallel, using [Re z; Im z] as tokens, and outputs an estimate of the noise subspace; (3) a MUSIC-type spectrum is computed from that subspace, with a learned peak finder for gridless DOA (loss: RMSPE, root mean squared periodic error); (4) a small fully connected network with softmax estimates the number of sources K (cross-entropy). Training is end to end with Adam (learning rate 1e-3), batch 64, 256,000 training samples, 25,600 validation and 25,600 test samples (Table I). Simulation: ULA with M = 8 half-wavelength elements; K in {2,3,4,5}; L = 200 snapshots; training SNR 0, 5, 10 dB. Sources and noise are complex Gaussian. Code: github.com/jijunkai/Transformer_Music (linked in the paper footnote).

**3.4 Yeganegi, Eamaz, Esmaeilbeig, Soltanalian (2024; arXiv 2405.09712; SAM 2024 per OpenAlex).** VERIFIED from the cached full text. Steps: (1) one-bit uniform dithering of the complex samples with two random dither sequences (Sec. II, eq. 7); (2) the sample covariance of the dithered one-bit data estimates the source covariance (eq. 8); (3) a LISTA network (learned iterative shrinkage-thresholding; Sec. III, eqs. 12-16) recovers a sparse vector from that covariance estimate; (4) DOAs are read from the support. Convergence is stated under Remark 1 and a theorem in Sec. III. Simulation: ULA with one-bit ADCs, several snapshots (count, sample sizes and element numbers are garbled in the cached text and are UNVERIFIED), two or three targets. Baseline: MUSIC only, in Fig. 2 (FIGURE-ONLY). Code: github.com/TaraEsmaeilbeig/one_bit_DoA_estimation (linked in the paper).

**3.5 Fesl, Turan, Bock, Utschick (arXiv 2309.04014; venue UNVERIFIED).** VERIFIED, full text. Steps: (1) a conditionally Gaussian latent model (GMM, mixture of factor analysers, or VAE) gives a prior on the channel; (2) conditioning on the latent variable makes the channel locally Gaussian, so the Bussgang decomposition applies; (3) a parameterised linear MMSE estimator is averaged over the latent components; (4) training uses quantised pilot observations only (no ground-truth channel), with covariance recovery from quantised samples. Setup in the main MSE figure: N = 64 antennas, P = 1 pilot (single snapshot), B in {1,2,3} bits, 3GPP channel with one or three clusters. Baselines (Sec. VI): BLS (Bussgang least squares with sample covariance), Buss-Scov (Bussgang with sample covariance from ground-truth channels), Buss-genie (Bussgang with true covariance; a bound, not feasible), EM-GM-GAMP (online EM plus GAMP in the angular domain, their ref. [17]), and a three-layer feed-forward DNN mapping pilots to channels. The MSE results are FIGURE-ONLY; the text says the proposed estimators beat BLS, Buss-Scov and EM-GM-GAMP over the whole SNR range.

**3.6 Fesl, Banna, Utschick (arXiv 2405.03542).** ABSTRACT and a WebFetch summary of the arXiv abstract page; the full text was not read. A Gaussian mixture model learned offline serves as the prior. A MAP estimate of the most responsible mixture component is used as side information inside the classical one-bit EM channel estimator. Baselines: "a simplistic Gaussian prior and current state-of-the-art channel estimators" (abstract; not named there). No numbers in the abstract. UNVERIFIED for any number.

**3.7 Nguyen, Nguyen, Swindlehurst (arXiv 2107.11958; venue UNVERIFIED).** Summary from a WebFetch of the full text, so no verbatim quotes are used. FBM-CENet (channel estimation) unfolds gradient ascent on a reformulated ML objective (the normal CDF is replaced by a sigmoid to avoid vanishing gradients). The pilot matrix is the first-layer weight, so the pilot is learned jointly with the estimator. Trainable: per-layer step sizes and a sigmoid scale beta. FBM-DetNet (data detection) unfolds projected gradient ascent on the same reformulated ML objective. B-DetNet unfolds projected gradient on the Bussgang-linearised model. Setup: K single-antenna users, N-antenna base station, b-bit uniform ADC on Re and Im with b in {1,2,3}, pilot length 5K, SNR rho = 1/N0. Baselines: a one-bit SVM channel estimator (their ref. 18), Bussgang linear estimators BMMSE and BWZF (refs. 3, 22), and a conventional DNN with joint pilot design (ref. 23). Results are qualitative in the text (no dB values): FBM-CENet slightly beats the SVM at one bit at medium to high SNR and significantly beats the other estimators at few bits (Fig. 7). Training: TensorFlow, Adam, batch 1000, learning rate 0.002 decayed by 0.97 every 100 epochs; training-set size not stated.

**3.8 Doshi and Andrews (arXiv 2211.08635; journal UNVERIFIED).** VERIFIED, full text. Steps: (1) a WGAN-GP is trained to produce beamspace channel realisations, one generator per CDL channel model (A-E), or one conditional CWGAN with an LOS/NLOS label; (2) the channel is estimated from one-bit pilot measurements by optimising the generator's latent input to maximise a correlation-based loss (GCE with quantisation, "QGCE"; Adam, step 0.1; Algorithm 1); (3) the one-bit pilot model uses sign quantisation of Re and Im with phase-shift quantisation at transmitter and receiver (Sec. II). Baseline: BG-GAMP (Bernoulli-Gaussian approximate message passing on the angular domain, their refs. [5][6]). NMSE uses a genie-aided scale factor kappa (Sec. V-C1: "a seemingly genie-aided scaling factor"). Antenna counts per experiment: UNVERIFIED here.

**3.9 Abstract-level learned papers (ABSTRACT).**
- Sheng, Chen, Yao, Wu, Chen (Electronics 10(6):738, 2021; DOI 10.3390/electronics10060738). Two networks: ADCnet (fully connected layers with ELU units, denoises the low-bit ADC output) and AtomicNet (estimates DOA from the denoised signal, with atomic vectors equal to steering vectors; loss balances reconstruction and sparsity). Body not read: MDPI returned HTTP 403. Abstract taken from a search snippet.
- Zhou, Liang, Zhang, Jiang, Li, Jin (arXiv 2408.10501, 2024): diffusion-model posterior inference for MIMO channels with quantised measurements; Stein's unbiased risk estimator for training on noisy data; claims a 10x lower latency than prior schemes. Abstract only.
- Karakoca, Nayir, Gurbuz, Qaraqe (arXiv 2306.08882, 2023): conditional GAN with a feature-attention denoiser for one-bit mmWave channel estimation. Abstract only.
- Park, Kim, Choi (arXiv 2605.08855, May 2026): a non-learned low-complexity beamspace denoiser for low-resolution-ADC mmWave systems (Bayesian binary hypothesis test under a Bernoulli-complex Gaussian prior, closed-form threshold, FPGA implementation). Abstract only. This is a classical comparator for the beamspace setting, not a learned one.
- He, Wang, Jin, Wen, Li (arXiv 2006.16628, 2020): LDGEC, a model-driven unsupervised network for beamspace wideband channel estimation with limited RF chains (beam squint); trained with a Stein's unbiased risk estimator; ADC bits not stated in the abstract. Abstract only.

**3.10 Bibliography-only records (BIB-ONLY; content UNVERIFIED).** Rahman et al. (IEEE Access 2024; one-bit ADC channel estimation with fewer pilots); Sun, Ren, Zhou (Globecom 2023; one-bit mmWave channel estimation); Li, Gao et al. (ICSPS 2025; MLP covariance reconstruction for one-bit DOA with array imperfections); Li, Zhao, Liu, Bi (Sensors 22(9), 2022; strongest AoA with a 1-bit hybrid architecture); Katyal, Bharadwaja H., Murthy (EUSIPCO 2024; deep-unfolding channel estimation and soft decoding with low-resolution ADCs); Li, El-Hajjar, Yang (IEEE OJCOMS 2021; two-stage few-bit channel estimation for mmWave localisation); Xu, Gao, Zhou, Ma, Zhang (IEEE TWC 2022, DOI 10.1109/TWC.2022.3200378 per OpenAlex; joint channel estimation and mixed-ADC allocation); Azhdari and Farhang (see the honesty note). The TransMUSIC conference record (ICASSP 2024, DOI 10.1109/ICASSP48485.2024.10447483, per OpenAlex) was not confirmed on the IEEE page.

---

## 4. Assumptions (what each method needs to know)

| Work | Noise variance known? | Number of paths known? | Dither? | AGC / scale | Snapshots | Grid? | ADC model |
|---|---|---|---|---|---|---|---|
| Hu 2025 (SBRI) | No: amplitudes are in noise units; likelihood shape fixed; logistic (a, b) hand-tuned for SBRI-X | Sparsity prior; K <= 3 in tests | No | Not needed (one bit) | 1 | Off-grid variant (first-order approximation) | sign, zero threshold |
| Hu 2024 (Asilomar) | Not stated (UNVERIFIED) | K = 2 in tests | No | Not needed | 1 | Fixed 91-point grid plus gap learning | sign of Re and Im |
| TransMUSIC | Not given; trained at SNR 0/5/10 dB | Estimated by a learned head | No | Not needed (one bit) | L = 200 (tested with fewer) | Gridless (learned peak finder) | 1-bit complex sign, 1/sqrt 2 |
| Yeganegi 2024 | Not stated | 2-3 targets in tests | Yes, uniform dither | Dither scale | Several (exact count UNVERIFIED) | Not stated | one bit with dither |
| Fesl 2023 | Yes: SNR = 1/sigma^2 is used; GMM trained offline from quantised pilots | Not needed (statistical prior) | No | Channel-power normalisation | 1 pilot in the main figure | Not needed | B in {1,2,3}, Bussgang |
| Fesl 2024 | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | one-bit EM |
| Nguyen 2021 | Yes (N0 in the likelihood) | Not needed | No | Not stated | Pilot length 5K | Not needed | b in {1,2,3} uniform |
| Doshi & Andrews 2022 | Partly: a genie-aided scale kappa is used for NMSE | Not needed | No | Genie scale (VERIFIED) | One pilot block | Not needed | one bit with phase-shift quantisation |
| Sheng 2021 | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | low-bit ADC |

The unrolled papers assume the learned parameters transfer across SNR. Hu 2025 tests only upward in SNR (35-50 dB) and never below 0 dB (explainer section 5.3). Its networks are trained over 0-30 dB.

---

## 5. Reported results (exact numbers where the text gives them; otherwise FIGURE-ONLY)

- **Hu, Sun, Zhang 2025 (JSTSP).** No results table. Figure readings by eye (+-0.01-0.02; `onebit_unrolled.md` section 5.2, FIGURE-ONLY): on-grid hit rate at 0 dB, SBRI-Net 0.56 vs parent SBRI 0.30; at 5 dB, 0.88 vs 0.59. On-grid, SBRI-X-Net vs parent SBRI-X: 0.70 vs 0.71 at 0 dB, 0.94 vs 0.96 at 5 dB. Sparser array with three targets at 0 dB: 0.56 vs 0.245 and 0.57 vs 0.065 for network vs parent (explainer, figure readings). Off-grid: the model-based networks beat the purely data-driven CNN-DNN (FIGURE-ONLY). Hit-rate threshold: 2 degrees on all angles (unit inferred from the 1-degree grid; UNVERIFIED).
- **Hu, Sun, Zhang 2024 (arXiv 2412.10976).** Quantitative results are FIGURE-ONLY (Figs. 4-6). The text says the proposed method "achieves the lowest RMSE and the highest detection rate" on the 18-element SLA and keeps lower RMSE on the 10-element SLA (Sec. V). Baselines: OGIR and CNN-DNN; no unquantised MUSIC, NOMP or quantiser-ML comparison.
- **TransMUSIC (arXiv 2309.08174 v2).** FIGURE-ONLY (Figs. 2-5): RMSPE vs SNR; probability of correctly estimating K vs SNR and vs L. The text says TransMUSIC with one-bit data "even outperforms those with unquantized data" (Sec. IV, Fig. 2 discussion), and that source-count accuracy drops sharply with fewer snapshots (Sec. IV). Baselines: MUSIC (unquantised), conventional beamformer, DA-MUSIC (unquantised, RNN-based), one-bit MUSIC (their ref. [15]). No NOMP, no quantiser-ML.
- **Yeganegi et al. 2024.** FIGURE-ONLY: Fig. 2(a)-(b) estimated DOAs versus MUSIC for two and three targets; Fig. 2(c) convergence vs epochs. No RMSE or detection numbers in the text. Baseline: MUSIC only.
- **Fesl, Turan, Bock, Utschick 2023.** FIGURE-ONLY MSE curves (Fig. 2 covariance recovery; Fig. 3 MSE vs SNR for B in {1,2,3}, N = 64, P = 1; Fig. 4 rate bound; Fig. 5 QuaDRiGa LOS). The text says the Buss-Scov estimator fails because the Bussgang theorem does not hold for the non-Gaussian channels (Sec. VI).
- **Fesl, Banna, Utschick 2024.** The abstract says only "significant performance improvement". No numbers read (UNVERIFIED).
- **Nguyen, Nguyen, Swindlehurst 2021.** Qualitative only (Figs. 7-10). Estimation: FBM-CENet beats the SVM at one bit (slightly, at medium to high SNR) and beats the others significantly at few bits. Detection: FBM-DetNet beats the other detectors (BER, Figs. 9-10). No dB values in the text.
- **Doshi and Andrews 2022.** VERIFIED text, NMSE gains: the individually trained WGAN-GP "outperforms BG-GAMP, by 0.5 dB in CDL-B and C, 1 dB in CDL-A and 5 dB for CDL-E and D" (BG-GAMP subsection and the following results paragraph). The one-label CWGAN is worse than the individually trained models and only "competitive with BG-GAMP" on NLOS channels.
- **Sheng et al. 2021.** Abstract only: DOA performance "degrades severely" with low-bit ADCs; ADCnet + AtomicNet improve it. No numbers read.

Complexity: the explainer for Hu 2025 gives O(N_iter N^3) for the iterative parent and O(KN) on-grid / O(KN^2) off-grid for the network (Table I, Sec. IV-E; VERIFIED in `onebit_unrolled.md` section 5.1). None of the other papers reports wall-clock latency in the text read here (Doshi and Andrews report spectral efficiency, not time; timing claims are UNVERIFIED).

---

## 6. Citable claims (quote at most 15 words; section given)

- C1. TransMUSIC, Abstract: "Numerical results demonstrate the superiority of the TransMUSIC algorithm" (9 words). Cite for: a Transformer front end can denoise one-bit subspace estimation for DOA.
- C2. TransMUSIC, Sec. V (Conclusion): "surpasses the classic MUSIC algorithm and other counterparts that rely on unquantized data" (13 words). Cite for: one-bit learned subspaces beat unquantised MUSIC. Caveat: the comparator is MUSIC at L = 200 snapshots, not a quantiser-aware ML.
- C3. Hu, Sun, Zhang 2024 (arXiv 2412.10976), Sec. VI (Conclusion): "the proposed approach outperforms state-of-the-art methods across a range of SNR scenarios" (12 words). Cite for: learned sparse-Bayes one-bit DOA, the precursor of the JSTSP paper; the baselines are OGIR and CNN-DNN only.
- C4. Hu, Sun, Zhang 2025 (JSTSP), Sec. V-A, p. 921 (the one verbatim quote in the explainer, checked in the full text): "SBRI-X-Net achieves comparable RMSE to SBRI-X at SNRs from 0 dB to 15 dB" (11 words). Cite for: an unrolled network matching its iterative parent on-grid; not a claim of beating it.
- C5. Doshi & Andrews 2022, Sec. V-C1 (BG-GAMP): "by a careful tuning of the sparsity rate" (8 words), and "Such sparsity rates cannot be obtained in practice from pilot measurements alone" (12 words). Cite for: a caution about tuned baselines; our NOMP and quantiser-ML must be tuned by the same rule.
- C6. Fesl, Turan, Bock, Utschick 2023, Sec. I (Introduction): "adaptable to different SNRs, pilot sequences, and quantization bits" (9 words). Cite for: generative latent priors can be reused across SNRs and bit depths without retraining.
- C7. Fesl, Turan, Bock, Utschick 2023, numerical results (Sec. VI): "are outperformed with a considerable performance gap over the whole SNR range" (12 words), referring to BLS, Buss-Scov and EM-GM-GAMP. Cite for: a learned prior beats an EM plus GAMP baseline on channel estimation (figure-based, no exact dB number).
- C8. Yeganegi et al. 2024, Fig. 2 caption: "LISTA converges even under coarse quantization of high-resolution measurements" (9 words). Cite for: unrolled LISTA with one-bit dithered data converges; no accuracy number is given.

---

## 7. Limitations and what the papers do not show

- No paper in this sweep runs NOMP, GNOMP, quantiser-aware ML, or self-calibration as a baseline (checked in the full text, or in the abstract for the abstract-only papers). The earlier survey agrees: only Han et al. (IEEE JSAC 2020, DOI 10.1109/JSAC.2020.3000836) and Li et al. (IEEE TWC 2025, DOI 10.1109/TWC.2025.3548626) run NOMP, and both keep NOMP's Newton refinement inside their pipeline (`lit/baselines_survey.md`, verified there).
- Most learned DOA results are FIGURE-ONLY. Only a few numbers appear in the text (Doshi and Andrews dB gains; the Hu 2025 figure readings). No paper reports a table with Pd, RMSE and confidence intervals.
- Single-angle problems: every DOA paper estimates one angle per path. None estimates a pair (AoA, AoD) on a 2-D beam grid.
- Bit depth: the DOA papers (Hu 2025, Hu 2024, TransMUSIC, Yeganegi) use one bit only. Two- to four-bit DOA results appear only in Sheng 2021 (abstract only). The channel papers (Fesl 2023, Nguyen 2021) cover 1-3 bits.
- Known noise: Hu 2025 does not know the noise variance and never defines SNR. Fesl 2023 and Nguyen 2021 use SNR = 1/sigma^2 (or 1/N0) explicitly. Doshi and Andrews use a genie scale for NMSE.
- Test range: Hu 2025 tests only upward in SNR (35-50 dB), not below 0 dB, where one-bit data are hardest.
- Snapshots: TransMUSIC uses L = 200 snapshots in its main experiments. The one-bit single-snapshot case appears only in Hu 2025 and Hu 2024, and in the channel papers with P = 1 pilot.
- Code: TransMUSIC (github.com/jijunkai/Transformer_Music) and Yeganegi et al. (github.com/TaraEsmaeilbeig/one_bit_DoA_estimation) link code in their papers. The explainer records no code for Hu 2025. For Fesl 2023/2024, Nguyen 2021 and Doshi and Andrews, code availability is UNVERIFIED.
- Real data: TransMUSIC lists real-data experiments as future work (Sec. V).
- None of the unrolled works compares against a Cramer-Rao bound with plotted curves. Hu 2025 cites a conservative CRB (their ref. [2]) without plotting it (explainer section 7).

---

## 8. Relevance to us

**What to reuse.**
- From Hu 2025 (and its predecessor): the probit likelihood as the one-bit case of our quantiser likelihood; the logistic variant as a tunable-likelihood idea for 2 bits (the paper does not cover 2 bits); the unrolled MM/IRLS loop as a learnable baseline shape (`onebit_unrolled.md` has pseudo-code).
- From TransMUSIC: parallel processing of snapshot tokens with a Transformer; the learned source-count head; the end-to-end periodic angle loss. For us, the "snapshots" could be rows or columns of Y. That mapping is our own design choice, not theirs, and it is UNVERIFIED that it works.
- From Yeganegi: dithering as an option (we use none); the convergence reasoning for LISTA.
- From Fesl 2023 and Nguyen 2021: learning the prior from quantised data only (no ground-truth channels), and joint pilot/beam design. Our beam codebooks are fixed DFT; a learned codebook would be a new contribution, not a reproduction.
- From Doshi and Andrews: the caution about genie-tuned baselines (C5).

**Must one be a baseline?** Yes: Hu 2025 (SBRI-X-Net and SBRI-Net) should be a baseline in the one-bit single-snapshot setting. Implement it with the same quantiser (sign at b = 1) and the same metric (paper-style Pd and strict Pd with Hungarian matching, not their 2-degree hit rate). It is the closest learned, model-based, one-bit single-snapshot comparator. TransMUSIC is a second candidate for the one-bit multi-snapshot setting. Its published comparators are unquantised, so it must be re-run with our sign quantiser and our snapshot count. Fesl 2023 (EM-GM-GAMP and BLS) is a channel-estimation baseline, not DOA; include it only if we add a channel experiment. Nguyen 2021 (FBM-CENet) is the closest conceptual cousin of our hybrid (learned start plus quantiser-aware refinement) and should be cited as "unfolded from quantised gradient ascent". Its code is not known (UNVERIFIED).

**Fair comparison rule.** Same bits, same quantiser (peak AGC, mid-rise), same SNR definition (our per-scene SNR from the noise variance we set), same L (3, known to all decoders), same metric (paper-style Pd and strict Pd with 1-degree tolerance, mean over SNR groups), same paired bootstrap (2000 resamples, seed 0). Learned baselines are trained on seeds 80,000,000+ and tested on 12000-12999. Quantiser-ML uses 13 starts plus EM as in `finetune/adversarial/mlq.py`.

**What it implies for our network design.**
- A quantiser-aware refinement (Hu 2025's probit or logistic likelihood, or our exact cell-probability likelihood) should follow the learned seed, as in our hybrid. Nothing in this sweep shows a learned step alone beating an exact-likelihood refinement.
- Dither is not a free option. Yeganegi relies on it for the covariance route; our pipeline has no dither, so that route would need a hardware change.
- A learned prior trained on quantised data only (Fesl 2023/2024) might give a better seed than the CNN alone if the channel statistics are stable. This is an UNVERIFIED design idea for us.

**CPU feasibility (not measured).** Hu 2025 networks are small (O(KN) on-grid), but their 10^6-sample training set is large; a 1-D ULA with N = 16 is cheap per sample. TransMUSIC (M = 8, 256,000 samples, Table I) is a Transformer; a 16 x 16 problem on 8 cores is plausible but unmeasured. The Fesl GMM prior trains offline and is cheap at test time. Nothing here was run. Treat feasibility as UNVERIFIED until timed.

---

## 9. "If a reviewer asks..."

1. "Why not compare against a learned method that is published for 2-D AoA/AoD?" Honest answer: none of the works in this sweep estimates a 2-D AoA/AoD pair on a beam grid with 1-2 bits. The closest one-bit learned DOA papers (Hu 2025, TransMUSIC) are 1-D. We compare against the closest 1-D models after adapting them to our measurement model, and we say so.
2. "Are the learned gains real, or only against weak baselines?" Honest answer: most learned DOA papers compare with MUSIC on unquantised data, OGIR, or CNN-DNN. Those are weak baselines for a 1-2 bit problem. Our own tests found that the network read-out alone reaches 78-92% of quantiser-ML at about 1/80 of its cost, and that a network-seeded hybrid beats multi-start ML only at 1 bit, 15 dB (+0.032*). The sweep does not change those numbers; it adds comparators, none of which we have run.
3. "Does the TransMUSIC claim (one-bit beats unquantised MUSIC) contradict your ADC results?" Honest answer: no. It compares against unquantised MUSIC with L = 200 snapshots, not against quantiser-ML. It is consistent with our finding that quantisation-aware methods do well at low bits, but it does not show that a learned model beats quantiser-ML. We have not run TransMUSIC.

---

## 10. Glossary (plain English)

- AoA / AoD: angle of arrival / angle of departure; the directions a path comes from or leaves towards.
- ADC: analogue-to-digital converter. "b-bit" means each real and imaginary part is stored with b bits.
- AGC: automatic gain control; scales the signal so the quantiser range fits. Our A is the per-scene peak.
- Bussgang decomposition: writes a quantised output as a scaled input plus uncorrelated distortion; gives a linear estimator when the input is Gaussian.
- BG-GAMP: Bernoulli-Gaussian approximate message passing; a sparse-recovery algorithm used as a baseline.
- CDL / WGAN-GP: channel models from the 3GPP standard; a generative adversarial network trained with a gradient penalty.
- CFAR: constant false-alarm rate; a threshold that does not depend on the data (GNOMP).
- Dither: a small known random noise added before quantisation, so the quantiser error averages out.
- DOA: direction of arrival.
- EM: expectation-maximisation; an iterative maximum-likelihood method.
- GMM / MFA / VAE: Gaussian mixture model / mixture of factor analysers / variational autoencoder; latent-variable prior models.
- GNOMP: generalised NOMP for few-bit quantised line spectra (Zhu et al. 2026).
- LISTA: learned iterative shrinkage-thresholding algorithm; ISTA with learned matrices and thresholds.
- MUSIC: a subspace method for DOA that uses the eigenvectors of a covariance matrix.
- NOMP: orthogonal matching pursuit with Newton refinement (Mamandipoor et al., 2016); our classical reference.
- OGIR: off-grid iterative reweighting for one-bit DOA (Feng et al., IEEE TVT 2023), a baseline in Hu 2024.
- Pd: detection probability; here the fraction of scenes where all angle errors are within a tolerance.
- Probit / logistic likelihood: the probability of a one-bit output as a Gaussian CDF (probit) or a sigmoid (logistic).
- RMSPE: root mean squared periodic error (angle error on a circle); the loss used by TransMUSIC.
- SBRI: sparse Bayesian recovery from one-bit data (Hu et al.); SBRI-Net and SBRI-X-Net are the unrolled versions.
- Snapshot: one time sample of the received vector. "Single-snapshot" means one such sample per problem.
- Unrolling / unfolding: turning K iterations of an algorithm into K network layers whose parameters are learned.

---

## Summary ranking table (closest first, with verification status)

| Rank | Work (year, venue, DOI or arXiv) | What it estimates | Data model (snapshots, array, bits, noise known?) | Method | Baselines it beat (as reported) | Numbers (status) | Stronger than our current baselines? | Use |
|---|---|---|---|---|---|---|---|---|
| 1 | Hu, Sun, Zhang, "Model-Based Learning for DOA Estimation With One-Bit Single-Snapshot Sparse Arrays", IEEE JSTSP 19(6):909-925, 2025, doi 10.1109/JSTSP.2025.3627141, arXiv 2502.17473 | 1-D DOA, on and off grid | 1 snapshot, ULA/SLA, 1 bit, noise unknown, SNR undefined | SBRI MM/IRLS unrolled (SBRI-Net, SBRI-X-Net) | SBRI, SLIM, LIKES, SPICE, OGIR, CNN-DNN (figure-only) | Hit rate 0 dB: 0.56 vs 0.30 (SBRI); 5 dB: 0.88 vs 0.59 (figure readings, +-0.02). Text VERIFIED, figure values read by eye | Yes: the strongest one-bit single-snapshot learned comparator | Baseline; reproduce on our 1-bit data |
| 2 | Ji, Mao, Xi, Chen, "TransMUSIC: A Transformer-Aided Subspace Method for DOA Estimation with Low-Resolution ADCs", arXiv 2309.08174 v2 (Jan 2024); conference record per OpenAlex: ICASSP 2024, doi 10.1109/ICASSP48485.2024.10447483 (UNVERIFIED on IEEE) | 1-D DOA, K sources | L = 200 snapshots, M = 8 ULA, 1-bit complex sign, SNR 0-10 dB, K in {2..5} | Transformer subspace + MUSIC + learned peak finder | MUSIC, beamformer, DA-MUSIC (unquantised), one-bit MUSIC | FIGURE-ONLY | Partly: one-bit compared with unquantised MUSIC only | Candidate baseline after re-running with our quantiser |
| 3 | Fesl, Turan, Bock, Utschick, "Channel Estimation for Quantized Systems based on Conditionally Gaussian Latent Models", arXiv 2309.04014 (venue UNVERIFIED) | Channel vector (not DOA) | P = 1 pilot, N = 64, B in {1,2,3}, SNR = 1/sigma^2 | GMM/MFA/VAE + conditional Bussgang LMMSE | BLS, Buss-Scov, EM-GM-GAMP, DNN; Buss-genie as bound | FIGURE-ONLY (text: outperforms over the whole SNR range) | Yes for channel estimation with 1-3 bits | Baseline only if a channel experiment is added |
| 4 | Nguyen, Nguyen, Swindlehurst, "Deep Learning for Estimation and Pilot Signal Design in Few-Bit Massive MIMO Systems", arXiv 2107.11958 (venue UNVERIFIED) | Channel and data detection | K users, N antennas, b in {1,2,3}, pilot 5K | FBM-CENet: unrolled gradient ascent on sigmoid-ML, pilot learned | one-bit SVM, BMMSE, BWZF, conventional DNN | Qualitative (no dB in text) | Closest conceptual match (learned start plus quantised ML) | Cite; reproduce only if time allows |
| 5 | Doshi & Andrews, "One-bit mmWave MIMO Channel Estimation using Deep Generative Networks", arXiv 2211.08635 (journal UNVERIFIED) | Channel matrix | one-bit pilots, CDL-A..E, genie scale | WGAN-GP latent optimisation (QGCE) | BG-GAMP (genie-tuned sparsity) | WGAN-GP beats BG-GAMP by 0.5 dB (CDL-B, C), 1 dB (CDL-A), 5 dB (CDL-D, E) (VERIFIED text) | Yes against GAMP, but genie-tuned | Cite for the baseline-tuning caution |
| 6 | Hu, Sun, Zhang, "Enhancing Off-Grid One-Bit DOA Estimation with Learning-Based Sparse Bayesian Approach for Non-Uniform Sparse Array", Asilomar 2024, doi 10.1109/IEEECONF60004.2024.10942609, arXiv 2412.10976 | 1-D off-grid DOA | 1 snapshot, SLA 18 or 10 elements, 1 bit | Unrolled SBL, two blocks | OGIR, CNN-DNN | FIGURE-ONLY (Figs. 4-6) | Companion of rank 1; weaker baselines | Cite as the precursor to rank 1 |
| 7 | Yeganegi, Eamaz, Esmaeilbeig, Soltanalian, "Deep Learning-Enabled One-Bit DoA Estimation", arXiv 2405.09712 (SAM 2024, doi 10.1109/sam60225.2024.10636650 per OpenAlex) | 1-D DOA | dithered one-bit, ULA, several snapshots, 2-3 targets | LISTA on dithered covariance | MUSIC | FIGURE-ONLY | Weak (MUSIC only) | Cite for dithering with LISTA |
| 8 | Fesl, Banna, Utschick, "Enhancing Channel Estimation in Quantized Systems with a Generative Prior", arXiv 2405.03542 (2024) | Channel | one-bit EM | GMM prior picks a component, then one-bit EM | "simplistic Gaussian prior" and "state-of-the-art" (unnamed) | ABSTRACT only | Unknown | Read the full text before using |
| 9 | Sheng, Chen, Yao, Wu, Chen, "Atomic Network-Based DOA Estimation Using Low-Bit ADC", Electronics 10(6):738, 2021, doi 10.3390/electronics10060738 | 1-D DOA | low-bit ADC (bit count UNVERIFIED) | ADCnet + AtomicNet | UNVERIFIED | ABSTRACT only (MDPI returned 403) | Unknown | Read the full text (MDPI blocked) |
| 10 | Zhou et al., "Generative Diffusion Models for High Dimensional Channel Estimation", arXiv 2408.10501 (2024) | Channel | low-resolution quantised measurements | diffusion posterior sampling + SURE | UNVERIFIED | ABSTRACT only; "10x latency" claim (UNVERIFIED body) | Maybe, for channel | Context only |
| 11 | Karakoca et al., "RIDNet Assisted cGAN Based Channel Estimation for One-Bit ADC mmWave MIMO Systems", arXiv 2306.08882 (2023) | Channel | one-bit | cGAN + feature-attention denoiser | UNVERIFIED | ABSTRACT only | Unknown | Context only |
| 12 | Park, Kim, Choi, "Low-Complexity Beamspace Channel Denoiser for mmWave Massive MIMO with Low-Resolution ADCs", arXiv 2605.08855 (May 2026) | Beamspace channel denoising (non-learned) | low-res ADC, composite noise model | Bernoulli-complex Gaussian hypothesis test, closed-form threshold | UNVERIFIED | ABSTRACT only | Non-learned; a beamspace classical comparator | Context; compare against it if beamspace low-res is our setting |

Not in the table (UNVERIFIED or BIB-ONLY): Azhdari and Farhang (title not found); Zhan, Xi, Chen (ICICSP 2022; not found); Li, Gao et al. (ICSPS 2025); Li, Zhao, Liu, Bi (Sensors 2022); Rahman et al. (IEEE Access 2024); Sun, Ren, Zhou (Globecom 2023); Katyal, Bharadwaja H., Murthy (EUSIPCO 2024); Li, El-Hajjar, Yang (OJCOMS 2021); Xu, Gao, Zhou, Ma, Zhang (TWC 2022).

Nearby classical references verified in this folder (each has its own explainer here): GNOMP (Zhu et al., IEEE TAES 62, pp. 3933-3951, 2026, DOI 10.1109/TAES.2025.3649136, arXiv 2307.00491); LoRD-Net (Khobahi et al., IEEE TSP 69, pp. 5651-5664, 2021, DOI 10.1109/TSP.2021.3117503, arXiv 2102.02993); Mo, Schniter, Heath (IEEE TSP 66(5), pp. 1141-1154, 2018, DOI 10.1109/TSP.2017.2781644, arXiv 1610.02735); VALSE-EP (Zhu, Zhang, Meng, China Communications 18(10), pp. 77-95, 2021, DOI 10.23919/JCC.2021.10.005, arXiv 1811.05680).

---

## Conclusion: the real state of the art we must beat

1. For one-bit single-snapshot DOA, the strongest learned comparator is Hu, Sun, Zhang (JSTSP 2025), a model-based unrolled network. It beats its own iterative parent at low SNR in the sparse-array settings and matches it on-grid at 0-15 dB. It is not compared against NOMP or quantiser-ML.
2. For multi-snapshot one-bit DOA, TransMUSIC is the strongest learned candidate, but its baselines are unquantised; a fair comparison needs re-running.
3. For channel estimation with 1-3 bits, the most informative comparators are EM-GM-GAMP and the conditional Bussgang estimator (Fesl 2023). A genie-tuned BG-GAMP sets a ceiling for generative approaches (Doshi and Andrews 2022). None of these is a 2-D AoA/AoD method.
4. No learned work in this sweep beats an exact-likelihood quantiser-ML, and no learned work compares against NOMP for one-bit joint AoA/AoD. Our own result (the network-seeded hybrid beats multi-start quantiser-ML only at 1-bit, 15 dB, +0.032*) is therefore the claim to defend, and it is what reviewers will check.
5. Reproducibility on our problem within CPU limits: Hu 2025 (small unrolled network, ULA, 1 bit) and TransMUSIC (Transformer, M = 8 in the paper) are plausible to re-implement on CPU as 1-D reductions of our problem. No source covers a 2-D AoA/AoD version, so any learned 2-D baseline would be our own adaptation and must be labelled as such. Timing is not measured here.

Files produced in this sweep: this file. Cached sources and helper scripts (`abs_dump.py`, `html2txt.py`, `mathchar.py`, `absmeta.py`) are in `C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/sweep_scripts/` and `.../sweep/`. No new PDFs were saved to C:.
