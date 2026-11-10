# When do data-driven / hybrid estimators beat model-based gridless ones? (verified literature, 2026-10-15)

Every source in the table at the end was opened with WebFetch on 2026-10-15 (Crossref/OpenAlex record plus arXiv abstract or full text). Numbers below are quoted from those papers, not recomputed. "AO" means the abstract was all I could read, and it gives no numbers. "FT" means I read the full text (arXiv HTML or ar5iv).

**Bottom line.** In the verified papers, DL or hybrid estimators beat model-based ones in four situations. (i) The baseline does not model something real, such as coherent sources for plain subspace methods or unmodelled or large array errors. (ii) Sample-covariance methods break down because there are few snapshots or the SNR is low. (iii) The array is sub-sampled or hybrid, which makes classical estimators expensive or ill-posed. (iv) The comparison is runtime against convex or Bayesian gridless solvers. Against a gridless ML-type estimator that has the same model and information, the gain shrinks or disappears at adequate SNR or snapshot count [15][12][14]. **None of the papers found benchmarks against NOMP** [29]. Most baselines are grid MUSIC, l1-SVD, OMP or spatial smoothing. This matches our result that NOMP without a network equals the CNN-seeded hybrid under the nominal per-antenna gain/phase model. That model is a classical, self-calibratable parametric mismatch [32]: an unfolded pursuit learns it online [10], and self-calibration fixes it without data.

## (a) Array imperfections, mutual coupling, position errors
- Liu, Zhang, Yu [5] (FT not available) use a multitask autoencoder as spatial filters, followed by parallel one-vs-all grid classifiers. The abstract claims only "satisfying" generalization and imperfection adaptation. Per [9] and [6], its imperfections at full severity are gain ±0.2, phase ±30°, position ±0.2d and mutual coupling 0.3e^{j60°}. These are much larger than our 8° / 3 dB. The project's own audit (citations/audit_1.md C1-011) already narrowed this claim.
- TransDOA [6] (FT): DL models show "pronounced sensitivity to array imperfections", and a model trained on ideal data "fails on imperfect data". Under severe imperfection, TransDOA falls to roughly the level of classical methods on an ideal array. Transfer learning with 20 to 1000 calibration samples recovers part of the loss (Table III: 13.66 to 11.48, against about 23 for direct training; units not stated). On an ideal array at -5 dB with T=10, the RMSE is 9.65° against MUSIC 20.39°, Unity-ESPRIT 17.09° and l1-SVD 14.33°.
- Gao, Liu, Xiong [7] (AO): a standard DNN generalizes poorly when the test-time phase errors differ from the training ones. A magnitude-only DNN is "much more robust".
- mpNet [10] (FT) unfolds matching pursuit and learns the dictionary online without supervision. It corrects gain errors with σ_g = 0.15 or 0.3 and position errors with σ_p = 0.05λ or 0.1λ. Its error converges to that of OMP with the ideal, perfectly calibrated dictionary, whereas OMP with the nominal dictionary can be worse than LS. Small errors (σ_p = 0.03λ, σ_g = 0.09) already cost more than 10 dB of SNR. The baseline is grid OMP (A = 8N); no gridless estimator is compared.
- SubspaceNet [16] (FT): with sensor-spacing perturbations and steering-vector noise, the advantage grows with the mismatch, but results appear only as figures. SDOA-Net [8] (AO) claims to beat existing methods on imperfect arrays.
- **Verdict.** The gains are real when the imperfection is large, unmodelled by the baseline (nominal-manifold MUSIC or OMP), and the same at training and test time. They degrade when training and test conditions differ. No paper compares against a gridless estimator with gain/phase self-calibration, so the matched-information comparison is missing.

## (b) Unknown number of paths / model order
- DeepFreq [12] (FT; N=50, 1 to 10 sinusoids): the counting network beats AIC, MDL and SORTE except at very high SNR. Its frequency-error FNR beats PSnet by about 4 to 7 points, and only CBLasso matches it, at high SNR.
- ERNet/ECNet [13] (AO) beat AIC/MDL with few snapshots or low SNR. CovNet [14] (FT) is "significantly" more accurate than MDL and takes 2.6 s against 505.2 s per 1000 realizations.
- The CNN of [15] (FT) estimates K jointly with the angles. At -10 dB its confidence is 90%, 74% and 71% for K = 1, 2, 3. NeuPLSE [28] (FT) detects the model order correctly more than 90% of the time for L from 1 to 5.
- **Verdict.** The baselines are information criteria, never NOMP's CFAR or residual stopping rule [29]. A data-driven stopping rule is a plausible gain area, but nobody has tested it against a matched baseline.

## (c) Coherent and closely spaced sources
- SubspaceNet [16] (FT; N=8, T=100, 10 dB, coherent sources), RMSPE: ssn+RootMUSIC 0.20° against RootMUSIC 12.48° and spatial-smoothing RootMUSIC 1.01° for M=2. For M=4 the figures are 3.88°, 23.20° and 12.15°. With T=2 snapshots, ssn+RM reaches 1.89° against 8.27° for SPS+RM. DA-MUSIC [17] (AO) also estimates the number of coherent sources.
- Papageorgiou et al. [15] (FT; N=16, 1° grid):
  - At -10 dB, T=2000, Δθ=4.7°: CNN 0.31°, root-MUSIC 0.35°, MUSIC 5.01°.
  - At 0 dB, T=200, Δθ=2.11°: CNN 0.54°, l2,1-SVD 0.54°, root-MUSIC 11.17°.
  - At high SNR or large T, the grid methods, the CNN included, hit an error floor, and only gridless root-MUSIC reaches the CRLB.
- **Verdict.** These gains are over covariance or subspace methods. A single-snapshot ML-type greedy method such as NOMP is not hurt by coherence, so this is not an argument against NOMP.

## (d) 1-bit / low-resolution ADCs and phase shifters
- The evidence is thin, and all of it is abstract-only:
  - LISTA covariance recovery from dithered one-bit samples [18]: no numbers.
  - Unrolled majorization-minimization for one-bit single-snapshot sparse arrays [19]: accepted at IEEE JSTSP, no numbers.
  - DNN dequantization [20]: one-bit DOA "exceeds" two-bit performance.
  - DL-designed compressive beam alignment under low-resolution phase shifters [21]: beats random-phase compressed sensing.
- None compares against a gridless one-bit ML or NOMP with the same information. A known phase-shifter quantization is a deterministic, model-correctable error.

## (e) Phase noise
- I found no AoA/DOA paper that compares DL against gridless methods under oscillator phase noise. Closest: Verdecia-Peña et al. [22] (AO) train a CNN for mmWave channel estimation on modelled phase noise, IQ imbalance, CFO and PA nonlinearity, and test it on hardware; the abstract gives no baseline numbers. Static array phase errors are covered in [7]. This is a gap, and therefore an opportunity.

## (f) Near-field
- SDL-LISTA [23] (AO) beats LISTA with 10 times fewer dictionary atoms. Unfolded UAMP-SBL for wideband THz near-field beam split [24] (AO) claims qualitative gains only. Both baselines are grid or compressed-sensing methods. A 16-element mmWave ULA is far-field at practical ranges, so near-field is not a framing for this paper.

## (g) Deep unfolding / unrolling
- LISTA [25] reaches the same error as coordinate descent with about 10 times less computation, and can also initialize exact solvers.
- LAMP [26] (FT; i.i.d. Gaussian, N=500, M=250): reaching about -34 dB NMSE takes AMP 25 iterations, tied LISTA 15 layers and tied LAMP about 7 layers. Untied LAMP with 10 layers comes within about 0.5 dB of the support oracle.
- LDAMP [27] (FT): 64-antenna lens mmWave array, 4-path channel. It beats SD, SCAMPI and D-AMP and converges in 5 layers. The text gives no numbers and does not say whether the angles are on or off the grid.
- NeuPLSE [28] (FT; TSP 2026) is a DFT initialization followed by learnable gradient-step frequency refinement and LMMSE amplitudes. It stays close to the CRLB, like VALSE. In 1-D sparse runs with N=128 it takes 5.15 ms against VALSE 273.18 ms, ANM-Net 10.90 ms and OMP 2.58 ms. NOMP is not a baseline.
- I found no paper that unfolds NOMP itself. NeuPLSE notes that existing off-grid unfolded methods approximate ANM and lose accuracy.

## (h) Latency and complexity
- DeepFreq takes 42 ms against 1.71 s for CBLasso per signal [12]. Barthelme's network takes 5.4 s against 127.2 s for SSR per 1000 runs [14]. NeuPLSE is about 53 times faster than VALSE but about 2 times slower than OMP [28]. [30] claims DOA "without additional complexity" (AO).
- **Verdict.** Speed wins against convex, Bayesian or grid-search solvers, not against greedy OMP or NOMP. NOMP measures about 10 ms per scene on our CPU.

## (i) Learned initialization + model-based refinement
- Barthelme and Utschick [14] (FT) refine a network estimate with gradient steps on the likelihood:
  - At SNR ≥ 10 dB, on the best 90% of runs, the hybrid "almost" reaches genie ML. The network mainly removes outliers.
  - The gain is concentrated at low snapshot counts. At N=1000, and for a fully sampled array, the gap to SSR almost vanishes.
- Deep-MLE [31] (AO; single snapshot, up to 4 targets) beats both pure ML and pure NN. NeuPLSE [28] and LISTA used as a warm start [25] follow the same pattern.
- **Verdict.** The hybrid's value is in the quality of the initialization, which means fewer outliers, not in asymptotic accuracy. When the classical initialization (NOMP's oversampled-DFT greedy step) is already good under the true model, a learned seed adds nothing, which is what we observe.

## Framing references
- **Shlezinger et al. [1]:** hybrids pay off with partial domain knowledge and limited data, because classical models are sensitive to inaccuracies. **Monga et al. [2]:** unrolling gives interpretable, efficient networks that train on datasets of reasonable size.
- **Surveys:** Ge et al. [3] review 25 DL-DOA papers (metrics, SNR, snapshots, number of sources). Grumiaux et al. [4] survey acoustic localization. **Warning:** Zheng et al., TCCN 2024 [33], is a method paper, not a survey.

## Defensible journal framings for "CNN heatmap + model-based refinement + self-calibration"
1. **Strong-baseline re-evaluation (the main framing, matching DECISIONS P-A).** Claim: under the nominal per-antenna gain/phase model, NOMP without a network and with the same information equals CNN-seeded hybrids and beats the heatmap U-Net. [15][12][14] show that DL gains vanish against matched gridless or ML estimators. Experiment: the existing E13 paired bootstrap over 51 banks, plus Pd against wall-clock time on identical hardware.
2. **Where the network earns its place: impairments outside the self-calibratable model.** Experiment: paired Pd (hybrid minus NOMP+selfcal) against impairment level for several impairments, using the [5] magnitudes:
   - mutual coupling (0.3e^{j60°});
   - element position errors (±0.2d);
   - gain errors beyond the training range (gain_g4_snr15 is the only bank in RESULTS_NEW §1.2 where the hybrid leads);
   - phase drifting within a scene (phase noise);
   - 2- to 3-bit phase-shifter quantization of W and F.

   Make a claim only where the paired interval excludes 0. Support: [6][7][10].
3. **Unknown L.** Replace the known L with a network count from the heatmap, and compare it against NOMP's CFAR/residual stopping and against MDL. Experiment: new variable-L banks (none exist yet, per RESULTS_NEW §1), scored on count accuracy and strict Pd. Support: [12][13][14][15].
4. **Fixed-latency, compute-matched estimator.** This is defensible only if the heat seed lets the decoder run fewer Newton or cyclic passes, or a fixed-depth unrolled decoder. Experiment: a Pd-against-FLOPs (or wall-clock) Pareto plot with NOMP's oversampling and refinement counts reduced. Support: [25][26][28]. Caveat: [28] is slower than OMP.
5. **Outlier and tail robustness.** Experiment: strict Pd, P(error > 1°) and worst-decile error on the sep_{1,2,3}deg and tail_snr{-25,-20} banks. Claim only where the paired interval is above 0. Support: [14] (the network mainly removes outliers) and [15] (close spacing at low SNR). In every framing, cite [32] as the origin of gain/phase self-calibration and do not claim it as new.

## References
| # | Title | Authors | Venue, year | DOI / URL | Verified | Claim supported |
|---|---|---|---|---|---|---|
| 1 | Model-Based Deep Learning | N. Shlezinger, J. Whang, Y. C. Eldar, A. G. Dimakis | Proc. IEEE 111(5):465-499, 2023 | 10.1109/JPROC.2023.3247480; arXiv 2012.08405 | yes (Crossref + arXiv abstract) | hybrids help with partial knowledge and limited data; classical models sensitive to inaccuracy |
| 2 | Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing | V. Monga, Y. Li, Y. C. Eldar | IEEE SPM 38(2):18-44, 2021 | 10.1109/MSP.2020.3016905; arXiv 1912.10557 | yes (Crossref + arXiv abstract) | unrolling: interpretable, efficient, reasonable-size training data |
| 3 | Deep Learning Approach in DOA Estimation: A Systematic Literature Review | S. Ge, K. Li, S. N. M. Rum | Mobile Inf. Syst., 2021 | 10.1155/2021/6392875 | yes (OpenAlex abstract) | DL-DOA survey (25 papers) |
| 4 | A survey of sound source localization with deep learning methods | P.-A. Grumiaux, S. Kitić, L. Girin, A. Guérin | JASA, 2022 | 10.1121/10.0011809 | yes (OpenAlex abstract) | DL localization survey (acoustic) |
| 5 | Direction-of-Arrival Estimation Based on Deep Neural Networks With Robustness to Array Imperfections | Z.-M. Liu, C. Zhang, P. S. Yu | IEEE TAP 66(12):7315-7327, 2018 | 10.1109/TAP.2018.2874430 | yes (Crossref + S2 abstract; magnitudes via [6],[9]) | DNN designed for imperfections; no numbers in abstract |
| 6 | TransDOA: Calibrating Array Imperfections via Transformer-based Transfer Learning | B. Zhou, K. Xu, Y. Quan, M. Xing | arXiv 2504.13394, 2025 | arxiv.org/abs/2504.13394 | yes (FT) | DL sensitive to imperfections; few-shot calibration; RMSE numbers |
| 7 | Robust DOA Estimation Based on DNNs in Presence of Array Phase Errors | X. Gao, A. Liu, Y. Xiong | SSPD 2022 | 10.1109/SSPD54131.2022.9896221 | yes (repository abstract) | DNN generalizes poorly under unseen phase errors |
| 8 | SDOA-Net: An Efficient DL-Based DOA Estimation Network for Imperfect Array | P. Chen, Z. Chen, L. Liu, Y. Chen, X. Wang | IEEE TIM 2024 (per arXiv) | arXiv 2203.10231 | partial (arXiv abstract; journal DOI not checked) | DL for imperfect arrays (qualitative) |
| 9 | Deep Autoencoders for DOA Estimation of Coherent Sources using Imperfect Antenna Array | A. M. Ahmed, O. Eissa, A. Sezgin | arXiv 2005.02728, 2020 | arxiv.org/abs/2005.02728 | yes (FT) | imperfection magnitudes of [5]; [5] ignores coherence |
| 10 | mpNet: Variable Depth Unfolded Neural Network for Massive MIMO Channel Estimation | T. Yassine, L. Le Magoarou | IEEE TWC 21(7):5703-5714, 2022 | 10.1109/TWC.2022.3142737 | yes (Crossref + FT) | unfolded MP learns gain/position errors online, near ideal-dictionary OMP |
| 11 | Online unsupervised deep unfolding for MIMO channel estimation | L. Le Magoarou, S. Paquelet | arXiv 2004.14615, 2020/21 | arxiv.org/abs/2004.14615 | yes (arXiv abstract) | precursor of [10] |
| 12 | Data-driven Estimation of Sinusoid Frequencies (DeepFreq) | G. Izacard, S. Mohan, C. Fernandez-Granda | NeurIPS 32, 2019 | arXiv 1906.00823; papers.nips.cc/paper/8756 | yes (FT) | counting beats AIC/MDL/SORTE; CBLasso matches at high SNR; 42 ms vs 1.71 s |
| 13 | Model-Aided Deep Neural Network for Source Number Detection | Y. Yang, F. Gao, C. Qian, G. Liao | IEEE SPL 27:91-95, 2020 | 10.1109/LSP.2019.2957673 | yes (Crossref + arXiv abstract) | beats AIC/MDL at low SNR / few snapshots |
| 14 | A Machine Learning Approach to DoA Estimation and Model Order Selection for Antenna Arrays With Subarray Sampling | A. Barthelme, W. Utschick | IEEE TSP 69:3075-3087, 2021 | 10.1109/TSP.2021.3081047 | yes (Crossref + FT) | NN + likelihood gradient steps; gains vanish at many snapshots; runtimes |
| 15 | Deep Networks for Direction-of-Arrival Estimation in Low SNR | G. K. Papageorgiou, M. Sellathurai, Y. C. Eldar | IEEE TSP 69:3714-3729, 2021 | 10.1109/TSP.2021.3089927 | yes (Crossref + FT) | low-SNR / close-spacing gains; only root-MUSIC reaches CRLB at high SNR |
| 16 | SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation | D. H. Shmuel, J. P. Merkofer, G. Revach, R. J. G. van Sloun, N. Shlezinger | IEEE TVT 74(3):4962-4976, 2025 | 10.1109/TVT.2024.3496119 | yes (Crossref + FT) | coherent / few-snapshot RMSPE tables |
| 17 | DA-MUSIC: Data-Driven DoA Estimation via Deep Augmented MUSIC Algorithm | J. P. Merkofer, G. Revach, N. Shlezinger, T. Routtenberg, R. J. G. van Sloun | IEEE TVT 73(2):2771-2785, 2024 | 10.1109/TVT.2023.3320360 | yes (OpenAlex + arXiv abstract) | coherent sources and their count |
| 18 | Deep Learning-Enabled One-Bit DoA Estimation | F. Yeganegi, A. Eamaz, T. Esmaeilbeig, M. Soltanalian | arXiv 2405.09712, 2024 | arxiv.org/abs/2405.09712 | yes (abstract) | LISTA for one-bit covariance (no numbers) |
| 19 | Model-Based Learning for DOA Estimation with One-Bit Single-Snapshot Sparse Arrays | Y. Hu, S. Sun, Y. D. Zhang | arXiv 2502.17473 (accepted IEEE JSTSP per arXiv) | arxiv.org/abs/2502.17473 | yes (abstract) | unrolled MM, one-bit off-grid (no numbers) |
| 20 | Deep Neural Network-Based Quantized Signal Reconstruction for DOA Estimation | W. Han, P. Chen, Z. Cao | arXiv 2005.01102, 2020 | arxiv.org/abs/2005.01102 | yes (abstract) | 1-bit + DNN exceeds 2-bit |
| 21 | Deep Learning-based Compressive Beam Alignment in mmWave Vehicular Systems | Y. Wang, N. J. Myers, N. González-Prelcic, R. W. Heath Jr. | arXiv 2103.00125, 2021 | arxiv.org/abs/2103.00125 | yes (abstract) | DL CS design with low-res phase shifters beats random CS |
| 22 | Enhancing mmWave Channel Estimation: ... Modeled PHY Impairments Incorporated in DL Training | R. Verdecia-Peña, R. Oliveira, J. I. Alonso | IEEE OJCOMS, 2024 | 10.1109/OJCOMS.2024.3421519 | yes (OpenAlex abstract) | CNN trained with phase noise/IQI/CFO/PA (no baseline numbers) |
| 23 | Near-Field Channel Estimation for Extremely Large-Scale Array Communications: A model-based deep learning approach | X. Zhang, Z. Wang, H. Zhang, L. Yang | arXiv 2211.15440, 2022 | arxiv.org/abs/2211.15440 | yes (abstract) | SDL-LISTA, 10x fewer atoms |
| 24 | Deep Unfolding-Based Channel Estimation for Wideband THz Near-Field Massive MIMO Systems | J. Gao, X. Cheng, G. Y. Li | arXiv 2308.13381, 2023/24 | arxiv.org/abs/2308.13381 | yes (abstract) | unfolded UAMP-SBL near-field (qualitative) |
| 25 | Learning Fast Approximations of Sparse Coding (LISTA) | K. Gregor, Y. LeCun | ICML 2010, pp. 399-406 | mlanthology.org/icml/2010/gregor2010icml-learning | yes (proceedings abstract) | ~10x less computation; can initialize exact solvers |
| 26 | AMP-Inspired Deep Networks for Sparse Linear Inverse Problems | M. Borgerding, P. Schniter, S. Rangan | IEEE TSP 65(16):4293-4308, 2017 | 10.1109/TSP.2017.2708040 | yes (Crossref + FT, partial) | LAMP 7 layers vs AMP 25 iterations |
| 27 | Deep Learning-Based Channel Estimation for Beamspace mmWave Massive MIMO Systems (LDAMP) | H. He, C.-K. Wen, S. Jin, G. Y. Li | IEEE WCL 7(5):852-855, 2018 | 10.1109/LWC.2018.2832128 | yes (Crossref + FT) | learned D-AMP beats CS baselines (figures only) |
| 28 | Low-Complexity Neural Pursuit for Line Spectral Estimation (NeuPLSE) | C. Liu, W. Jiang, X. Yuan | IEEE TSP 74:3317-3332, 2026 (per arXiv) | arXiv 2609.22731 | yes (FT; journal DOI not checked) | near-CRLB; runtime vs VALSE/OMP; no NOMP baseline |
| 29 | Newtonized Orthogonal Matching Pursuit: Frequency Estimation Over the Continuum | B. Mamandipoor, D. Ramasamy, U. Madhow | IEEE TSP 64(19):5066-5081, 2016 | 10.1109/TSP.2016.2580523 | yes (Crossref + arXiv abstract) | NOMP near-CRB, CFAR stopping |
| 30 | Deep Learning for Super-Resolution Channel Estimation and DOA Estimation Based Massive MIMO System | H. Huang, J. Yang, H. Huang, Y. Song, G. Gui | IEEE TVT 67(9):8549-8560, 2018 | 10.1109/TVT.2018.2851783 | yes (Crossref + OpenAlex abstract) | DL DOA "without additional complexity" (no numbers) |
| 31 | Deep-MLE: Fusion between a Neural Network and MLE for a Single Snapshot DOA Estimation | M. L. Lima de Oliveira, M. J. G. Bekooij | ICASSP 2022, pp. 3673-3677 | 10.1109/ICASSP43922.2022.9747692 | yes (repository abstract) | NN init + MLE correction beats either alone |
| 32 | Eigenstructure methods for direction finding with sensor gain and phase uncertainties | A. J. Weiss, B. Friedlander | Circuits Syst. Signal Process. 9(3):271-300, 1990 | 10.1007/BF01201215 | partial (Crossref metadata only) | classical gain/phase self-calibration lineage |
| 33 | Deep Learning-Based DOA Estimation | S. Zheng et al. | IEEE TCCN 10(3):819-835, 2024 | 10.1109/TCCN.2024.3360527 | yes (Crossref; type per project audit C1-010) | method paper, NOT a survey |
