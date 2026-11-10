# SignalNet (Dreifuerst and Heath, 2022): study note for the ADC project

**Citation (verified via Crossref, 2026-10-16):** R. M. Dreifuerst and R. W. Heath Jr., "SignalNet: A Low Resolution Sinusoid Decomposition and Estimation Network," *IEEE Trans. Signal Processing*, vol. 70, pp. 4454-4467, 2022, doi: 10.1109/TSP.2022.3201336. arXiv:2106.05490 (v1 10 Jun 2021; v2 13 Jan 2022, comment "Submitted to IEEE Transactions on Signal Processing"). Issue number: UNVERIFIED (Crossref gives none; the IEEE Xplore page returned empty content).

**What was read and how sure we are**
- Full text read: arXiv v2 PDF (copy: `C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/signalnet_arxiv_v2/signalnet_2106.05490v2.pdf`; extracted text `C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/signalnet_v2.txt`), pages 1-16 including references. Eq. (3) and Algorithm 1 were checked against rendered page images (`.../papers/signalnet_pages/eq3b.png`, `alg1.png`).
- The published TSP version was NOT read (IEEE Xplore page empty; NSF PAR unreachable). Differences between arXiv v2 and the TSP version are UNVERIFIED. Every statement below is about arXiv v2.
- Figure values quoted below are digitised from vector figures in an earlier run (`signalnet_digitize*.py`, logs in this folder). Their accuracy is estimated at about 0.5 dB and was not independently verified. They are marked "digitised".
- Source code: the footnote link `github.com/Ryandry1st/Carrier-Frequency-Offset` redirects to `Low-Res-Carrier-Frequency-Offset` (last push 2020-12-12). That repo is for the 2020 one-bit CFO paper (ref. [12] of SignalNet), not SignalNet. No SignalNet code was found there (`signalnet_code_check.log`). Whether SignalNet code exists elsewhere: UNVERIFIED.

---

## 1. Plain-English summary

Some radios and radars sample signals with very coarse converters (1 to 3 bits per number). The job is to find how many sinusoids (pure tones) are in the signal and what each one's strength, frequency and phase are. The authors build a neural network called SignalNet that works in two stages: first it guesses how many tones there are, then, for that count, it finds the tones one at a time. After finding each tone it rebuilds that tone, subtracts it from the received data, and looks for the next one. In their simulations the network beats standard signal-processing methods (Periodogram/FFT, AIC/MDL), especially at 3 bits. At 1 bit it often does not beat a "learning threshold", the error of a predictor that ignores its input and always outputs the average. The authors use this to argue that at 1 bit the network learns the output distribution rather than real features. We agree the threshold idea is useful, but one of their threshold formulas appears wrong, and several claims rest on digitised figures that we could not fully reconcile with the text.

---

## 2. Problem and system model

Paper's model (Sec. II, Eqs. 1-6): a sum of m complex sinusoids with amplitudes a_i, phases phi_i and normalised frequencies f_i, observed as N = 64 samples plus IID complex Gaussian noise v. The samples are normalised (Eq. 3) and quantised by a b-bit uniform quantiser Q_b applied to real and imaginary parts separately (Eqs. 4-5). The received vector is stored as an N x 2 real matrix X = [Re z, Im z] (Eq. 6).

| Paper symbol | Meaning in paper | Our project counterpart | Notes |
|---|---|---|---|
| u[n], N = 64 | noise-free 1-D sinusoid sum, 64 samples | vec(Y), Y = W^H H F + Z, 16x16 complex = 256 beamspace samples | Paper is 1-D; ours is 2-D. Our N is 256, not 64. |
| m (sinusoid count) | number of tones, M = 5 max | L = 3 paths, assumed known | Paper's detector estimates m; we give L. |
| a_i | real amplitude, U(0.1, 1.0) | abs of complex path gain g_l | Paper's amplitudes are real; ours complex. |
| phi_i | phase, U(0, 2pi) | arg(g_l) | |
| f_i | normalised frequency, cycles per sample, in (0, 0.5] | AoA / AoD spatial frequencies (u, v) for each path | Exact mapping from our steering-vector convention: UNVERIFIED in this task; check the project code. |
| v[k] | IID Gaussian noise | Z | Our noise is complex Gaussian per project convention. |
| b | bits per real/imag sample, b = 1 or 3 | b (1, 2, 3, ...) | Same meaning. |
| A (AGC level) | none; signal normalised to unit norm (Eq. 3) | per-scene peak A = max(abs Re, abs Im) over the 256 beam samples; step 2A / 2^b | Different AGC. Results must not be compared on this basis. |
| SNR | used to generate data; the formula is not given in the text | integrated SNR and SNR-loss tables in project | Paper's SNR definition: UNVERIFIED. |
| W, F | none (no beamforming) | W (receive), F (transmit) DFT codebooks | |
| H~ = W Y F^H | none | antenna-domain reconstruction | |
| Pd (paper-style, ours) | not reported; paper uses MSE and Chamfer | paper-style Pd, strict Pd, RMSE | Use our metric throughout. |

---

## 3. The method, step by step

### 3a. Data generation (Algorithm 1, p.7 of the arXiv PDF)
1. Draw m at random from [0, M-1]. As printed, this gives m = 0 to 4, while the detector outputs 1 to 5, and the sum runs i = 0..m (m+1 tones). This looks like an off-by-one error. Our re-implementation should use m in {1..5} tones. UNVERIFIED what the authors did.
2. Draw w_0 ~ U(0, 0.25); set f_1 = w_0.
3. For i = 1..m-1: draw w_i ~ |N(0, 2.5/N)| and set f_{i+1} = w_0 + i/N + w_i. This keeps tones roughly 1/N apart (folded-normal spacing).
4. Reject and redraw while max f_i > 0.5 (Nyquist limit).
5. Draw a_i ~ U(0.1, 1.0) and phi_i ~ U(0, 2pi).
6. y = sum_i a_i exp(j 2 pi f_i n + j phi_i) + v, with noise set for the desired SNR.
7. Normalise (line 13 of the algorithm): y_i <- y_i / ||y||. Eq. (3) instead writes z = Q_b(N u / ||u||^2). The two differ by a factor of about sqrt(N) = 8 in per-sample scale, which matters for b > 1 because the quantiser levels are fixed in [-1, 1]. Which one the authors used: UNVERIFIED (no code).
8. Quantise: x = Q_b(y) per real/imaginary part (uniform levels; first and last bins extend to +/- infinity, Sec. II).

### 3b. Detector (Sec. III, Table I)
- Three conv blocks (32, 64, 128 filters; pooling; BatchNorm; ReLU), dropout 0.7, FC 128, FC 64, FC 5 with softmax. Output: one-hot estimate of m in {1..5}.
- Trained with the asymmetric loss (Eq. 11): L_det(m, m_hat) = exp(m - m_hat) - 1 if m >= m_hat, else 0.5 (m - m_hat)^2. Under-estimating the count is penalised exponentially, over-estimating only quadratically. The loss is meant to make the detector over-estimate when unsure.
- Training: 20 epochs, 50,000 realisations (Sec. VII).
- Open issue: a softmax output is discrete, but Eq. (11) is written for an integer m_hat. How the authors made it differentiable is not described. UNVERIFIED.

### 3c. Estimator (Sec. III, Figs. 1-2, Table I)
One estimator per m (M separate networks per bit depth; the authors give reasons: real-time inference and the reconstruction design, Sec. III).
For k = 1..m (successive cancellation, as in SIC, ref. [42]):
- Block k has two branches of two conv layers each. Branch A has BatchNorm and outputs frequency and phase. Branch B has no BatchNorm and outputs amplitude (to keep amplitude information). Each block outputs (a_k, f_k, phi_k).
- Internal reconstruction: rebuild the time-domain tone from the estimate and subtract it from the residual, so the next block sees the residual. The text does not give the reconstruction equation. Our reading (inferred, not quoted): s_k(n) = a_k exp(j(2 pi f_k n + phi_k)), residual R <- R - [Re s_k, Im s_k]. UNVERIFIED.
- Gradients come only from the parameter error, not from the reconstruction error (Sec. III: "gradients are only updated from the parameter error").
- Loss: MSE per parameter, each weighted by 1/L_threshold (Eqs. 24-26), so amplitude, frequency and phase are on a comparable scale. Our reading of the dot-product form of Eq. (26) is an interpretation.
- Training: 100,000 samples, Adam, learning rate 1e-3, batch 32, learning-rate reduction and early stopping on a validation set (Sec. VII).
- Size: whole SignalNet about 300,000 parameters, at most 120,000 per network (Sec. VII).

### 3d. Combination
Detector -> m_hat -> run the estimator network for that m_hat -> output {a_k, f_k, phi_k}. Evaluated with Chamfer distance (Eq. 12) between estimated and true frequency sets, because the number of estimated tones can differ from the true count.

### 3e. Baselines (Sec. V)
- Detection: AIC and MDL (information-criterion model-order rules); EM-VAMP on a 2x oversampled DFT grid. The authors note AIC and MDL do not converge at 1 bit, so they show only 2- and 3-bit curves.
- Estimation: zero-padded FFT (N_0 = 2^16) peak picking, with phase from atan2 at the peak; all combined with Bussgang linearisation (Eq. 13).
- Architecture comparison (Fig. 8): MLP and CNN of similar size, trained the same way (10,000 test samples).

### 3f. Pseudo-code for re-implementation (our reading)

    # data: N=64 complex samples -> X in R^{N x 2}; training data from Algorithm 1
    # Detector
    p = softmax(CNN_det(X))            # 5 classes, m in {1..5}
    m_hat = argmax(p)                  # train with L_det (Eq. 11): HOW is UNVERIFIED
    # Estimator for a given m_hat (one network per m)
    R = X
    for k in 1..m_hat:
        (a_k, f_k, phi_k) = BlockNet_k(R)          # branch A (BN): f, phi; branch B: a
        s_k = a_k * exp(j*(2*pi*f_k*n + phi_k))     # reconstruction (inferred)
        R   = R - [Re(s_k), Im(s_k)]
    # loss: sum_k sum_params (pred - true)^2 / L_threshold[param]; Adam lr 1e-3, batch 32

---

## 4. Assumptions (explicit or implied)

- **Known noise variance?** Not needed at inference (no SNR input). It is needed to generate training data. The network must infer the SNR implicitly from the data.
- **Known number of paths?** The estimator assumes m known (one network per m). The detector estimates m separately. Our benchmark gives L = 3, which matches the paper's estimator setting.
- **Dithering?** None.
- **AGC?** Ideal gain control: unit-norm normalisation before quantisation (Eq. 3 or line 13). Clipping is not modelled. This differs from our per-scene peak AGC.
- **Snapshots?** A single snapshot of N = 64 samples per trial.
- **Grid?** Off-grid: frequencies are continuous. The spacing prior (folded normal, about 1/N) is chosen to make resolution hard but not impossible.
- **Quantiser?** Uniform. Main results use b in {1, 3}. The Fig. 6 legend (digitised) also says "2 bit". Real and imaginary parts are quantised separately.
- **Prior on parameters:** a_i ~ U(0.1, 1); phi_i ~ U(0, 2pi); w_0 ~ U(0, 0.25); spacing ~ |N(0, 2.5/N)|. Training and test use the same distribution. OOD test (Sec. VIII-G): frequencies uniform on the same range, m = 2 only.
- **Real-valued network:** complex numbers are split into two channels. Complex-valued networks are left to future work (Sec. II).
- **Data:** synthetic only; no real hardware data.

---

## 5. Reported results (with conditions)

| Quantity | Value | Where | Condition |
|---|---|---|---|
| Detection threshold (continuous) | m_hat ~ 3.69, L_thr ~ 1.67 | Sec. VIII-A, p.10 | M = 5, loss Eq. (11), constant predictor |
| Detection, 1-bit | "never able to surpass the threshold" | Sec. VIII-A, p.10 | |
| Detection, 3-bit | beats threshold for SNR > -4 dB | Sec. VIII-A, p.10 | |
| Frequency thresholds (per m = 1..5) | -18, -16.4, -16, -15.8, -15.7 dB | Sec. VIII-C, p.12 (Eq. 45) | Formula appears wrong; see Sec. 7 |
| Frequency vs Periodogram | 3-8 dB better for m in {2,3} | Sec. VIII-C, p.12 | |
| Frequency, 3-bit, vs Periodogram | 11-15 dB better for m = 2, 3 | Fig. 9b caption, p.12 | |
| Amplitude threshold | 0.0675 (= Var U(0.1,1)), -11.7 dB | Sec. VIII-D, p.13 | |
| Amplitude, architecture | "nearly 10 dB improvement" over MLP/CNN, 3-bit, m = 5 | Sec. VIII-B, p.11 | |
| Phase threshold | pi^2/3 (about 3.29, about 5.2 dB) | Sec. VIII-E, p.13 | |
| Whole system vs benchmarks | "between 2-10dB improvement" from the estimator | Conclusion, p.15 | Fig. 12 |
| Inference | "under 1 ms" on an Nvidia 3090 GPU | Sec. VII, p.9 | "limited optimization" |
| OOD (uniform frequencies, m = 2) | "almost identical" to training distribution | Fig. 13, p.14 | |
| Digitised, 3-bit, Chamfer at -20 dB | SignalNet about 24.2 vs AIC+Est about 20.2 vs Detector+FFT about 18.2 | Fig. 12b, digitised | SignalNet worse at low SNR; the paper claims universal improvement (Conclusion). Medium confidence. |
| Digitised, 1-bit amplitude, m = 3..5 | plotted curves about -16 to -22 dB; drawn threshold about -11.3 dB | Fig. 10a, digitised | Text (Sec. VIII-D) says 1-bit estimators are not learning meaningful features. Unresolved; see Sec. 7. |

The paper gives no table of final MSE or Chamfer numbers, no confidence intervals and no seeds. Each point is one run.

---

## 6. Citable claims (quotes of at most 15 words, with location)

1. **Abstract, p.1:** "our algorithm is always able to surpass the threshold for three-bit data." Use: 3-bit results beat their threshold in simulation.
2. **Abstract, p.1:** "often cannot exceed the threshold for one-bit data." Use: 1-bit learning often fails the threshold test.
3. **Sec. VI, p.8:** "It is only truly useful to judge whether a basic level of learning has occurred." Use: how the authors themselves describe the threshold, as a sanity check and not an estimator.
4. **Sec. V, p.7:** "no other algorithms designed for sinusoidal decomposition and estimation from quantized data." Use: the authors' own statement that their baselines are general-purpose, not quantisation-aware ML. Do not cite it as "no other work exists".
5. **Sec. VIII-A, p.10:** "the one-bit detection module is never able to surpass the threshold." Use: negative result at 1 bit.
6. **Sec. VIII-D, p.12:** "Even with infinite SNR very little amplitude information is recoverable from such limited data." Use: amplitude is hard at 1 bit even without noise.
7. **Sec. VII, p.9:** "our algorithm is not capable of memorizing the data." Use: training-set size argument; no proof given.
8. **Fig. 9 caption, p.12:** "Our estimator outperforms the threshold universally." Use: the paper's own claim for frequency. The threshold it uses appears too high (Sec. 7).

---

## 7. Limitations and what the paper does not show

- **Simulation only.** No hardware, no real signal, no channel, no 2-D structure. One-dimensional sinusoids with N = 64.
- **Weak classical baselines.** AIC, MDL and EM-VAMP for detection; zero-padded FFT/Periodogram for estimation. No ML estimator, no NOMP, no off-grid sparse method, no quantisation-aware ML. The authors say AIC and MDL fail at 1 bit. This paper cannot show that SignalNet beats quantisation-aware ML.
- **The frequency threshold formula appears wrong.** Algorithm 1 gives w_0 ~ U(0, 0.25), so Var(w_0) = 0.25^2/12 = 1/192 = 0.00521. Eq. (45) uses 1/64 = 0.0156, which equals E[w_0]^2 = 0.125^2. That is a probable mix-up of mean and variance (our inference). Monte Carlo over Algorithm 1 (1e6 accepted draws per m, N = 64) gives m = 1: -22.83 dB; m = 2: -20.67; m = 3: -20.21; m = 4: -20.06; m = 5: -20.01. The paper's values are -18.06, -16.44, -16.01, -15.81, -15.69 dB (source: `signalnet_thresholds.log`, A2). Consequence: "outperforms the threshold universally" (Fig. 9 caption) does not hold with the recomputed threshold. Digitised: the 1-bit network is above the corrected threshold at low SNR (for m = 2 it is about -17.1 dB at -20 dB SNR, against a threshold of -20.7 dB; `signalnet_crossings.log`).
- **The detection threshold of 1.67 cannot be reached by an integer output.** The detector outputs an integer m_hat. Our check (`signalnet_thresholds.log`, A1): E[L_det] for constant m_hat = 1..5 is 16.16, 5.54, 2.12, 1.74, 3.00. The best integer is 4, with 1.7437 (verified by hand: 8.718 / 5). A randomised mix of 3 and 4 is linear in the mixing weight, so it cannot reach 1.67 either. The paper's continuous threshold is not achievable by any input-independent integer detector, so "beats the threshold" is a comparison against a bound the detector cannot attain. Digitised: the 1-bit detection curve never falls below 1.67 (`signalnet_crossings.log`).
- **Amplitude curves at 1 bit are unexplained.** Digitised Fig. 10a: the m = 3..5 curves sit about 5-11 dB below the drawn threshold and are flat in SNR. The text says these estimators learn nothing. Our reproduction of the stated Algorithm 1 with a raw Periodogram gives amplitude MSE of -9.8 dB at 1 bit, -20 dB SNR, m = 5 (`signalnet_amp_floor.log`), while the digitised paper curve is -17.9 dB. We could not reconcile these. A value about 10 dB below any constant predictor at 1 bit would need information the text does not explain, so the figure, the loss scaling or the training target may differ from what is written. Treat the 1-bit amplitude results as UNVERIFIED.
- **Chamfer claim at low SNR.** Digitised Fig. 12b (3-bit): SignalNet is worse than AIC+Est and Detector+FFT at -20 and -15 dB, which conflicts with the paper's "universally improves" (Conclusion, p.15). Medium confidence.
- **Legend mismatch:** the Fig. 6 legend (digitised) says "2 bit Detection Mod", while the text says three-bit. Which is meant: UNVERIFIED.
- **Ambiguities:** Eq. (3) vs Algorithm 1 normalisation (factor about sqrt(N)); m in {0..4} vs {1..5}; 8,000 test samples (Sec. VII) vs 10,000 (Sec. VIII-B); the SNR definition; the reconstruction equation; how the softmax detector is trained with Eq. (11).
- **Statistics:** one run per point, no confidence intervals, no seeds reported.
- **Threshold scope:** the paper's threshold is a blind-constant MSE bound per output. It says nothing about joint or permutation-matched metrics (such as our Pd with Hungarian matching).
- **Model order:** the detector is trained with random m in {1..5}; our benchmark fixes L = 3.
- **Published version unchecked.** arXiv v2 (Jan 2022) may differ from the TSP paper.

---

## 8. Relevance to us

**What to reuse**
1. **Blind-floor / learning-threshold test.** The paper's logic: a learned estimator should beat the error of the best input-independent predictor. We already have the Pd analogue: a blind predictor gets Pd of about 0.012-0.021 on our geometry (`signalnet_blind_ours.log`: centre 0.0124, random set 0.0187 held out, best of 40 random sets 0.0208 on the selection half). Our network Pd (1-bit / 15 dB: NOMP 0.191, hybrid 0.572) is roughly 8 to 30 times that floor. So this test is easy for us to pass. Report it as a sanity floor, not as evidence of learning (see Sec. 9, Q2).
2. **Threshold-normalised loss (Eqs. 24-26)** if we train a network on angles, so that AoA and AoD errors are weighted by their prior variance.
3. **Successive cancellation with internal reconstruction** as an architecture idea, applied to the 2-D steering model: rebuild one path, subtract, estimate the next. Our hybrid already refines seeds with quantisation-aware ML, so this would be an alternative seeding network.
4. **Asymmetric detection loss (Eq. 11)** if we ever estimate L. We fix L = 3, so this is optional.
5. **OOD test idea (Sec. VIII-G):** change the prior (here, uniform spacing), keep the model, compare. For us: change the angle prior or the path spacing.

**Must it be a baseline?** Yes, if we claim a learned method beats classical ones. It is the closest published learned sinusoid estimator with quantised inputs, and reviewers will ask. The official code is not released, so we must build a SignalNet-style baseline and call it "SignalNet-style (our adaptation)".

**How to make the comparison fair**
- Same scenes, same ADC model (our peak AGC, uniform mid-rise, per real/imaginary part, as in `finetune/adc/acommon.py`), same SNR grid, same L = 3 given to every method (the classical methods already get true L).
- Same metric (paper-style Pd, strict Pd, RMSE; paired bootstrap with 2000 resamples, seed 0, per our rules). The paper reports MSE and Chamfer, not Pd.
- Same seed rules (training >= 80,000,000; validation >= 81,000,000; test banks 12000-12999; adversarial banks 13000-13999).
- Include: NOMP, consistent NOMP, quantiser-ML (13 starts + EM), the existing ADC-CNN read-out, the hybrid, a size-matched MLP and CNN (the paper's own Fig. 8 control), the blind floor, and a label-shuffled control (Sec. 9).
- Do not compare our Pd with the paper's MSE or Chamfer numbers. The AGC, dimension, N and metric all differ.

**What it implies for our network design**
- The paper's main lesson is about the test: at 1 bit, distributional learning and feature learning look the same against a constant baseline. We should test feature learning directly (label-shuffle, input-blind control), not rely on a threshold alone.
- Our 1-bit hybrid beats quantisation-aware ML at 15 dB (+0.032*, CI excludes 0). That argues against pure distributional learning. But it is one cell, one quantiser model and one seed set. At 0 dB our 1-bit hybrid is not better than ML (0.274 vs 0.275, n.s.). Keep the claim limited to 15 dB.
- A 2-D successive estimator needs one network per L; only L = 3 is needed, so one network. Using BatchNorm only on the angle branch (as the paper does for frequency and phase) is an untested design choice.

---

## 9. "If a reviewer asks..." (3 likely questions)

**Q1. "SignalNet is the closest learned method. Why is it not in your comparison?"**
A: It is a 1-D frequency estimator, and its code is not released (the footnote link points to the 2020 CFO repo). We report a SignalNet-style adaptation to our 2-D problem, with its architecture ideas and a matched training budget and ADC model. We do not claim it reproduces the paper's numbers, and its MSE/Chamfer results cannot be compared directly with our Pd.

**Q2. "Your 1-bit network is well above the blind floor. Does it pass the learning-threshold test the SignalNet paper proposes?"**
A: Yes, trivially: our Pd is roughly 8 to 30 times the blind floor. That shows only that the network beats a constant, which any method with real information does. The paper itself says the threshold is "only truly useful to judge whether a basic level of learning has occurred". More informative controls: (a) a label-shuffled network (same architecture, training labels permuted), which should fall to the floor if it learns real structure; (b) a network that sees only noise as input. We have not run these yet. Also, do not use the paper's frequency thresholds: Eq. (45) appears wrong (Sec. 7).

**Q3. "The paper says 1-bit learners only learn the output distribution. Is that true for your ADC network?"**
A: We do not know yet. The paper's argument rests on a threshold comparison and on a curve (Fig. 10a) we could not reconcile with the text. Our evidence against pure distribution learning is the 15 dB hybrid gain over quantisation-aware ML (+0.032*), but that is one cell, one quantiser model and one seed set. Until the label-shuffle and noise-input controls are run, the honest answer is "not tested".

---

## 10. Glossary (plain English)

- **Sinusoid / tone:** a pure wave with one frequency, amplitude and phase.
- **Quantiser (b bits):** rounds each real and imaginary sample to one of 2^b levels. 1 bit keeps only the sign.
- **ADC:** analogue-to-digital converter; the device that quantises.
- **AGC (automatic gain control):** rescales the signal before quantising. Paper: unit norm. Ours: peak per scene.
- **Periodogram / FFT:** a standard frequency estimator (squared magnitude of the Fourier transform). Peaks at the tone frequencies.
- **AIC, MDL:** model-order rules that choose how many tones are present by penalising complexity.
- **EM-VAMP:** an iterative sparse-recovery algorithm on a grid. Used here only for detection.
- **Bussgang decomposition:** writes a quantised signal as a scaled copy of the original plus uncorrelated distortion. Lets linear methods be applied to quantised data.
- **Learning threshold:** the error of the best constant predictor (the average of the output). A learned estimator should beat it if it learned from the input.
- **Distributional learning:** learning only the output distribution (the average) and ignoring the input.
- **Chamfer distance:** a set-matching distance between estimated and true frequency sets. Compares sets of different size.
- **Successive cancellation / internal reconstruction:** estimate one tone, rebuild it, subtract it from the input, estimate the next.
- **Out-of-distribution (OOD) test:** evaluate on a different data prior (here, uniform frequencies) to see whether the network generalises.
- **Blind floor (ours):** the Pd of a predictor that outputs fixed angles regardless of data. Our analogue of the learning threshold.
- **Label-shuffle control (ours):** train the same network with training labels randomly permuted. If it still beats the floor, it is not learning the input-output link.

---

## 11. Our re-checks (what was computed and where)

All in `D:/ai_ml_project/paper/evidence/adc_program/papers/`. Scripts and logs come from the earlier run and were not re-run in this session, except the web checks.

| Check | Result | File |
|---|---|---|
| Detection threshold (continuous, Lambert W, Eq. 38) | m_hat* = 3.6900, threshold 1.6707 (reproduces the paper) | `signalnet_thresholds.log` A1 |
| Best integer constant for Eq. (11) | m_hat = 4 gives 1.7437; no integer or randomised constant reaches 1.67 | same, A1 |
| Frequency threshold, Monte Carlo of Algorithm 1 | -22.83, -20.67, -20.21, -20.06, -20.01 dB (m = 1..5) vs paper -18.06..-15.69 | `signalnet_thresholds.log` A2 |
| Amplitude threshold | Var U(0.1, 1) = 0.0675 (-11.71 dB) | `signalnet_thresholds.log` A3 |
| Phase threshold | Var U(0, 2pi) = pi^2/3 = 3.2899 (5.17 dB) | same |
| Digitised figures (approximate) | frequency, amplitude, detection, Chamfer | `signalnet_digitize.log`, `signalnet_digitize_chamfer.log` |
| Crossings with paper and recomputed thresholds | frequency, amplitude, phase, detection, MSE | `signalnet_crossings.log` |
| Blind (input-independent) Pd on our geometry | 0.0124 (centre), 0.0187 (random set), 0.0192 (spread triangle); analytic RMSE on hits 0.577 deg | `signalnet_blind_ours.log` |
| Amplitude floor vs Periodogram, our reproduction | blind constant MSE 0.067; raw Periodogram m = 5, 1 bit, SNR -20 dB: -9.79 dB (paper digitised -17.85 dB) | `signalnet_amp_floor.log` |
| Web checks, this session | Crossref metadata; arXiv history; IEEE page empty; NSF PAR unreachable | `signalnet_web_checks.log` |

**Caveat on the blind Pd floor (ours):** it is estimated on 2000-scene selection halves, not the maximum over all constant sets. The true floor (the paper's "min over constants") may be a little higher. More constant sets must be searched before quoting a single number.

---

## 12. Plan for a SignalNet-style baseline on our 2-D problem (not yet implemented)

1. Input: Re and Im of the quantised beamspace Y (16 x 16) as two 2-D channels, quantised with our ADC model. Same input as the hybrid.
2. Detector: skip (L = 3 is given). Optional later.
3. Estimator: three successive blocks (k = 1..3). Each block: a conv branch with BatchNorm (predicts AoA and AoD) and a branch without BatchNorm (predicts complex-gain amplitude). Reconstruct each path with the steering matrices of our codebooks, subtract in the beamspace domain, pass the residual on.
4. Loss: sum of threshold-normalised parameter MSE (Eqs. 24-26), with thresholds from the prior (angle variance from U[0, pi]; see Sec. 8). Match paths across estimates by Hungarian assignment, as in our metric.
5. Training: the paper's budget (Adam, lr 1e-3, batch 32, 100,000 samples) as a starting point; training seeds >= 80,000,000 per our rules.
6. Controls: size-matched MLP and CNN; label-shuffled network; noise-input network; NOMP; quantiser-ML; the existing hybrid.
7. Metric and test banks as in Sec. 8.

Expected problems to check before running: the paper does not define the 2-D reconstruction step; the successive scheme may be poor when two paths are close (the paper's own point, Sec. VIII-D); and the threshold-normalised loss needs the prior variance of our angles, which we must compute from our generator.

---

## 13. Bibliography entry (for citation)

R. M. Dreifuerst and R. W. Heath Jr., "SignalNet: A Low Resolution Sinusoid Decomposition and Estimation Network," *IEEE Trans. Signal Process.*, vol. 70, pp. 4454-4467, 2022, doi: 10.1109/TSP.2022.3201336. arXiv:2106.05490.
(Issue number UNVERIFIED. Title, authors, volume, pages, year and DOI from Crossref.)
