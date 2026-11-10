# LoRD-Net explained: unfolded one-bit ML detection, and what it means for our quantised-beamspace angle estimator

**Paper (bibliographic data checked against Crossref and the arXiv abstract page, 2026-10-16):**
S. Khobahi, N. Shlezinger, M. Soltanalian, Y. C. Eldar, "LoRD-Net: Unfolded Deep Detection Network With Low-Resolution Receivers,"
*IEEE Transactions on Signal Processing*, vol. 69, pp. 5651-5664, 2021. DOI 10.1109/TSP.2021.3117503. arXiv:2102.02993 [eess.SP]
(v1, 5 Feb 2021; arXiv lists **only v1**). A conference version appeared at ICASSP 2019 (their ref. [1], which is also where they say eq. (3) is proved).

**What was read.** The full arXiv v1 text (13 pages: abstract, Secs. I-V, Algorithm 1, Figs. 1-10, 60 references), extracted from
`C:/Users/Milon/iabr_deep_research_cache/adc_program/papers/lordnet_pdf/lordnet_arxiv_2102.02993v1.pdf` (text: `lordnet_v1.txt`, same folder).
**Page and section numbers below are those of arXiv v1.** I could **not** read the published IEEE full text (14 pages): the Weizmann copy
reset the connection, the IEEE author-accepted-manuscript link returned HTTP 418, NSF-PAR refused the connection, and the DOI (checked 2026-10-16) resolves to IEEE Xplore document 9557819, whose title matches but whose body is empty both to the fetch tool and to the browser. **Any difference between v1 and the journal version is UNVERIFIED.** Footnote 1 of the paper gives github.com/skhobahi/LoRD-Net as the source code. Checked 2026-10-16: the repository holds only README.md and LICENSE; the README's code section is a "[TBW]" placeholder, and its license text (GPLv2) disagrees with the sidebar (MIT). **So there is no usable public code.**

**Labels used in this note.** *UNVERIFIED* = could not be checked in the paper. *INFERENCE* = my reading, not stated by the authors.
*OUR CHECK* = a number from a script I ran, saved in a log:

| Our check | Script | Log / output |
|---|---|---|
| Exact read-out of every plotted point of Figs. 4-10 (the arXiv figures are vector graphics, so each data point is a polyline vertex; it is converted to data units via the major tick marks; tick-fit residual at most 3e-5 axis units) | `papers/lordnet_check/lordnet_digitize.py` | `lordnet_digitize.log`, `lordnet_digitized.json` |
| Small re-implementation of LoRD-Net in the paper's own toy setting (BPSK, 128x16, one-bit, 6 channels per SNR, 4 and 8 dB, B = 512 pilots, 2048 test vectors) | `papers/lordnet_check/lordnet_mini.py` | `lordnet_mini.log`, `lordnet_mini.json` |
| The paper's baseline nML (Choi, Mo, Heath 2016, Algorithm 1) re-implemented on the same channels with the true channel | `papers/lordnet_check/lordnet_nml_check.py` | `lordnet_nml_check.log`, `lordnet_nml_check.json`, `lordnet_nml_check_run1_partial.log`; NaN/overflow re-check of the selected runs: `papers/lordnet_check/lordnet_nml_nan_check.py` | `lordnet_nml_nan_check.log`, `lordnet_nml_nan_check.json` |

(Paths under `D:/ai_ml_project/paper/evidence/adc_program/`. The nML reference was read in full text: J. Choi, J. Mo, R. W. Heath,
"Near maximum-likelihood detector and channel estimator for uplink multiuser massive MIMO systems with one-bit ADCs," IEEE Trans. Commun.,
vol. 64, no. 5, pp. 2005-2018, 2016, as cited by LoRD-Net ref. [30]; arXiv:1507.04452; copy in `.../papers/choi2016/`.)

---

## 1. One-paragraph plain-English summary

A receiver with very cheap one-bit converters only learns whether each antenna's signal was positive or negative. LoRD-Net is a small
neural network that recovers the users' transmitted bits from these signs without knowing the wireless channel. Its layers are copies of
the steps of a textbook method: repeatedly nudge a guess so that the observed signs become more plausible, then round the guess to +1 or -1.
Only two things are learned: a stand-in channel used inside the plausibility score, and one step size per layer and per user. They are
learned from a few hundred to two thousand known pilot symbols, in two stages. First the stand-in channel is learned, with the step sizes held fixed.
Then the step sizes are learned, with the stand-in channel held fixed. The authors' key observation is that the best stand-in channel is not the true channel,
because the rounding step changes which model is best to optimise. In their simulations (one bit, 128 antennas, 16 users, BPSK), the network has 2,528
numbers to learn. It beats a data-hungry black-box detector (DeepSIC) and an iterative detector (nML) that is given the true channel.
Our own re-implementation adds a caution. The learned network matched the paper's own numbers, but a plain, non-learned gradient method that knows
the true channel was as good or better. So the large reported win over nML says more about that particular baseline than about the learned
stand-in channel (Sections 5.3, 5.4 and 7).

---

## 2. The problem and the system model

**Their model (Sec. II-A, eqs. (1)-(2), p. 3).** m one-bit ADCs observe y = Hx + n, with n ~ N(0, C) and C diagonal. The ADC output is
r = sign(y - b), with a fixed, known threshold vector b. x is drawn from a finite alphabet M^n (BPSK {-1, +1} in all experiments, Sec. IV-A).
The task is to recover x from **one** vector r (one channel use). The setting is **blind**: Theta = {H, C} is not given to the receiver. Instead
the receiver has B labelled pilot pairs {x_p, r_p}. The model is real-valued. The authors say a complex model can be handled by conventional transformations
(p. 3) but do not demonstrate it. Experiments use b = 0 and C = I known (only H unknown), except Fig. 6, where a diagonal C is learned too.

**Notation clashes (important when presenting).** Their **b** is the threshold vector, not the number of bits. Their **L** is the number of
network layers (30), not the number of paths. Their **n** is the number of users (columns of H), and the same letter is also used for the noise vector.

| Our symbol (project) | Meaning in our project | Closest LoRD-Net counterpart | Comment |
|---|---|---|---|
| **Y** | 16x16 complex beamspace measurement W^H H F + Z, then quantised: Q_b(Y) | **y** (unquantised m-vector); **r** = sign(y - b) is the quantised output | We quantise Re and Im of 256 beam samples (512 real ADC outputs); they have m = 128 or 64 real one-bit outputs |
| **W, F** | known 16x16 DFT combiner / precoder codebooks | none: no analog processing; recovery is "only in the digital domain" (Sec. I) | Our W, F are known and become part of the known measurement operator |
| **H** | 16x16 antenna-domain channel, whose angles we estimate | none directly. Their **H** (m x n) is the *unknown linear map from the unknowns to the measurements*. Our analogue is the known operator A(u, v), whose columns vec(W^H a_r(u_l) a_t(v_l)^H F) map path gains to beam samples | They learn their H *per channel realisation* as a nuisance; our operator is known up to the angles we want |
| **u, v** (AoA / AoD spatial frequencies; our array uses a_r[n] = exp(-j pi u n)/4, u = cos psi) | parameters of interest, nonlinear | none. Their unknown of interest x enters **linearly** | Biggest structural difference: their relaxed problem is convex; ours is non-convex in (u, v) |
| path gains alpha_l | linear unknowns (nuisance) | **x** (symbols, linear) | Their x lies in a finite alphabet and is rounded at the end |
| **L** = 3 paths (known in the benchmark) | model order | **n** = number of users (16 or 10), known | Both assume the model order is known |
| **b** = bits per real ADC output (1 or 2 here) | quantiser resolution | always **1 bit**. Multi-bit is called a simple extension but left to future work (Sec. III-D, p. 7) | Not tested above 1 bit |
| **A** = per-scene peak AGC full scale; step 2A/2^b | gain control before the quantiser | none. Thresholds fixed and known (b = 0); threshold optimisation left to future work (Sec. III-D) | At 1 bit with zero threshold our model equals theirs: sign(Re), sign(Im) do not depend on A |
| **SNR** (-10..25 dB; noise variance 10^(-SNR/10)/2 per real component; unit-norm paths) | per-scene noise level, known to our quantiser ML | eq. (17): SNR = E\|\|Hx\|\|^2 / E\|\|n\|\|^2; 0-10 dB; C = I known | How SNR was varied with C = I fixed is **not stated** (UNVERIFIED). With H ~ N(0, I) and C = I it would be fixed at n. Our re-implementation scales H |
| noise variance sigma^2 | known to our ML | **C** (diagonal); known except in Fig. 6 | Learning C is LoRD-Net's version of "unknown SNR" |
| training scenes (seeds 80,000,000+); 1000 test scenes per cell | | **B** pilots (32-2048) for one channel; 2048 test vectors | Their training is per channel and per SNR (the "per channel" part is INFERENCE, see Sec. 4) |

---

## 3. The method, step by step

**Step 1 - likelihood of the signs (eq. (3), p. 3).** For one-bit output r_i, the probability of the observed sign is the Gaussian tail
Q((r_i/sigma_i)(b_i - h_i^T x)), i.e. the chance that the noise put y_i on the observed side. Summing the logs over the m ADCs gives the
log-likelihood. *Plain words: score a guess x by how plausible all the observed signs are under it.*
Printing issue: eq. (3) carries a leading minus sign, which makes it the **negative** log-likelihood. Yet (4)-(5) maximise it, and the text calls (6) the
gradient of the negative log-likelihood. OUR CHECK: eq. (6) equals the automatic-differentiation gradient of eq. (3) **as printed** (max relative
difference 7.3e-16, `lordnet_mini.log` line 1). So the update (10) is descent on the negative log-likelihood, i.e. ascent on the likelihood, which is
consistent. Only the labels of (3)-(5) are off by a sign.

**Step 2 - relax the alphabet (eqs. (4)-(5), pp. 3-4).** Exact ML searches all |M|^n symbol vectors, which is exponential in n. Letting x be any real vector gives
the relaxed ML (5). Because Q is log-concave, the relaxed problem is **convex** (p. 4).
*Plain words: replace "pick the best of 2^16 candidates" by "slide x smoothly to the top of one hill".*

**Step 3 - why the true model may not be the best model after relaxation (end of Sec. II-B, p. 4; Fig. 2).** The relaxed maximiser need not be
close to the discrete ML answer, especially at high noise. The authors argue that after relaxing, the true (H, C) are no longer guaranteed to be
the best parameters to put into the relaxed objective. A "competitive objective" with different parameters can have its continuous maximiser closer to the
true symbols (Fig. 2 sketches this for n = 1).

**Step 4 - gradient (eq. (6), p. 4).** grad = H^T R~ eta(R~ (b - Hx)), with eta = Q'/Q element-wise (minus the inverse Mills ratio) and
R~ = Diag(r) C^(-1/2). *Plain words: every ADC whose observed sign looks "surprising" under the current guess pushes x, along its row of H,
towards making that sign likely.*

**Step 5 - unfold the iterations into layers (eqs. (9)-(12), p. 5; Fig. 3).** Layer i computes x_{i+1} = x_i - G_i z_i, where z_i is the gradient (6)
evaluated with the network's own Theta = {H, C}. G_i is a positive-definite preconditioner (**diagonal** in all experiments, Sec. IV-A, p. 7),
parametrised as G_i = W_i W_i^T so that it stays positive semi-definite (p. 6). Theta is **shared by all layers**; only G_i is layer-specific.
With G_i = delta I the network is exactly L steps of gradient descent with step delta (p. 5).

**Step 6 - round only at the end (eq. (13), p. 5).** The output is projected onto the alphabet, x_hat = P_M(x_L) (sign for BPSK). Projecting after
every layer (unfolded projected gradient) caused vanishing gradients and clearly worse detection, so the projection is applied once, at inference only.

**Step 7 - Training Stage 1: learn the competitive objective (eqs. (8), (14)-(15), pp. 4-5).** Fix G_i = delta I (the "basic optimisation
policy") and use a large L, so the network approximates the relaxed ML for the current Theta. Then choose Theta to minimise the mean squared
error between the network output and the pilot symbols: min over Theta of (1/B) sum_b || G^L_{delta I}(x0; Theta, r_p^b) - x_p^b ||^2.
*Plain words: tune the model so that "run gradient ascent on this model's likelihood" lands on the right symbols. The goal is not a model equal to the true channel.*

**Step 8 - Training Stage 2: learn the step sizes (eq. (16), p. 6; Algorithm 1).** Freeze Theta* and train G_i = W_i W_i^T with the same MSE loss.
Stage 2 may use fewer layers than Stage 1 (Sec. III-C), but the experiments use L = 30 in both stages (Sec. IV-A).
The paper also mentions one-stage joint training and alternating training as options (Sec. III-D) and compares all three in Fig. 9.

**Hyper-parameters actually used (Sec. IV-A, p. 7).** L = 30; delta = 0.01. Stage 1: Adam, constant learning rate 1e-3, 400 epochs. Stage 2: Adam,
learning rate 1e-4, mini-batch 512, 400 epochs. Diagonal G_i. Only H learned unless stated (C = I known). Every data-driven receiver is trained
separately for each SNR (Sec. IV-B, p. 8). B in {1024, 2048} for Figs. 4-7; B = 512 for the training study (Fig. 9, 8 dB); 2048 test vectors (Figs. 7-9).
**Not stated anywhere in v1 (UNVERIFIED):** the initial point x0; the initialisation of H in Stage 1; the Stage-1 mini-batch size; how many
channel realisations or trials are averaged; how the SNR is varied with C = I fixed; that the test symbols use the same H as the pilots
(INFERENCE: they must, since H is a trained parameter of the network).

**Pseudo-code (enough to re-implement; items marked * are our choices because the paper does not state them):**

```
inputs : pilots {(x_p^b, r_p^b)}_{b=1..B}, thresholds b (=0), alphabet M, L = 30, delta = 0.01
eta(z)            = -phi(z) / Q(z)                       # Q'/Q; compute through log Q for stability
grad(x; H, C, r)  = H^T Rt eta(Rt (b - H x)),  Rt = Diag(r) C^(-1/2)
net(r; H, C, {G}) : x <- x0 (*x0 = 0*);  for i = 0..L-1:  x <- x - G_i grad(x; H, C, r);  return x
# Stage 1: competitive objective
H <- H_init (*pilot correlation R_p^T X_p / B, rescaled; a random init failed in our check*);  G_i <- delta I
for 400 epochs (Adam, lr 1e-3, *mini-batch 512*):
    loss = mean_b || net(r_p^b; H, C, {delta I}) - x_p^b ||^2;   update H (and diagonal C if unknown)
# Stage 2: learned preconditioners
freeze H*, C*;   W_i <- sqrt(delta) * ones(n) (diagonal),  G_i = Diag(W_i)^2
for 400 epochs (Adam, lr 1e-4, mini-batch 512):
    loss = mean_b || net(r_p^b; H*, C*, {G_i}) - x_p^b ||^2;   update {W_i}
# inference on new data from the SAME channel
x_hat = P_M( net(r; H*, C*, {G_i*}) )                    # sign(.) for BPSK
```

**Parameter count (Sec. IV-B, p. 8; Sec. IV-C, p. 9).** |H| + |phi| = mn + nL = n(L + m) = 16 x (30 + 128) = **2528** for 128x16 with diagonal G_i.
Learning a diagonal C adds m (2656). The over-parametrised unfolded Benchmark 1 has 2Lnm = 122,880, the rank-1 Benchmark 2 has 2Lr(m + n) = 8640, and
DeepSIC has more than 8 x 10^5 (p. 8). (For 64x10 the formula gives 10 x 94 = 940; the paper does not print this number.)

**What made it work, according to the paper (and the evidence it gives):**
1. *The surrogate model carries most of the gain, the step sizes much less.* In Fig. 9 (8 dB, B = 512) most of the BER drop happens in Stage 1 (p. 11).
   Read-out (OUR CHECK of the plot): the two-stage test BER is 1.17e-3 at the end of Stage 1 (x = 80, i.e. epoch 400) and 7.63e-4 at the end of Stage 2.
2. *Stage 2 mainly accelerates.* Fig. 10(a) (Rayleigh, 8 dB): after Stage 1, about 30 layers are needed (BER 3.43e-2 at index 10, 7.32e-4 at index 29).
   After Stage 2 the network reaches 7.32e-4 at index 10 and 6.71e-4 at index 29; the authors say about 10 layers suffice (p. 11).
3. *Few, shared parameters plus staged training regularise.* Fig. 9: two-stage training has the smallest generalisation gap (final train/test
   6.41e-4 / 7.63e-4) vs one-stage joint training (7.32e-4 / 1.95e-3) and alternating training (4.88e-4 / 1.13e-3). The authors attribute this to an
   implicit limit on model capacity (p. 11).
4. *Faithfulness to the model beats over-parametrised unfolding.* Fig. 7: the same recursion with free per-layer matrices A_i, B_i (eq. (18)) is
   worse at equal data, despite about 50x more parameters.
5. *Round only at the end* (Step 6) and *keep G_i PSD* (Step 5).

---

## 4. Assumptions (what the method needs)

| Question | LoRD-Net | Where |
|---|---|---|
| Known noise variance? | Yes in the main results (C = I known). Fig. 6 also learns a diagonal C and reports a small gain | Sec. IV-A p. 7; Sec. IV-B p. 8 |
| Known channel? | No (blind), but the same channel must hold for the B pilots and for the data the network is applied to (INFERENCE: H is a trained parameter). Periodic pilots and re-training are suggested for block fading | Sec. III-D p. 6; Sec. IV-B p. 8 |
| Known number of unknowns? | Yes, n users, fixed (analogue of our known L) | Sec. II-A |
| Dithering? | None. Fixed known thresholds, b = 0 in experiments; optimising b is future work | Sec. II-A p. 3; Sec. III-D p. 7; Sec. IV-A p. 7 |
| AGC / gain control? | Not needed and not discussed: one-bit with zero threshold is scale-invariant | (INFERENCE) |
| Bits? | One bit only; multi-bit described as a simple change of eq. (3) but not shown | Sec. III-D p. 7 |
| Snapshots? | One received vector per detection; B pilot vectors of the same channel for training | Sec. II-A; Sec. IV |
| Grid? | No grid: continuous relaxation, then projection onto the finite alphabet | eqs. (5), (7), (13) |
| Signal model | Real-valued, linear, i.i.d. symbols; BPSK in all experiments | Sec. II-A; Sec. IV-A |
| Training data | Labelled pilots per SNR (B = 32-2048). Channels: Rayleigh (H ~ N(0, I)) or COST-2100 (2.6 GHz indoor, closely spaced users, ULA at the BS) | Sec. IV-A p. 7 |

---

## 5. Reported results (exact numbers)

The paper prints almost no numbers in its text and has **no tables**; the results are in figures. The numbers below are the **plotted values,
read exactly from the vector graphics** of arXiv v1 (`lordnet_digitize.log`, 3-4 significant digits). "Index" in Fig. 10 is the x-axis
position 0-29. Index 0 has BER of about 0.5, consistent with an uninformative x0 (INFERENCE).

**Text claims (p. 8) checked against the read-out.** Fig. 4(a): LoRD-Net (B = 2048) reaches BER 1e-2 at about 3 dB, while DeepSIC needs about 5 dB.
The read-out gives 2.85 dB vs 5.27 dB, consistent. For COST-2100 at BER 3e-2 the paper claims a similar gap; the read-out gives 1.55 dB vs 3.28 dB. Also in
the text: the parameter counts (Section 3), 700 nML iterations vs 30 LoRD-Net layers (p. 8), and about 512 training samples being enough (p. 10).

### 5.1 BER vs SNR, Figs. 4-5 (trained per SNR; nML is given the true channel; DeepSIC uses B = 2048)

| Fig. / setting | Method | 0 dB | 2 dB | 4 dB | 6 dB | 8 dB | 10 dB |
|---|---|---|---|---|---|---|---|
| 4(a) 128x16 Rayleigh | nML, perfect CSI | 8.22e-2 | 5.55e-2 | 3.39e-2 | 1.93e-2 | 1.04e-2 | 4.68e-3 |
| | DeepSIC, B=2048 | 4.93e-2 | 2.93e-2 | 1.52e-2 | 7.89e-3 | 4.30e-3 | 2.48e-3 |
| | LoRD-Net, B=1024 | 3.46e-2 | 1.57e-2 | 6.24e-3 | 2.22e-3 | 7.04e-4 | 2.02e-4 |
| | LoRD-Net, B=2048 | 3.31e-2 | 1.50e-2 | 5.80e-3 | 1.97e-3 | 6.07e-4 | 1.93e-4 |
| 4(b) 128x16 COST-2100 | nML, perfect CSI | 9.10e-2 | 7.67e-2 | 6.46e-2 | 5.50e-2 | 4.68e-2 | 3.93e-2 |
| | DeepSIC, B=2048 | 6.48e-2 | 4.20e-2 | 2.49e-2 | 1.48e-2 | 9.27e-3 | 5.98e-3 |
| | LoRD-Net, B=1024 | 5.04e-2 | 2.79e-2 | 1.44e-2 | 6.98e-3 | 3.34e-3 | 1.78e-3 |
| | LoRD-Net, B=2048 | 4.79e-2 | 2.62e-2 | 1.31e-2 | 6.43e-3 | 3.20e-3 | 1.58e-3 |
| 5(a) 64x10 Rayleigh | nML, perfect CSI | 9.77e-2 | 6.90e-2 | 4.48e-2 | 2.74e-2 | 1.59e-2 | 8.12e-3 |
| | DeepSIC, B=2048 | 6.27e-2 | 3.75e-2 | 1.94e-2 | 9.82e-3 | 5.16e-3 | 2.73e-3 |
| | LoRD-Net, B=1024 | 4.77e-2 | 2.47e-2 | 1.14e-2 | 4.82e-3 | 1.79e-3 | 7.23e-4 |
| | LoRD-Net, B=2048 | 4.60e-2 | 2.35e-2 | 1.08e-2 | 4.57e-3 | 1.76e-3 | 6.93e-4 |
| 5(b) 64x10 COST-2100 | nML, perfect CSI | 9.59e-2 | 8.00e-2 | 6.72e-2 | 5.87e-2 | 5.33e-2 | 4.95e-2 |
| | DeepSIC, B=2048 | 7.95e-2 | 5.23e-2 | 3.15e-2 | 1.89e-2 | 1.14e-2 | 6.74e-3 |
| | LoRD-Net, B=1024 | 6.68e-2 | 4.09e-2 | 2.35e-2 | 1.31e-2 | 7.37e-3 | 4.44e-3 |
| | LoRD-Net, B=2048 | 6.34e-2 | 3.83e-2 | 2.19e-2 | 1.22e-2 | 6.93e-3 | 3.93e-3 |

The ratio nML / LoRD-Net (B = 2048) grows with SNR: from 2.5x at 0 dB to 24x at 10 dB in Fig. 4(a), and from 1.5x to 12.6x in Fig. 5(b). DeepSIC /
LoRD-Net ranges from 1.3x to 12.9x. Doubling B from 1024 to 2048 changes LoRD-Net's BER by only 1-14 %.

### 5.2 Other figures

* **Fig. 6 (B = 1024, learning C too).** Rayleigh, Theta = {H, C}: 3.44e-2, 1.57e-2, 6.07e-3, 2.30e-3, 6.59e-4, 1.71e-4 at 0-10 dB. Theta = {H} is the same curve
  as Fig. 4(a), B = 1024. COST-2100, Theta = {H, C}: 4.94e-2, 2.71e-2, 1.33e-2, 6.29e-3, 2.82e-3, 1.26e-3 (vs 5.04e-2 ... 1.78e-3 for {H}). The gain is small and
  not uniform: at 6 dB (Rayleigh), {H, C} is slightly worse (2.30e-3 vs 2.22e-3).
* **Fig. 7 (128x16 Rayleigh; all L = 30; test 2048).** Benchmark 1 (free A_i, B_i per layer; 122,880 parameters), B = 1024: 4.31e-2, 2.13e-2, 1.00e-2,
  4.59e-3, 2.49e-3, 1.21e-3. Benchmark 1, B = 2048: 3.70e-2, 1.77e-2, 7.97e-3, 3.74e-3, 1.76e-3, 9.34e-4. Benchmark 2 (rank 1; 8640 parameters), B = 1024:
  4.12e-2, 2.04e-2, 9.23e-3, 4.19e-3, 2.15e-3, 1.02e-3. LoRD-Net (2528 parameters), B = 1024: 3.46e-2, 1.57e-2, 6.24e-3, 2.22e-3, 7.04e-4, 2.02e-4.
* **Fig. 8 (BER vs training size B; 128x16 Rayleigh; 2048 test vectors).**

| SNR \ B | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|
| 0 dB | 0.229 | 0.119 | 0.0700 | 0.0498 | 0.0416 | 0.0370 | 0.0351 |
| 2 dB | 0.173 | 0.0699 | 0.0373 | 0.0256 | 0.0193 | 0.0175 | 0.0165 |
| 4 dB | 0.123 | 0.0387 | 0.0193 | 0.0114 | 8.73e-3 | 7.48e-3 | 6.99e-3 |
| 6 dB | 0.0806 | 0.0206 | 8.64e-3 | 5.37e-3 | 3.97e-3 | 3.05e-3 | 2.81e-3 |
| 8 dB | 0.0628 | 0.0127 | 4.09e-3 | 2.63e-3 | 1.16e-3 | 1.04e-3 | 8.85e-4 |
| 10 dB | 0.0480 | 7.54e-3 | 1.86e-3 | 9.77e-4 | 3.66e-4 | 3.36e-4 | 3.05e-4 |

* **Fig. 9 (8 dB, B = 512, test 2048; x-axis = epochs/5; Stage-1 to Stage-2 switch at x = 80).** Final test BER: two-stage 7.63e-4, alternating 1.13e-3,
  one-stage 1.95e-3. Final train BER: 6.41e-4, 4.88e-4, 7.32e-4.
* **Fig. 10 (8 dB), BER at index 0, 5, 10, 15, 20, 29.**
  (a) Rayleigh: nML (perfect CSI) 0.50, 7.18e-2, 2.16e-2, 1.55e-2, 1.43e-2, 1.43e-2. LoRD-Net with the true H and learned G_i ("perfect CSI"): 0.498, 1.71e-3,
  1.53e-3, 1.46e-3, 1.46e-3, 1.50e-3. After Stage 1: 0.498, 0.203, 3.43e-2, 6.68e-3, 2.22e-3, 7.32e-4. After Stage 2: 0.498, 2.60e-3, 7.32e-4, 6.61e-4, 6.71e-4, 6.71e-4.
  (b) COST-2100: nML 0.50, 0.103, 6.57e-2, 5.88e-2, 5.60e-2, 5.19e-2. Perfect-CSI LoRD-Net: 0.502, 1.63e-2, 1.28e-2, 1.25e-2, 1.25e-2, 1.24e-2. Stage 1: 0.498, 0.164,
  3.99e-2, 1.25e-2, 6.05e-3, 3.51e-3. Stage 2: 0.498, 9.54e-3, 3.91e-3, 3.33e-3, 2.95e-3, 2.93e-3.
  **What the figure itself implies:** with the *true* channel, learning only the step sizes already takes BER from nML's 1.43e-2 to 1.50e-3 (about 10x).
  The learned surrogate channel adds a further factor of about 2.2 (to 6.71e-4). So most of the "beats nML" gap does not require a surrogate model.

**Statistical resolution (OUR CHECK, last blocks of `lordnet_digitize.log`).** Every plotted value in Fig. 8 is an exact integer multiple of
1/(2048 x 16) = 1/32,768 (largest deviation 0.01 of an error). So Fig. 8 is one test set of 32,768 bits; for example, 10 bit errors at 10 dB with B = 2048.
Figs. 9-10 are multiples of one third of that unit (consistent with an average over 3 runs; INFERENCE). Figs. 4-7 are not integer multiples, consistent with
averaging over an unstated number of runs. Values for the same nominal setting differ between figures: at 8 dB with B = 512 they are 1.16e-3 (Fig. 8), 7.63e-4 (Fig. 9)
and 6.71e-4 (Fig. 10(a)), a spread of 1.7x. That spread is larger than several reported differences (B = 1024 vs 2048; Theta = {H} vs {H, C}). There are no error bars or confidence intervals.

### 5.3 OUR CHECK 1: re-implementing LoRD-Net in the paper's toy setting (6 channels per SNR; each 2048 test vectors x 16 users = 32,768 bits)

Setting (`lordnet_mini.py`): BPSK, m = 128, n = 16, one bit, b = 0, C = I known. H has i.i.d. N(0, SNR/n) entries; one fixed H per channel is used for pilots and
test. B = 512 pilots (the Fig. 9/10 setting); L = 30; delta = 0.01. Stage 1: Adam 1e-3, 400 epochs. Stage 2: Adam 1e-4, 400 epochs, diagonal G_i = W_i W_i^T,
mini-batch 512. x0 = 0. Our choices where the paper is silent: the H initialisation ("corr" = pilot correlation R_p^T X_p / B, rescaled; "rand" = random) and the
Stage-1 mini-batch (512).

| Mean BER over 6 channels | 4 dB | 8 dB | Paper's nearest number |
|---|---|---|---|
| Relaxed ML with the TRUE H: 700 plain gradient steps from 0, step grid-searched on the pilots (no learning) | **6.26e-3** | **6.9e-4** | - |
| True H, basic policy G_i = delta I, 30 layers (no learning) | 6.57e-3 | 5.9e-4 | - |
| True H + learned diagonal G_i ("LoRD-Net perfect CSI") | 6.13e-3 | 6.4e-4 | Fig. 10(a), 8 dB: 1.50e-3 |
| LoRD-Net two-stage, "corr" init: after Stage 1 / after Stage 2 | 9.31e-3 / **8.60e-3** | 1.20e-3 / **1.15e-3** | Fig. 8, B = 512: **8.73e-3 / 1.16e-3** |
| LoRD-Net two-stage, random init: after Stage 2 | 0.106 | 0.179 | - |
| Channels where LoRD-Net (Stage 2) beats relaxed ML with the true H | 0 of 6 | 0 of 6 | - |
| Learned H vs true H, "corr" init: cosine similarity / norm ratio | 0.967 / 0.90 | 0.976 / 0.83 | paper: the learned channel is not the true one |
| Stage-1 train BER vs test BER ("corr") | 3.6e-3 vs 9.3e-3 | 2.6e-4 vs 1.2e-3 | Fig. 9 reports an almost closed gap |

What this shows:
(i) Our LoRD-Net reproduces the paper's own B = 512 numbers (Fig. 8) closely, so the re-implementation is plausible.
(ii) The learned surrogate is indeed *not* the true channel. It points almost the same way, but it is 10-17 % smaller in norm.
(iii) In this setting a **non-learned** relaxed ML that knows the true channel beats LoRD-Net in every channel. LoRD-Net with the true H plus learned
G_i is no better than plain gradient steps.
(iv) Stage 1 is **sensitive to the H initialisation**, which the paper does not report; from a random start it gets stuck at BER 0.1-0.18.
(v) Stage 2 accelerates, but much less than in Fig. 10. At 8 dB the per-layer BER at layer 4 is 4.69e-3 after Stage 2 vs 6.56e-3 after Stage 1, and the curves
nearly meet by layer 30.
Caveat: 6 channels per SNR, 2 SNRs, one pilot size. This is a mechanism check, not a full reproduction of all figures.

### 5.4 OUR CHECK 2: how strong is the paper's nML baseline? (no learning, true H, same toy channels)

Script `lordnet_nml_check.py` (log `lordnet_nml_check.log`, JSON `lordnet_nml_check.json`). It implements the nML of Choi, Mo, Heath (2016): projected gradient ascent on the relaxed one-bit likelihood, started at the maximum-ratio point, with the ball constraint ||x||^2 <= K (K = n = 16; Choi eq. (28), "selected from a ball"; Algorithm 1, projected-gradient steps). LoRD-Net's own nML uses 700 iterations and a grid-searched step (LoRD-Net Sec. IV-A, "chosen using a grid search method"); we use the same. The step is chosen on the 512 pilots from {1e-4, ..., 1e-1}. Test: 2048 vectors x 16 bits per channel, 6 channels per SNR, the true H throughout. "GD from 0" is unconstrained gradient ascent from x = 0 (the same plain gradient as in 5.3, recomputed).

**Integrity check (OUR CHECK):** `lordnet_nml_nan_check.py` re-runs only the pilot-selected step of each variant and counts non-finite iterates. The final run printed overflow and invalid-value RuntimeWarnings (`lordnet_nml_check.err`); they come from unselected grid steps. The selected runs have **0 non-finite rows** in all 24 runs, and the recomputed BERs equal the logged values (`lordnet_nml_nan_check.log`).

| Detector, no learning, true H (2048 test vectors x 6 channels) | 4 dB | 8 dB |
|---|---|---|
| nML, Choi Algorithm 1 (ball radius sqrt(n)) | **6.24e-3** | **7.48e-4** |
| Unconstrained gradient ascent from 0 | 6.26e-3 | 6.92e-4 |
| Paper's nML, perfect CSI, Fig. 4(a), 128x16 Rayleigh (digitised) | 3.39e-2 | 1.04e-2 |
| Paper's nML, perfect CSI, Fig. 10(a), index 29 (digitised) | - | 1.43e-2 |

Per channel, the gradient beat nML on 4 of 6 channels at each SNR (one tie at 8 dB). Radius 2, 4, 8 and infinity gave identical BER on channel 0 in a first run (`lordnet_nml_check_run1_partial.log`), so the radius does not matter in this setting.

What this shows:
(i) A faithful nML is **not** weak: at 4 dB it matches plain gradient ascent with the true H, and both have lower BER than LoRD-Net's surrogate-H Stage 2 on all 12 channels (Sec. 5.3; per-channel values in `lordnet_mini.log` and `lordnet_nml_check.log`).
(ii) The paper's nML curves are **5.4x (4 dB) and 14x (8 dB)** higher BER than our faithful nML at the nominal same SNR and channel family. Our SNR convention matches eq. (17) (SNR = E||Hx||^2 / E||n||^2), so the two should agree. The cause is **UNVERIFIED**: it could be the paper's nML implementation details (step rule, start point, constraint handling), or how the plotted SNR is scaled with C = I fixed (Sec. 2 above). We cannot tell which from the paper. (INFERENCE:) most of the reported "up to 24x" gap to nML therefore reflects that baseline rather than the learned surrogate.
(iii) Caveat: 6 channels per SNR, two SNRs, one pilot size, and our reading of the paper's figures is digitised from vector graphics. This is a mechanism check, not a reproduction of the paper's nML numbers.

---

## 6. Citable claims (with locators)

House rule for this project's notes: at most one verbatim quotation per document, so only claim 1 is quoted. Claims 2-8 are close
paraphrases with exact locators (arXiv v1); check the wording in the paper before quoting any of them in our manuscript.

1. **The learned model is not the true channel.** Quote (Sec. I, p. 2): "the alternative channel learned by LoRD-Net is in general not the true channel."
   *Cite for:* learning a task-optimal surrogate model rather than estimating the physical one. Also supported for our angle problem by OUR CHECK
   (cosine 0.97-0.98, norm 0.83-0.90 of the true H).
2. **After relaxation, the true parameters lose their optimality** (Sec. II-B, last paragraph, p. 4; illustrated in Fig. 2, p. 4). The true Theta is optimal only for
   the discrete problem (4), not necessarily for the relaxed one (5). *Cite for:* motivating a learned surrogate likelihood or surrogate noise level when
   the ML search is approximate or relaxed.
3. **The one-bit relaxed ML is convex** because Q is log-concave (Sec. II-B, first sentence of p. 4). *Cite for:* the contrast with our non-convex angle problem.
4. **Projecting inside every layer fails** (vanishing gradients and worse performance), so projection is applied only at the output, and only at inference
   (Sec. III-B, p. 5). *Cite for:* keeping our layers continuous and making discrete decisions (rounding, grid snapping, candidate selection) only at the end.
5. **Preconditioners kept PSD by G_i = W_i W_i^T, and two-stage training** (Sec. III-C, p. 6; Algorithm 1, p. 6). *Cite for:* our learned per-layer step and
   preconditioner design, and for staged training.
6. **2,528 trainable parameters for 128x16, i.e. n(L + m), against more than 8 x 10^5 for DeepSIC** (Sec. IV-B, p. 8; Benchmarks 1-2 in Sec. IV-C, p. 9).
   *Cite for:* parameter efficiency of model-based unfolding.
7. **Most of the improvement comes in Stage 1; Stage 2 mainly accelerates, so about 10 layers suffice** (Sec. IV-D, p. 11; Figs. 9-10). Joint one-stage
   and alternating training show larger generalisation gaps (Fig. 9, p. 11). *Cite for:* "learn the model first, then the step sizes", and for the
   claim that joint end-to-end training of everything can overfit small data.
8. **Multi-bit ADCs and optimised thresholds are left for future work**, and multi-bit is described as a simple change of eq. (3) (Sec. III-D, p. 7).
   *Cite for:* the fact that LoRD-Net was never shown at 2+ bits, with AGC, or with complex data. These are gaps our work covers.

---

## 7. Limitations / what the paper does not show

1. **One bit, BPSK, real-valued only.** No multi-bit, no AGC or clipping, no thresholds other than zero, no complex constellation (Secs. III-D, IV-A).
2. **Training per channel and per SNR.** H is a learned parameter, so a trained LoRD-Net is tied to one channel realisation (INFERENCE from the
   formulation). Generalisation to a new channel is never tested; the remedy proposed is periodic re-training. With B = 512 pilots and 16 users, the
   pilot overhead is 512 channel uses per channel.
3. **Weak and narrow baselines.** The only coherent baseline is nML [30] (plus a "perfect CSI" LoRD-Net with learned steps in Fig. 10). The paper
   describes the obvious alternative (estimate H from the pilots, then run the relaxed or nML detector) in Sec. III-A but never plots it. In OUR CHECKs, a
   plain relaxed-ML gradient with the true H beat LoRD-Net's Stage-2 network (surrogate H) on all 12 channels (`lordnet_mini.log`), and a faithful nML was far better than the plotted nML (Sec. 5.4).
   Figure 10 itself shows that learned step sizes on the true model already explain about 10x of the reported gap to nML.
4. **No uncertainty.** No error bars, no number of runs; test sets of 32,768 bits mean a few to a few tens of errors at the lowest BERs. The same setting
   differs by 1.7x between figures (Sec. 5.2).
5. **Unreported details that matter.** x0, the H initialisation (our check: a random init fails), the Stage-1 batch size, the SNR scaling with C = I,
   and the number of averaged runs (Sec. 3).
6. **Internal inconsistency.** Sec. III-C recommends fewer layers in Stage 2 than in Stage 1, but the experiments use 30 in both. Eq. (3) has a sign/label slip (Sec. 3, Step 1).
7. **No timing.** Complexity is argued only as 30 layers vs 700 iterations. There is no wall-clock or FLOP count, and no cost of the per-channel training.
8. **No analysis of the learned surrogate** beyond "it is not the true channel" (no norm, no direction, no interpretation). OUR CHECK suggests it is mainly a
   shrunken version of the true channel.
9. **No usable public code.** The paper cites a repository that holds only a README and a LICENSE (checked 2026-10-16). The published version could not be compared with v1 (UNVERIFIED).

---

## 8. Relevance to us

**Ideas that transfer to angle estimation from quantised beamspace data (with how):**

1. **Unfold gradient (or LM) steps on the exact quantiser likelihood, with a learned PSD diagonal preconditioner per layer.** For us the likelihood is
   our existing cell-probability likelihood (`finetune/adversarial/mlq.py`; one-bit is the zero-threshold special case: the LoRD-Net term
   log Q(r_i (b_i - h_i^T x)/sigma) is exactly our 1-bit cell probability). The unknowns are (alpha, u, v) instead of x, so the step acts on
   gains and angles. Caveat: our ML already uses Levenberg-Marquardt, which is itself a curvature-adapted (preconditioned) step. LoRD-Net's own
   Fig. 10 shows learned steps mainly reduce the number of layers needed, not the final error. Expect a speed gain (fewer LM layers than 13 starts x many
   iterations; ML costs 93.8 ms per start, 482.4 ms with all starts), not an accuracy gain.
2. **"Competitive objective": learn a few parameters of the model used inside the optimiser, not the optimiser.** Our transferable analogue is not the
   channel (our operator A(u, v) is known) but the **likelihood's nuisance parameters**: an effective noise level / temperature per bit-depth and SNR bin, and
   possibly an AGC-scale correction, trained so that the *final estimate* lands on the true angles. This directly targets two verified weaknesses:
   (a) ML loses 0.04-0.07 Pd when its assumed SNR is off by 5 dB, and at 1 bit the noise level is not identifiable from the data;
   (b) in unfolded_v2 the ORACLE pick of the candidate pool is +0.036 to +0.060 above 13-start ML, but picking by lowest true-model NLL only reaches ML's level.
   That is exactly the situation LoRD-Net describes: the true-model objective is not the best objective once the search is approximate. A learned
   **selection score** (NLL with learned sigma_eff plus a handful of scalars) is the LoRD-Net idea applied to our selector.
3. **Two-stage training with very few shared parameters.** Stage 1: fix the search/optimiser policy (our classical LM polish, fixed steps) and learn
   only the few shared model parameters (sigma_eff, selection scalars). Stage 2: freeze them and learn per-layer step / damping scalars. Initialise
   everything so that the network starts as the classical algorithm. This is the opposite of what unfolded_v2 did (hypernet, fusion, step MLP, selector and
   stop head trained jointly, none improved validation over initialisation). LoRD-Net's Fig. 9 shows the same failure mode for joint training (larger
   generalisation gap), and OUR CHECK shows Stage 1 is fragile to initialisation, so start from the classical values.
4. **Make discrete decisions only at the end.** Keep angles continuous inside the layers, and do any grid snapping, rounding, or selection of the best
   candidate once, at the output (claim 4).
5. **Use an MSE-type loss on the continuous output** (theirs: squared error to the pilot symbols; ours: matched angle error after Hungarian matching)
   rather than a loss on hard decisions.

**What does not transfer.** Learning the channel matrix itself (our scenes change every snapshot, so there is no fixed H to learn); the convexity argument
(our problem has local minima in (u, v), which is why 13 starts plus EM, and CNN seeds, matter); the BER metric.

**Must it be a baseline?** **No, not as published.** It solves a different task (detecting linear BPSK symbols in one fixed unknown channel, trained per channel
from pilots, one-bit only), and there is no code. It cannot be run on our benchmark without redesigning it. The fair thing is to (i) cite it as the prior
art for unfolded one-bit ML with learned preconditioners and a learned surrogate objective, and (ii) if we build the LoRD-Net-style variant in point 2-3,
report it as an ablation in our setting: "unfolded from quantiser-ML gradient (LoRD-Net-style surrogate noise + PSD preconditioner)". Fair comparison
rules: same banks (test 12000-12999, fresh adversarial 13000-13999), same ADC model with per-scene peak AGC, same metrics (paper-style Pd, strict Pd,
RMSE), same seeds/candidates as the 13-start ML and the hybrid, the same SNR knowledge (report both known-SNR and blind variants), latency at 1
thread, and paired bootstrap CIs (2000 resamples within SNR groups, seed 0). Compare against the strongest non-learned method that shares its search,
not only against NOMP. LoRD-Net's own weakness was comparing against a weaker relative (Sec. 5.4).

**What it implies for our network design.** (a) Spend learning capacity on the objective (surrogate noise/selection score), not on a big
step-generator. (b) Use very few shared parameters, train in stages, initialise at the classical algorithm. (c) Expect an unfolded quantiser-ML to be
mainly a faster ML, unless the learned objective fixes the selection gap. (d) Always include the non-learned "same search, true model" baseline. OUR CHECK
shows that, without it, a learned-surrogate gain can be an artefact of the baseline.

---

## 9. "If a reviewer asks..."

**Q1. "LoRD-Net already unfolds the one-bit ML with learned preconditioners and a learned model. What is new in your decoder?"**
LoRD-Net detects linear BPSK symbols in one fixed, unknown channel. It learns that channel from 512-2048 pilots per channel realisation and per SNR,
at one bit with zero thresholds, real-valued, and its relaxed objective is convex (Secs. II-B, IV-A). Our problem differs in each of these respects.
The unknowns are continuous angles that enter nonlinearly (a non-convex likelihood with local minima). The data are complex beamspace samples at 1-2 bits
with a per-scene AGC. Each scene has its own channel and one snapshot, so nothing can be fitted per channel, and the learned parts must generalise
across scenes. The physical model (W, F, arrays) is known. Where we use LoRD-Net's ideas (staged training, PSD per-layer steps, a learned surrogate
objective), we credit them and call the layers "unfolded from the quantiser-likelihood gradient / LM".

**Q2. "Why do you not compare with LoRD-Net?"**
It cannot be applied as published. It needs the same channel for all pilots and test data, it estimates linear symbols rather than angles, it is one-bit only,
and no code is public. A faithful port would be a different algorithm. We instead report the LoRD-Net-style ablation inside our framework under the
fair-comparison rules of Sec. 8, and we compare against the strongest non-learned method with the same search (13-start quantiser ML with EM). Be honest:
this ablation has **not been run yet**.

**Q3. "LoRD-Net reports a learned detector beating ML-type detection by up to 24x in BER. Why does your learned hybrid gain only +0.03 Pd over quantiser ML?"**
LoRD-Net's ML-type reference was the nML of Choi et al. as the authors ran it. In our re-implementation of their own toy setting, a plain non-learned
relaxed-ML gradient with the true channel beat LoRD-Net's Stage-2 network (surrogate H) on all 12 of 12 channels (mean BER 6.26e-3 vs 8.60e-3 at 4 dB; 6.9e-4 vs 1.15e-3 at 8 dB; per-channel values in `lordnet_mini.log`), and a
faithful nML was close to it (Sec. 5.4). So the large reported gap mostly reflects that baseline. Our reference is the strongest classical method we could
build: exact cell-probability ML with 13 starts plus EM. Against it, the hybrid (ML started from CNN seeds) gains +0.032 Pd at 1-bit / 15 dB (0.572 vs 0.540,
95 % CI excludes 0, replicated +0.028 / +0.036) and ties in the other cells (2-bit / 15 dB 0.687 vs 0.682; 1-bit / 0 dB 0.274 vs 0.275; 2-bit / 0 dB 0.471 vs 0.476).
The remaining headroom is bounded by ML started at the true angles (0.626 at 1-bit / 15 dB).

(Extra, if asked about unknown SNR.) LoRD-Net learns a diagonal C and reports only a small, non-uniform gain (Fig. 6). Our ML assumes a known SNR and loses
0.04-0.07 Pd when it is wrong by 5 dB. The network read-out needs no noise knowledge. This is where a learned surrogate noise level (Sec. 8, point 2) is most
likely to pay off.

---

## 10. Glossary (plain English)

| Term | Meaning |
|---|---|
| One-bit ADC / one-bit quantiser | Converter that only records the sign of its input (above or below a threshold) |
| Threshold vector b | The reference levels the ADCs compare against (zero in all their experiments). Not "bits" |
| Blind / non-coherent detection | Detecting symbols without being told the channel |
| CSI | Channel state information: knowledge of the channel matrix H (and noise level) |
| Pilots / labelled samples | Known transmitted symbols with their received signs, used for training |
| Likelihood, log-likelihood | How probable the observed data are for a given guess; its logarithm is easier to optimise |
| Q-function, Q(z) | Probability that a standard Gaussian exceeds z; the one-bit likelihood is built from it |
| Log-concave | Its logarithm is concave (one hill, no false peaks), which makes the relaxed problem convex |
| eta = Q'/Q (inverse Mills ratio, with a minus sign) | Ratio that sets how strongly each "surprising" sign pushes the estimate |
| MLE / ML | Maximum-likelihood estimate: the guess that makes the data most probable |
| Relaxed ML | ML where discrete symbols are allowed to be any real numbers, then rounded |
| Projection P_M | Rounding a real vector to the nearest allowed symbol vector (sign for BPSK) |
| nML | "Near-ML" detector of Choi, Mo, Heath (2016): projected gradient on the relaxed one-bit likelihood with a norm-ball constraint, then normalise and round |
| DeepSIC | Data-driven soft interference cancellation detector (Shlezinger et al.), a generic learned detector with about 8 x 10^5 parameters here |
| Deep unfolding / unrolling | Writing a fixed number of iterations of an algorithm as network layers, then learning some of their parameters |
| Preconditioner G_i | A matrix (here diagonal) that rescales the gradient in each layer; a learned per-user step size |
| PSD / positive (semi-)definite | A matrix that never turns a descent direction into an ascent direction; guaranteed by writing G = W W^T |
| Competitive objective / surrogate model | A model (here a learned H, optionally C) chosen so that optimising *its* likelihood gives the right answer, even if it is not the true model |
| Basic optimisation policy | All G_i = delta I, i.e. plain gradient descent with step delta, used in Stage 1 |
| Two-stage training | Stage 1 learns the surrogate model with fixed steps; Stage 2 learns the steps with the model frozen |
| One-stage / alternating training | Learning everything jointly / switching between the two stages every epoch |
| Generalisation gap | Test error minus training error; large means overfitting |
| BER | Bit error rate: the fraction of wrongly detected bits |
| Rayleigh fading (here) | Channel entries drawn i.i.d. Gaussian |
| COST-2100 | A standard geometry-based stochastic channel simulator; here 2.6 GHz, indoor, closely spaced users, ULA at the base station |
| Epoch | One pass through the training set |
| Adam | A standard stochastic-gradient optimiser |

---

### Appendix A - correction to our literature survey

`deep_research/unfolded_v2/lit/unfolding_survey.md` (row [36] and lessons (ii)) says LoRD-Net's gain shows that learned parts "can beat the plain
relaxed-ML gradient on a non-convex quantised objective". Two corrections, both verified here. (1) LoRD-Net's relaxed objective is **convex** (Sec. II-B, p. 4).
(2) LoRD-Net is compared with nML [30], not with the plain relaxed-ML gradient. In OUR CHECK the plain relaxed-ML gradient with the true channel was **not**
beaten by LoRD-Net (0 of 6 channels at 4 dB, 0 of 6 at 8 dB). So LoRD-Net should not be cited as evidence that an unfolded quantised-likelihood
method beats its non-learned parent when both use the same model and search. It shows a win over a particular nML implementation, achieved without CSI.
