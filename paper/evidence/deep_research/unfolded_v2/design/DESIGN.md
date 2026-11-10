# Unfolded decoder v2: design choice and specification (unfolded_v2/design)

Written 2026-10-16. Nothing has been trained or tested for this design yet. `criteria.json` (same folder) holds the
pre-registered success criteria; it was written before any training or test run and must not be edited after the
first test-bank evaluation (later changes go in a separate `criteria_amendments.json` with a reason and timestamp).

Every design below is **unfolded from NOMP (Mamandipoor, Ramasamy, Madhow, IEEE TSP 2016)**. Its quantised-data parts are also
unfolded from quantisation-aware NOMP / ML (GNOMP, Zhu et al., IEEE TAES 2026). NOMP, NOMP os8 rc6 and the strongest classical
variants stay named baselines in every table.

## 0. What the measured headroom allows (inputs to the choice)

| mechanism | measured legal headroom | source | used? |
|---|---|---|---|
| learned Newton-type step on the ML objective, ideal hw | 0 (lurd re-learned Newton, 0.91 corr.; -0.0073* vs NOMP) | nn_decoder/REPORT.md | only as a zero-initialised correction, never as the gain mechanism |
| candidate selection (ML / marginal lik. / posterior mass / MBR / learned GBM), ideal hw | +0.0007 vs os8rc6 at best, n.s.; learned selector -0.0007 / -0.0018 | diag 2 | NOT expected to gain on official bank |
| min-separation (pi/6) prior / MAP | +0.0003 legal; -0.03* when prior violated | diag 3 | excluded from all learned inputs (no pairwise-separation feature) |
| SNR-adaptive settings, ideal hw | +0.0006 oracle | diag 5 | not a gain mechanism on ideal hw |
| self-calibration, ideal hw | -0.001 to -0.002 | diag 5 | strength output by hypernetwork, expected to learn 0 on clean data |
| -10 dB posterior-mass decision | +0.007* (only at -10 dB) | diag 4 | may be learned by the selector; small |
| **1-bit ADC search loss** | wide classical restart +0.014*, ML from network seeds +0.032* vs 13-start ML | finetune/adversarial s3, s7 | **yes: learned seeds + annealing** |
| **ADC objective/decision headroom** | ML from truth 0.626/0.719/0.439/0.540 vs 0.540/0.682/0.275/0.476 | s3 | yes, but its search-vs-objective split is unknown -> gate G0 |
| **assumed-SNR mismatch, ADC** | ML loses 0.04-0.07 at +-5 dB | s3 | **yes: learned noise level (sigma_hat, kappa, eta)** |
| **position error M2 (sigma 0.2, 15 dB)** | network seeds +0.044* vs NOMP, +0.055* vs POS-MAP | finetune/adversarial VERDICT | yes: learned seeds |
| phase noise M6 | network-seeded PN-ML +0.012* over PN-ML; seeds alone lose to PN-ML | VERDICT | yes: heteroscedastic noise term eta (PN-ML-like) |
| large per-antenna phase/gain M3 | wins belong to IABC-v2b; EB-ML-g5 is the classical bar | iabc_v2, VERDICT | expected loss vs EB-ML (pre-registered as such) |
| unknown number of paths | not measured; literature: learned order selection can beat fixed-threshold detectors | lit [7,26] | yes: learned stopping head vs NOMP's own residual-threshold (CFAR-type) stopping |

Conclusion used for the design: on ideal hardware with the benchmark prior a NOMP-derived decoder can at best tie NOMP os8 rc6
(the official-bank primary criterion is pre-registered with a predicted FAIL). Real headroom exists where NOMP's Gaussian
likelihood is wrong or under-specified (1-2-bit ADC, unknown noise level, position error, phase noise) and in model-order
selection. The designs therefore change the likelihood (quantised + heteroscedastic, noise level estimated, annealed) and the
starts, not the step rule.

## 1. Ranked designs

### Rank 1 (train): A = "CA-UNOMP", condition-adaptive unfolded NOMP with quantised/heteroscedastic likelihood, learned seeds, learned stopping and a learned hypothesis selector

Fixed classical structure (unfolded from NOMP / GNOMP), per hypothesis, batched over hypotheses and scenes:
1. Data model: z = W^H H F + n, y = ADC_b(z) (b in {1,2,3,4,inf}; the receiver knows b and its AGC, as in s3). Per real sample i the
   likelihood is the interval-censored Gaussian of mlq.py with std s_i = sqrt(kappa_t^2 sigma_hat^2 + eta_t |mu_i|^2)
   (b = inf: Gaussian NLL = NOMP's residual energy when eta = 0, kappa = 1). eta models multiplicative errors
   (phase noise, gain/phase errors) as in PN-ML; kappa_t >= 1 is an annealing temperature (graduated non-convexity).
2. Greedy detection, k = 1..L_max (8): E-step de-quantisation z_bar = E[z | cell, mu, s] (truncated-normal mean, as in mlq.em;
   identity for b = inf); residual r = to_ant(z_bar - mu); detection map = fusion conv (learned, below) over
   |FFT_os4(r)|^2 / ||r||^2, the CNN heat logits (upsampled to 64x64) and a mask of already-detected paths; new path at the
   arg-max (hypothesis branches take the 2nd/3rd peak, MMP-style); gains by exact (convex) quantised-NLL LM, 2 iterations
   (LS for b = inf).
3. Cyclic refinement after each detection: 2 rounds over the current paths; per path one damped Newton step on (u, v) of the
   current NLL with the other paths subtracted (residual subtraction), step = -g/(h + mu_t), clipped to c_t, plus
   g_t * MLP_corr(features) (zero-initialised), then the joint gain update. Self-calibration (exp5 rank-one per-antenna fit,
   identifiable-subspace projection) with strength lambda_t.
4. Final polish: 4 joint layers (all paths, LM on all 4K real parameters as in mlq.lm), kappa annealed to its final value.
5. Stopping: L given -> exactly L detections (official protocol). L unknown -> stop head after each detection.
Unrolled sequential layers for L = 3: 12 cyclic path-steps + 4 joint = 16 (layer index t = 1..24 for L up to 8; deeper scenes reuse t = 24).

Learned parts (parameter counts computed from the layer shapes):
| module | layers | params |
|---|---|---|
| seed CNN | IABRv2(32,10,32) initialised from finetune/adc/cache/adc_ft.pt; outputs heat 32x32 + Hc | 191,702 |
| bit/energy FiLM | Linear(6 -> 640): per-block (gamma, beta) for 10 blocks x 32 ch from bits one-hot(5) + log scene energy | 4,480 |
| scene head | GAP(trunk, 32) + 8 hand statistics (fraction of samples in outer ADC cells, energy moments, top-5 FFT peak ratios) + bits(5) = 45 -> 256 -> 256 -> 256 -> 15 (log sigma_hat; impairment-type logits {none, coupling, position, per-antenna phase/gain, phase noise}; L_hat 0..8) | 147,215 |
| hypernetwork | [bits(5), log sigma_hat, impairment probs(5), L_hat probs(9), Embedding(24,16) layer index] = 36 -> 256 -> 256 -> 256 -> 8 per layer: kappa_t, mu_t, c_t, g_t, lambda_t (self-cal), beta_t (heat weight in detection), eta_t, stop bias | 143,496 |
| detection fusion | conv3x3(5->32) -> conv3x3(32->32) -> conv3x3(32->1), circular padding, inputs FFT power, heat logit, detected mask, log sigma_hat, bits | 11,009 |
| step correction | MLP 16 -> 256 -> 256 -> 256 -> 2, last layer zero-init; inputs: g/sqrt(h) (u,v), log h (u,v), normalised cross-curvature, 4 probe NLL second differences (delta 0.02, 0.08 on u, v), max coherence with other paths, log |x|^2/sigma_hat^2, kappa_t, t/24, bits/4, log sigma_hat, eta_t | 136,450 |
| stop head | [Delta NLL / E[chi^2], |x_new|^2/sigma_hat^2, heat at new path, k/8, bits(5), log sigma_hat, L_hat probs(9)] = 19 -> 128 -> 128 -> 1 | 19,201 |
| hypothesis selector | per-hypothesis token (24 features: NLL at kappa=1 and at kappa_T with sigma_hat, NLL rank, per-path log Fisher det (Laplace evidence), sorted |x|, heat at estimates, n paths, sigma_hat, bits, eta) -> Linear(24,128) -> 2 transformer encoder layers (d 128, 4 heads, FF 256) -> Linear(128,1); listwise softmax over K = 6 hypotheses. NO pairwise-separation feature | 268,289 |
| **total** | | **921,842** |

No parameter was cut to save size; widths were set by CPU training time and by keeping the pretrained 32-channel trunk.
Optional A+ (only if stage-1 validation shows the seed CNN is the bottleneck, decided on val before any test run): 64-channel
trunk (~0.75 M) trained from scratch in stage 1 (adds ~1 h; then stage 2A is shortened to stay within budget).

Hypotheses (K = 6, all run through the same unfolded core): H1 greedy NOMP-style detection; H2 CNN-heat seeds (top-L peaks)
then cyclic + polish; H3-H4 first detection at the 2nd/3rd fused-map peak; H5-H6 second detection at the 2nd/3rd peak (MMP
branches). For b <= 2 H2 is also run at kappa_1 x 2 (wider annealing).

What it effectively optimises: each hypothesis ends near a stationary point of an adapted likelihood (quantised,
heteroscedastic, noise level estimated per scene, annealed), and the selector picks the hypothesis with the most expected hits
under the training distribution (an approximation of the Bayes rule for the Pd hit-or-miss cost). Why this can beat NOMP / the
classical bar: (i) where the data are quantised or carry multiplicative errors, NOMP's Gaussian residual is the wrong objective
(model correction); (ii) quantised ML needs the true SNR (-0.04 to -0.07 at +-5 dB) - here sigma, kappa, eta are estimated per
scene; (iii) 1-bit ML has measured search loss (+0.014* / +0.032*) that learned seeds and annealing attack; (iv) a learned
stop rule can use evidence (heat, curvature, SNR estimate) that a residual threshold ignores. On ideal hardware with known L
it reduces to NOMP + multi-hypothesis selection, for which the diagnosis predicts a tie with NOMP os8 rc6.

Adaptivity: bits (FiLM + hypernet), estimated SNR (sigma_hat drives kappa, mu, c, beta, stop), impairment type/level (eta,
lambda), variable depth (stop head; L unknown) and early exit of the polish when the max step < 1e-4 rad (fixed rule).

### Rank 2 (train, cheaper; also the main ablation of A): B = "HU-NOMP", hyper-conditioned unfolded NOMP

Same fixed core, seed CNN + FiLM, scene head, hypernetwork, fusion conv and stop head as A (517,103 params). Not learned: no
step-correction MLP (pure damped Newton / LM steps with learned per-layer kappa, mu, c, eta, lambda) and no learned selector:
K = 2 hypotheses (H1 greedy, H2 heat-seeded), chosen by the lowest NLL at kappa = 1, sigma_hat. This is the HyperLISTA-style
"learn only the hyperparameters" design. A - B isolates what the learned steps and the learned selector add.

### Rank 3 (not trained): C = posterior-mass decoder for ideal hardware (gain-marginalised evidence + separation prior + MCMC, per-path +-1 deg mass)
Rejected by the measured headroom: legal gain <= +0.0008 over os8 rc6 (n.s.), +0.007* only at -10 dB, and its main lever (the
pi/6 separation prior) costs -0.03* when violated. Kept only as the documented negative branch.

## 2. Training data, losses, schedule

Online generation with stress_sim.base_scenes / generate and iabr2_sim.sample_paths(sep=...). Training seeds
default_rng([70_200_000, step]); validation mix: 2,400 fixed scenes, seed 71_200_000 (+offset per component); G0 checks seed
71_200_100+. Disjoint from all earlier seeds (70,000,000 / 71,000,000-71,000,002 / 9500 / 9501 / 9515 / 9595 checked by grep).

Mix per batch (SNR uniform continuous in [-12, 27] dB unless stated):
* 30% ideal benchmark distribution (L = 3, sep pi/6, gains CN(0,1/L) sorted), b = inf;
* 35% ADC peak-AGC M5, b uniform in {1,2,3,4} (10% of these with the clipping AGC M5c);
* 20% impairments: M1 rho <= 0.3, M2 sigma <= 0.2, M3 phase <= 45 deg / gain <= 6 dB, M6 sigma <= 30 deg, each with
  probability 0.25, severity uniform; 20% of these also quantised (b in {2,3,4});
* 15% prior-violating geometry: L uniform 1..6, separation uniform in [0, 30] deg, b = inf or 2.
M4 (phase-shifter bits) and M8 (near field) are held out entirely (out-of-distribution tests).

Loss (deep supervision over hypotheses and the last 4 layers, weights 0.1/0.2/0.3/1.0):
* matched hit loss: Hungarian matching of estimated to true (psi, phi) pairs (cost = sum of |deg errors|); per matched
  component 1 - sigmoid((1 deg - |err|)/tau), tau annealed 0.5 -> 0.15 deg; + 0.05 x Huber(err, 1 deg); unmatched true path =
  2 misses, unmatched estimate = false alarm 1.0 (L unknown only);
* selector: listwise cross-entropy to softmax(hits_k / 0.25) over hypotheses (hypotheses detached);
* stop head: BCE to "true L not yet reached";
* auxiliary: heat BCE and Hc impairment MSE (as train_ft.py), log sigma MSE, L_hat CE, impairment-type CE (weights 1, 50, 0.5, 0.2, 0.2).
Adam, warm-up 100 steps, cosine 3e-4 -> 1e-5, grad-norm clip 5, 6 torch threads, checkpoint every 100 steps, resumable.

Schedule and CPU estimate (8 logical CPUs; adc_ft.pt trained at 0.9 s/step, batch 128):
| stage | what | steps x s/step (est.) | time |
|---|---|---|---|
| G0 | gates (below), 200 scenes per ADC cell, numpy | - | 0.5 h |
| S1 (shared) | CNN + FiLM + scene head, aux losses only, batch 128 | 4000 x 0.9 | 1.0 h |
| S2B | B: hypernet + fusion + stop head through the unrolled core (K = 2), front end LR x0.1, batch 64 | 2000 x 2.0 | 1.1 h |
| S2A | A: init from S2B, add step MLP + selector (K = 6), batch 48 | 3000 x 3.0 | 2.5 h |
| S3 | A joint low-LR fine-tune (1e-5), 50% benchmark distribution | 600 x 3.0 | 0.5 h |
| total | | | ~5.6 h |
A 50-step bench run fixes s/step before S1/S2; if a stage is > 1.5x slower than estimated, steps (not widths) are reduced and
the change logged. Checkpoint selection per design only on the validation mix (mean Pd over its cells), never on test banks.

Inference latency estimate (1 thread, batch 1): CNN 6 ms + 16-24 unrolled layers on a batch of K hypotheses (~1.5-2.5 ms each in
torch) + 3 fusion convs + selector -> A 45-80 ms, B 35-60 ms (multi-start quantised ML 482 ms, NOMP 5.8 ms, lurd pipeline ~98 ms).

## 3. Gates before full training (G0, pre-registered decisions)
* G0a (search vs objective on ADC): on 200 val scenes each of b1/15, b2/15, b1/0, b2/0, the share of scenes where ML from the
  truth reaches a LOWER exact NLL than 13-start ML. G0b: classical annealed ML (kappa 3 -> 1, same 13 starts) Pd vs ML.
  G0c: oracle Pd over B's/A's classical hypothesis pool (H1-H6 with classical steps).
* Decision: if the truth has lower NLL in < 10% of failing scenes in all four cells AND the G0c oracle is < +0.02 over ML in
  all four cells, the ADC gain cannot come from search/selection; A's selector stage is dropped (A = B + step MLP) and this is
  reported. Classical annealed ML (G0b) becomes a named baseline in every case.

## 4. Evaluation plan
Banks (fresh seeds 9520-9561; n = 1000 unless stated):
* official frozen eval_bank (8000, L given) + fresh ideal replicate seed 9520 (4000, 500 per SNR, -10..25).
* ADC M5 peak AGC: (b, SNR) = (1,15) 9521, (2,15) 9522, (3,15) 9523, (4,15) 9524, (1,0) 9525, (2,0) 9526, (3,0) 9527, (4,0) 9528;
  M5c clipping b2/15 9529 (report only); SNR sweep b1/b2 at -5, 5, 10, 25 dB, 500 each, 9530-9537 (report only).
* impairments: M2 0.2/15 9541, M6 30/15 9542, M6 30/0 9543, M3 (45,0)/0 9544, M3 (30,4)/15 9545, M1 0.3/15 9546;
  OOD report only: M8 0.5 m/15 9548, M4 2-bit/15 9547.
* unknown L: fresh seed 9550 (L uniform 1..6, SNR -10..25, 2000) + cached L1/L2/L4/L5/L6 at 0/15 dB.
* prior violation: cached sep_1/2/3/5/10/20deg_snr15; fresh nosep 9560 and sep10 9561 (L = 3, -10/-5/0/5 dB).
Baselines (all named): NOMP os4 rc3 (Mamandipoor 2016); NOMP os8 rc6; NOMP + tuned SC; consistent NOMP; quantised ML 13 starts
+ EM (true sigma; and sigma +-5 dB); quantised ML with classical joint sigma estimation (log sigma added to the LM parameters,
same starts; new classical baseline); classical annealed ML (G0b); hybrid ML from FT-network seeds (prior best system,
not classical); EB-ML-g5 (M3), PN-ML (M6), POS-MAP (M2) from finetune/adversarial; NOMP with its residual-threshold stopping
(P_fa 0.01, true sigma and estimated sigma) for unknown L; lurd all-neural pipeline; U-Net (Lloria et al.).
Per cell "best classical" = the best classical method on that test bank (chosen on the test bank, favouring classical).
"Deployable classical" = one fixed bit-aware classical pipeline, not told SNR or impairment: b <= 4 -> quantised ML with joint
sigma estimation; b = inf -> NOMP os8 rc6 (L given) or NOMP residual-threshold stopping with estimated sigma (L unknown).
Tables: Pd per SNR (official, replicate), per cell with paired bootstrap CIs (2000 resamples within SNR groups, seed 0),
strict Pd, RMSE, hit-F1 and false alarms for unknown L, latency (1 thread batch 1, and batch 256). Ablations of A: kappa fixed
at 1; eta fixed 0; sigma_hat replaced by true sigma; step MLP off (g = 0); selector -> lowest NLL; heat weight beta = 0.
All stdout saved as .log next to each script; large arrays in C:/Users/Milon/iabr_deep_research_cache/dr_cache/unfolded_v2/design/.

## 5. Pre-registered predictions (honest prior expectations)
* P1 official bank vs NOMP os4 rc3: likely + (about +0.0015, as ML over a pool in diag); vs NOMP os8 rc6: predicted n.s. -> P1 predicted FAIL.
* ADC: 1-bit/15 dB win likely (seeds +0.032* already measured); 2-bit/15 dB uncertain; 0 dB cells predicted ties unless G0 shows
  search headroom; 3-4-bit cells uncertain. Unknown SNR (vs mismatched or jointly-estimated sigma ML): win more likely.
* Impairments: M2 win likely; M6 uncertain (eta vs PN-ML); M3 and M1 predicted losses vs EB-ML-g5 / best classical.
* Prior violation: no separation feature is used, so losses should stay < 0.01; L != 3 depends on the stop head.
Expected overall label: "REGIME-ONLY" or "NULL" (see criteria.json), not "FULL".
