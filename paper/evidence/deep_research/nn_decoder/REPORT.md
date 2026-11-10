# All-neural IABR-Net at full scale (CNN heatmap peaks -> learned decoder), compared with published work

Folder: `paper/evidence/deep_research/nn_decoder/`. Every number below is read from a file written by a script in this folder (stdout in the `.log` next to it): `s0_inventory` (banks), `s1_reproduce` (reproduces `learned_decoder/results5.json` exactly), `s2_predict` (predictions), `s3_stats` (metrics, CIs, per-SNR, bank counts, figure), `s4_latency`, `s5_resnet` (optional ResNet port + validation), `s6_agreement` (scene-level agreement with the classical decoder), `s7_report` (this file). No existing file was modified; `final.pt` and `lurd.pt` were only read.

## Verdict

* **Official bank (N=8000, L=3, ideal hardware)**: all-neural IABR-Net (SC on) paper-style Pd 0.7508 [0.7468, 0.7545]; SC off 0.7457. It beats the published U-Net (0.7366; paired d +0.0142 [+0.0108, +0.0176], strict-Pd d +0.0568 [+0.0530, +0.0605]) and the published ResNet (validated port, 0.7103; d +0.0404 [+0.0369, +0.0439]), but it is **below NOMP-only** (0.7581; d -0.0073 [-0.0091, -0.0056]) and statistically identical to the same CNN with the classical heat-seeded decoder (0.7503; d +0.0005 [-0.0002, +0.0011]).
* The learned decoder reproduces the classical Newton/cyclic/self-calibration decoder rather than improving on it: on the official bank 85.5% of scenes agree with the classical decoder to < 0.1 deg in every angle (median max difference 0.029 deg), and only 36 vs 45 scenes are fully resolved by one and not the other.
* **Across banks** (paper-style Pd, 95% CI of the paired difference above / below / containing 0): L=3 test banks (28): vs U-Net (14, 0, 14), vs NOMP-only (1, 12, 15), vs classical-decoder IABR-Net (1, 0, 27); L!=3 test banks (23): vs U-Net (7, 2, 14), vs NOMP-only (0, 13, 10), vs classical decoder (1, 1, 21). The only bank where all-neural beats NOMP-only is gain_g4_snr15 (gain impairment 4 dB at 15 dB, outside the identifiable-impairment range NOMP handles; the gain comes from the CNN/IABC front end, as for the classical-decoder pipeline).
* Self-calibration matters for the learned decoder: SC off loses 0.005 Pd on the official bank and is below the classical decoder on 14/28 L=3 banks; SC on is never below it on L=3 test banks.
* **Latency (CPU, 1 thread, median)**: all-neural 101.4 ms per scene at batch 1 and 8.14 ms/scene at batch 256, vs NOMP 4.8 / 0.76 ms and the classical-decoder pipeline 12.3 / 3.88 ms. The learned decoder is the slowest component (about 50 small unrolled torch steps per call), so the all-neural pipeline is not a latency win either.
* Bottom line for a paper: the all-neural IABR-Net is a fair "fully learned" entry that clearly beats the two published deep-learning baselines on their own benchmark (U-Net +0.014, ResNet +0.040 paper-style Pd; much larger margins in strict Pd because both drop scenes), but a model-based NOMP with known L remains the strongest method on ideal hardware and must be listed as the reference.

## Pipeline and model facts

* **All-neural IABR-Net** = CNN `IABRv2(32,10,32)` from `final.pt` (191,702 parameters, including the 458-parameter IABC impairment head) on Y -> heat = sigmoid(heat logits) (32x32) and the IABC-corrected channel Hc -> top-L local maxima of the heatmap (notebook `heat_peaks`: 3x3 wrap-around maximum filter, grid cells, no sub-grid interpolation) as seeds -> learned decoder `StagedSC(RefinerV3(), pre=6, rounds=3, inner=2)` from `lurd.pt` (4,803 parameters). Total 196,505 parameters. No FFT/greedy search and no classical Newton step anywhere. Paths per scene = known L (as for every method here). Inputs are the cached CNN outputs `outputs/cache/net__<bank>.npz`; `s4_latency` confirms that a fresh CPU forward + decoder reproduces the cached-input predictions (256 scenes, max difference 0.000 deg).
* **What the learned decoder computes** (`learned_decoder/exp.py`, `exp3.py`, `exp5.py`). Atom a(u,v)[n,m] = exp(-j n u + j m v); for a residual r and a candidate (u,v) the analytic score is S = sum r e^{j n u - j m v}, f = |S|^2.
  * Learned part: one shared MLP `RefinerV3` (7 -> 64 -> 64 -> 2, ReLU; 4,802 weights) plus one scalar `lam` (self-calibration strength, learned value 1.0232). Output step (du, dv) = (pi/32) tanh(MLP(features)); the last layer is zero-initialised, so the step rule is learned from scratch.
  * Features fed to the MLP (all analytic, computed from r, u, v): the analytic gradient of f, (df/du)/f and (df/dv)/f; four probe second differences of the score surface, (f(u+d)+f(u-d)-2f)/f and (f(v+d)+f(v-d)-2f)/f for d = 0.02 and 0.08 rad; and the normalised coherence |S| / sqrt(256 ||r||^2). No Hessian and no Newton formula is given.
  * Fixed (not learned) structure around it: (i) 6 pre-steps refining every seed against the full Hc; (ii) 3 cyclic rounds over the L paths in which LS gains of all L atoms are re-solved (ridge 1e-6), the other paths' LS fits are subtracted (residual subtraction) and the MLP takes 2 steps on that path; (iii) with SC on, 2 self-calibration iterations: LS model of Hc from the current paths, SNR estimate gated at tau = 5 (hard gate at test time), 3 alternating rank-one LS fits of per-row/per-column complex distortions, projection onto the identifiable subspace (remove mean log-gain, mean phase, linear phase ramp), shrink by lam, apply to Hc, then 2 more cyclic rounds with the same MLP. The decoder therefore is an unrolled NOMP-style refinement whose step rule is learned; search (seed choice) is done by the CNN heatmap.
  * Training (earlier stage, `exp5.py`, not redone here): 1,500 steps, batch 64, seed 55, Adam 2e-3 cosine, deep-supervised permutation-matched angle loss, L = 3 only, CNN frozen, scenes from `iabr2_sim` with SNR -15..24 dB, phase <= 8 deg, gain <= 3 dB. Banks with L != 3 are therefore out of its training distribution and are reported separately.
* Reproduction check (`s1_reproduce.log`): the 2000-scene subsets of `results5.json` are reproduced exactly (official learned_no_sc 0.7437, learned_with_sc 0.7465; all diffs +0.0000), and the full-scale wrapper `ndcommon.learned_decode` gives bit-identical predictions to `exp.run_learned` on that subset.
* Other models: U-Net (Lloria et al., official weights, PyTorch port in `notebook_src.py`) 31,261,441 parameters (`outputs/logs/run_log.txt`); ResNet (Lloria et al., official weights, port in `s5_resnet.py`) 466,321 parameters; NOMP-only has none.

## Official bank (N=8000, L=3, 8 SNR groups x 1000; mean over SNR groups; 95% paired-bootstrap percentile CI, 2000 resamples within SNR groups, seed 0)

| method | Pd paper-style [95% CI] | Pd strict [95% CI] | RMSE deg [95% CI] | dropped |
|---|---|---|---|---|
| all-neural (SC on) | 0.7508 [0.7468, 0.7545] | 0.7508 [0.7468, 0.7545] | 0.3324 [0.3292, 0.3356] | 0 |
| all-neural (SC off) | 0.7457 [0.7418, 0.7496] | 0.7457 [0.7418, 0.7496] | 0.3417 [0.3384, 0.3446] | 0 |
| IABR-Net classical heat decoder | 0.7503 [0.7463, 0.7540] | 0.7503 [0.7463, 0.7540] | 0.3329 [0.3297, 0.3361] | 0 |
| IABR-v2 dual | 0.7551 [0.7513, 0.7587] | 0.7551 [0.7513, 0.7587] | 0.3337 [0.3304, 0.3368] | 0 |
| NOMP-only | 0.7581 [0.7544, 0.7618] | 0.7581 [0.7544, 0.7618] | 0.3277 [0.3243, 0.3307] | 0 |
| U-Net | 0.7366 [0.7327, 0.7405] | 0.6940 [0.6890, 0.6991] | 0.3611 [0.3577, 0.3644] | 538 |
| ResNet (Lloria et al., port) | 0.7103 [0.7063, 0.7146] | 0.6728 [0.6677, 0.6779] | 0.3762 [0.3727, 0.3796] | 481 |
| network only | 0.4507 [0.4460, 0.4551] | 0.4507 [0.4460, 0.4551] | 0.5660 [0.5625, 0.5696] | 0 |

| paired difference | d Pd paper-style | d Pd strict | d RMSE deg |
|---|---|---|---|
| all-neural (SC on) - U-Net | +0.0142 [+0.0108, +0.0176] | +0.0568 [+0.0530, +0.0605] | -0.0286 [-0.0324, -0.0249] |
| all-neural (SC on) - NOMP-only | -0.0073 [-0.0091, -0.0056] | -0.0073 [-0.0091, -0.0056] | +0.0048 [+0.0027, +0.0070] |
| all-neural (SC on) - IABR-Net classical heat decoder | +0.0005 [-0.0002, +0.0011] | +0.0005 [-0.0002, +0.0011] | -0.0005 [-0.0016, +0.0007] |
| all-neural (SC off) - U-Net | +0.0091 [+0.0056, +0.0125] | +0.0517 [+0.0479, +0.0554] | -0.0194 [-0.0232, -0.0156] |
| all-neural (SC off) - NOMP-only | -0.0124 [-0.0142, -0.0106] | -0.0124 [-0.0142, -0.0106] | +0.0140 [+0.0116, +0.0163] |
| all-neural (SC off) - IABR-Net classical heat decoder | -0.0046 [-0.0059, -0.0033] | -0.0046 [-0.0059, -0.0033] | +0.0088 [+0.0066, +0.0109] |
| all-neural (SC on) - all-neural (SC off) | +0.0051 [+0.0038, +0.0062] | +0.0051 [+0.0038, +0.0062] | -0.0092 [-0.0111, -0.0074] |
| all-neural (SC on) - ResNet (Lloria et al., port) | +0.0404 [+0.0369, +0.0439] | +0.0780 [+0.0737, +0.0821] | -0.0438 [-0.0479, -0.0399] |

"dropped" = scenes with fewer than L detections (excluded by the paper-style Pd, counted as 2L misses by the strict Pd). Methods that always output L paths have paper-style = strict.

### Official bank per SNR (paper-style Pd; per-SNR CIs from a paired bootstrap inside that SNR group)

| SNR dB | all-neural (SC on) | all-neural (SC off) | IABR-Net classical heat decoder | IABR-v2 dual | NOMP-only | U-Net | ResNet (Lloria et al., port) | network only | all-neural (SC on) - NOMP [95% CI] | all-neural (SC on) - U-Net [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| -10 | 0.2613 | 0.2613 | 0.2602 | 0.2630 | 0.2618 | 0.2225 (0.2027) | 0.2040 | 0.2517 | -0.0005 [-0.0062, +0.0050] | +0.0389 [+0.0272, +0.0511] |
| -5 | 0.5038 | 0.5032 | 0.5042 | 0.5067 | 0.5063 | 0.4708 (0.4143) | 0.4366 | 0.4548 | -0.0025 [-0.0070, +0.0020] | +0.0330 [+0.0213, +0.0446] |
| 0 | 0.6858 | 0.6870 | 0.6855 | 0.6915 | 0.6907 | 0.6788 (0.5967) | 0.6393 | 0.5537 | -0.0048 [-0.0098, -0.0002] | +0.0070 [-0.0046, +0.0184] |
| 5 | 0.8153 | 0.8137 | 0.8142 | 0.8187 | 0.8198 | 0.8129 (0.7478) | 0.7789 | 0.5407 | -0.0045 [-0.0088, -0.0002] | +0.0025 [-0.0071, +0.0126] |
| 10 | 0.8882 | 0.8837 | 0.8878 | 0.8955 | 0.8993 | 0.8885 (0.8325) | 0.8623 | 0.5027 | -0.0112 [-0.0167, -0.0057] | -0.0003 [-0.0097, +0.0086] |
| 15 | 0.9360 | 0.9285 | 0.9355 | 0.9390 | 0.9448 | 0.9244 (0.9032) | 0.8989 | 0.4508 | -0.0088 [-0.0132, -0.0043] | +0.0116 [+0.0044, +0.0186] |
| 20 | 0.9532 | 0.9407 | 0.9512 | 0.9582 | 0.9662 | 0.9440 (0.9280) | 0.9253 | 0.4278 | -0.0130 [-0.0175, -0.0088] | +0.0091 [+0.0024, +0.0160] |
| 25 | 0.9625 | 0.9477 | 0.9640 | 0.9687 | 0.9758 | 0.9508 (0.9270) | 0.9376 | 0.4235 | -0.0133 [-0.0183, -0.0087] | +0.0117 [+0.0044, +0.0193] |

Figure: `fig_pd_vs_snr.png` (left: Pd vs SNR for every method; right: paired differences all-neural minus NOMP-only / U-Net / classical-decoder IABR-Net with 95% CIs).

The all-neural pipeline is tied with NOMP below 0 dB and loses 0.004-0.013 Pd from 0 dB upward (CIs exclude 0 at every SNR >= 0 dB); it beats U-Net mainly at low SNR (-10/-5 dB) and at 15-25 dB.

## Table II banks (phase errors delta_max = 1, 2, 5 deg; N=8000 each)

| bank | all-neural SC on | SC off | classical-decoder IABR-Net | IABR-v2 dual | NOMP-only | U-Net (strict) | ResNet port (strict) | d vs U-Net | d vs NOMP-only | d vs classical decoder |
|---|---|---|---|---|---|---|---|---|---|---|
| official | 0.7508 | 0.7457 | 0.7503 | 0.7551 | 0.7581 | 0.7366 (0.6940) | 0.7103 (0.6728) | +0.0142 [+0.0108, +0.0176] | -0.0073 [-0.0091, -0.0056] | +0.0005 [-0.0002, +0.0011] |
| T2_d1 | 0.7500 | 0.7445 | 0.7493 | 0.7543 | 0.7573 | 0.7359 (0.6979) | 0.7115 (0.6763) | +0.0141 [+0.0108, +0.0174] | -0.0073 [-0.0089, -0.0056] | +0.0007 [-0.0000, +0.0015] |
| T2_d2 | 0.7430 | 0.7372 | 0.7422 | 0.7482 | 0.7515 | 0.7335 (0.6895) | 0.7105 (0.6710) | +0.0095 [+0.0061, +0.0130] | -0.0085 [-0.0102, -0.0069] | +0.0008 [-0.0000, +0.0015] |
| T2_d5 | 0.7417 | 0.7363 | 0.7412 | 0.7452 | 0.7468 | 0.7277 (0.6946) | 0.7020 (0.6720) | +0.0140 [+0.0107, +0.0172] | -0.0051 [-0.0068, -0.0034] | +0.0004 [-0.0003, +0.0012] |

Full per-method CI tables and per-SNR tables for T2_d1/T2_d2/T2_d5: `tables.md`.

## Comparison with published work (same benchmark: the authors' official test set, L=3, 16x16, SNR -10..25 dB, 1000 scenes per SNR)

| method | type | parameters | Pd paper-style (mean over SNR) [95% CI] | Pd strict | RMSE deg | Pd @ 0 dB | Pd @ 20 dB | source of the numbers |
|---|---|---|---|---|---|---|---|---|
| ResNet (Lloria et al., port) | DL, image-to-image + blob detection (Lloria et al., TVT 2026 / earlier ResNet paper) | 466,321 | 0.7103 [0.7063, 0.7146] | 0.6728 | 0.3762 | 0.6393 | 0.9253 | ResNet port of official weights, validated against the authors' repo results (max |d Pd| per SNR 0.0004) |
| U-Net | DL, image-to-image + blob detection (Lloria et al., TVT 2026) | 31,261,441 | 0.7366 [0.7327, 0.7405] | 0.6940 | 0.3611 | 0.6788 | 0.9440 | U-Net port of official weights, validated in notebook check V_unet_port (max |d Pd| 0.0002) |
| all-neural (SC on) | DL: CNN heatmap + IABC + learned unrolled decoder (this work, all-neural) | 196,505 | 0.7508 [0.7468, 0.7545] | 0.7508 | 0.3324 | 0.6858 | 0.9532 | s2/s3 here |
| IABR-Net classical heat decoder | CNN heatmap + IABC + classical Newton/cyclic/self-cal decoder (this work) | 191,702 | 0.7503 [0.7463, 0.7540] | 0.7503 | 0.3329 | 0.6855 | 0.9512 | new_results/preds_heat |
| IABR-v2 dual | CNN + IABC + dual-seeded classical decoder (this work, v2) | 191,702 | 0.7551 [0.7513, 0.7587] | 0.7551 | 0.3337 | 0.6915 | 0.9582 | pred__IABR-v2 cache |
| NOMP-only | model-based reference: NOMP greedy + Newton + cyclic refinement, known L, no network | 0 | 0.7581 [0.7544, 0.7618] | 0.7581 | 0.3277 | 0.6907 | 0.9662 | pred__NOMP_no_network (independently re-implemented in verify/) |

Paper-printed values for reference (TVT 2026, Table II, delta_max = 0, official test set): ResNet Pd 0.643 / 0.925 and RMSE 0.458 / 0.253 at 0 / 20 dB; U-Net Pd 0.679 / 0.945, RMSE 0.454 / 0.230. Our ports give ResNet 0.6393 / 0.9253 (RMSE 0.4578 / 0.2534) and U-Net 0.6788 / 0.9440. The paper's Pd is the paper-style Pd (dropped scenes excluded). The paper also lists TSDCE and DFT-CEA (Pd 0.524 / 0.885 and 0.622 / 0.891 at 0 / 20 dB, Table II); those were not re-implemented here.

### ResNet port validation (`s5_resnet.log`, `s5_resnet_validation.json`)

* Architecture from `DL_DOA/src/tvt_models.py` (`Resnet`): Conv2DTranspose(12, 5x5, stride 2, same) -> 64 residual blocks (two 5x5 convs with 12 filters + BatchNorm, ReLU, identity skip) -> Conv2DTranspose(1, 5x5, stride 2, same); weights read with h5py from `inf_model_007_256_resnet.h5` (128 conv, 128 BN, 2 transposed-conv layers, matched in creation order). Keras "same" transposed convolution was checked against TensorFlow's definition (adjoint of a SAME stride-2 conv, implemented with autograd): exact (0.00e+00); the naive PyTorch padding would differ by 7.98.
* Post-processing identical to the authors' `run_inference_and_metrics_resnet` (OpenCV blob detector, top-L by amplitude, `peaks_to_angles(sigma=0.07, M=256)`), input `upsample64(Y16)` exactly as for the cached U-Net predictions.

| SNR dB | Pd published (authors' repo `resnet_pd.pkl`) | Pd our port | RMSE published | RMSE our port |
|---|---|---|---|---|
| -10 | 0.2043 | 0.2040 | 0.5532 | 0.5526 |
| -5 | 0.4366 | 0.4366 | 0.5118 | 0.5117 |
| 0 | 0.6396 | 0.6393 | 0.4581 | 0.4578 |
| 5 | 0.7790 | 0.7789 | 0.3920 | 0.3918 |
| 10 | 0.8623 | 0.8623 | 0.3256 | 0.3255 |
| 15 | 0.8987 | 0.8989 | 0.2792 | 0.2794 |
| 20 | 0.9250 | 0.9253 | 0.2528 | 0.2534 |
| 25 | 0.9378 | 0.9376 | 0.2377 | 0.2373 |

max |d Pd| = 0.0004 -> PASS (same 0.01 threshold as the U-Net port check). Against the paper's Table II rows with phase errors (our regenerated T2 banks are new random draws of the same impairment model, so agreement is expected only up to sampling noise):

| delta_max | ResNet Pd 0 dB paper / ours | ResNet Pd 20 dB paper / ours | U-Net Pd 0 dB paper / ours | U-Net Pd 20 dB paper / ours |
|---|---|---|---|---|
| 1 deg | 0.643 / 0.6350 | 0.923 / 0.9326 | 0.678 / 0.6765 | 0.945 / 0.9518 |
| 2 deg | 0.642 / 0.6392 | 0.921 / 0.9205 | 0.677 / 0.6816 | 0.941 / 0.9389 |
| 5 deg | 0.638 / 0.6452 | 0.909 / 0.9054 | 0.677 / 0.6795 | 0.925 / 0.9235 |

## All banks (paper-style Pd, mean over SNR groups; d = paired difference of all-neural SC on minus reference with 95% CI)

| bank | N | L | all-neural SC on | SC off | classical decoder | IABR-v2 dual | NOMP-only | U-Net | network only | d vs U-Net | d vs NOMP-only | d vs classical decoder |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **L=3 test banks** | | | | | | | | | | | | |
| official | 8000 | 3 | 0.7508 | 0.7457 | 0.7503 | 0.7551 | 0.7581 | 0.7366 | 0.4507 | +0.0142 [+0.0108, +0.0176] | -0.0073 [-0.0091, -0.0056] | +0.0005 [-0.0002, +0.0011] |
| T2_d1 | 8000 | 3 | 0.7500 | 0.7445 | 0.7493 | 0.7543 | 0.7573 | 0.7359 | - | +0.0141 [+0.0108, +0.0174] | -0.0073 [-0.0089, -0.0056] | +0.0007 [-0.0000, +0.0015] |
| T2_d2 | 8000 | 3 | 0.7430 | 0.7372 | 0.7422 | 0.7482 | 0.7515 | 0.7335 | - | +0.0095 [+0.0061, +0.0130] | -0.0085 [-0.0102, -0.0069] | +0.0008 [-0.0000, +0.0015] |
| T2_d5 | 8000 | 3 | 0.7417 | 0.7363 | 0.7412 | 0.7452 | 0.7468 | 0.7277 | 0.4478 | +0.0140 [+0.0107, +0.0172] | -0.0051 [-0.0068, -0.0034] | +0.0004 [-0.0003, +0.0012] |
| gain_g0.5_snr0 | 500 | 3 | 0.7017 | 0.6990 | 0.7020 | 0.7070 | 0.7047 | 0.6794 | - | +0.0222 [+0.0060, +0.0374] | -0.0030 [-0.0100, +0.0037] | -0.0003 [-0.0030, +0.0020] |
| gain_g0.5_snr15 | 500 | 3 | 0.9300 | 0.9173 | 0.9307 | 0.9350 | 0.9393 | 0.9199 | - | +0.0101 [-0.0002, +0.0199] | -0.0093 [-0.0160, -0.0027] | -0.0007 [-0.0033, +0.0020] |
| gain_g1_snr0 | 500 | 3 | 0.7033 | 0.7030 | 0.7027 | 0.7080 | 0.7083 | 0.6982 | - | +0.0051 [-0.0098, +0.0213] | -0.0050 [-0.0120, +0.0013] | +0.0007 [-0.0020, +0.0040] |
| gain_g1_snr15 | 500 | 3 | 0.9297 | 0.9247 | 0.9297 | 0.9340 | 0.9350 | 0.9220 | - | +0.0077 [-0.0029, +0.0178] | -0.0053 [-0.0127, +0.0017] | +0.0000 [-0.0027, +0.0027] |
| gain_g2_snr0 | 500 | 3 | 0.7103 | 0.7117 | 0.7103 | 0.7147 | 0.7143 | 0.6819 | 0.5490 | +0.0284 [+0.0118, +0.0447] | -0.0040 [-0.0113, +0.0033] | +0.0000 [-0.0023, +0.0020] |
| gain_g2_snr15 | 500 | 3 | 0.9200 | 0.9043 | 0.9173 | 0.9217 | 0.9167 | 0.8855 | - | +0.0345 [+0.0237, +0.0459] | +0.0033 [-0.0053, +0.0117] | +0.0027 [+0.0000, +0.0060] |
| gain_g4_snr0 | 500 | 3 | 0.6850 | 0.6853 | 0.6843 | 0.6927 | 0.6893 | 0.6688 | - | +0.0162 [+0.0004, +0.0321] | -0.0043 [-0.0127, +0.0037] | +0.0007 [-0.0017, +0.0033] |
| gain_g4_snr15 | 500 | 3 | 0.9100 | 0.8847 | 0.9110 | 0.9123 | 0.8697 | 0.8126 | 0.4780 | +0.0974 [+0.0817, +0.1130] | +0.0403 [+0.0293, +0.0513] | -0.0010 [-0.0043, +0.0023] |
| oodphase_d10_snr0 | 500 | 3 | 0.6907 | 0.6907 | 0.6910 | 0.6957 | 0.6947 | 0.6936 | - | -0.0029 [-0.0195, +0.0138] | -0.0040 [-0.0110, +0.0030] | -0.0003 [-0.0030, +0.0027] |
| oodphase_d10_snr15 | 500 | 3 | 0.9093 | 0.8990 | 0.9117 | 0.9140 | 0.9127 | 0.8888 | - | +0.0206 [+0.0088, +0.0315] | -0.0033 [-0.0113, +0.0040] | -0.0023 [-0.0063, +0.0013] |
| oodphase_d15_snr0 | 500 | 3 | 0.6717 | 0.6703 | 0.6683 | 0.6763 | 0.6750 | 0.6659 | - | +0.0057 [-0.0098, +0.0214] | -0.0033 [-0.0097, +0.0027] | +0.0033 [+0.0000, +0.0073] |
| oodphase_d15_snr15 | 500 | 3 | 0.8653 | 0.8523 | 0.8667 | 0.8693 | 0.8633 | 0.8454 | 0.4917 | +0.0200 [+0.0076, +0.0327] | +0.0020 [-0.0070, +0.0113] | -0.0013 [-0.0037, +0.0010] |
| phase_d0_snr0 | 500 | 3 | 0.6807 | 0.6800 | 0.6803 | 0.6837 | 0.6817 | 0.6784 | 0.5487 | +0.0023 [-0.0150, +0.0198] | -0.0010 [-0.0077, +0.0050] | +0.0003 [-0.0027, +0.0040] |
| phase_d0_snr15 | 500 | 3 | 0.9383 | 0.9290 | 0.9370 | 0.9413 | 0.9497 | 0.9344 | 0.4370 | +0.0040 [-0.0071, +0.0150] | -0.0113 [-0.0163, -0.0067] | +0.0013 [-0.0013, +0.0043] |
| phase_d1_snr0 | 500 | 3 | 0.6847 | 0.6817 | 0.6840 | 0.6870 | 0.6900 | 0.6774 | - | +0.0073 [-0.0097, +0.0242] | -0.0053 [-0.0110, +0.0007] | +0.0007 [-0.0017, +0.0033] |
| phase_d1_snr15 | 500 | 3 | 0.9263 | 0.9197 | 0.9257 | 0.9353 | 0.9417 | 0.9223 | - | +0.0040 [-0.0081, +0.0163] | -0.0153 [-0.0227, -0.0083] | +0.0007 [-0.0020, +0.0033] |
| phase_d2_snr0 | 500 | 3 | 0.6983 | 0.6977 | 0.6983 | 0.7033 | 0.7077 | 0.6774 | - | +0.0209 [+0.0063, +0.0364] | -0.0093 [-0.0157, -0.0037] | +0.0000 [-0.0023, +0.0027] |
| phase_d2_snr15 | 500 | 3 | 0.9283 | 0.9160 | 0.9270 | 0.9310 | 0.9393 | 0.9217 | - | +0.0067 [-0.0044, +0.0180] | -0.0110 [-0.0167, -0.0057] | +0.0013 [-0.0003, +0.0033] |
| phase_d5_snr0 | 500 | 3 | 0.7107 | 0.7107 | 0.7100 | 0.7140 | 0.7153 | 0.6948 | - | +0.0159 [-0.0005, +0.0327] | -0.0047 [-0.0113, +0.0013] | +0.0007 [-0.0017, +0.0037] |
| phase_d5_snr15 | 500 | 3 | 0.9177 | 0.9110 | 0.9143 | 0.9220 | 0.9330 | 0.9119 | - | +0.0057 [-0.0055, +0.0168] | -0.0153 [-0.0227, -0.0077] | +0.0033 [+0.0007, +0.0063] |
| tail_snr-20 | 500 | 3 | 0.0313 | 0.0317 | 0.0310 | 0.0317 | 0.0303 | 0.0240 | - | +0.0073 [-0.0001, +0.0149] | +0.0010 [-0.0033, +0.0053] | +0.0003 [-0.0007, +0.0013] |
| tail_snr-25 | 500 | 3 | 0.0207 | 0.0220 | 0.0207 | 0.0193 | 0.0200 | 0.0247 | - | -0.0040 [-0.0110, +0.0030] | +0.0007 [-0.0037, +0.0050] | +0.0000 [-0.0010, +0.0010] |
| tail_snr30 | 500 | 3 | 0.9693 | 0.9537 | 0.9680 | 0.9743 | 0.9813 | 0.9582 | - | +0.0111 [+0.0007, +0.0205] | -0.0120 [-0.0173, -0.0070] | +0.0013 [-0.0020, +0.0057] |
| tail_snr35 | 500 | 3 | 0.9783 | 0.9527 | 0.9760 | 0.9833 | 0.9893 | 0.9643 | - | +0.0140 [+0.0064, +0.0218] | -0.0110 [-0.0173, -0.0047] | +0.0023 [-0.0003, +0.0053] |
| **L != 3 test banks (decoder trained with L=3 only)** | | | | | | | | | | | | |
| L10_snr0 | 500 | 10 | 0.4493 | 0.4481 | 0.4477 | 0.4566 | 0.4551 | 0.4210 | - | +0.0283 [+0.0122, +0.0443] | -0.0058 [-0.0123, +0.0005] | +0.0016 [-0.0003, +0.0036] |
| L10_snr15 | 500 | 10 | 0.8029 | 0.8017 | 0.8022 | 0.8378 | 0.8423 | 0.7941 | - | +0.0088 [-0.0022, +0.0192] | -0.0394 [-0.0471, -0.0321] | +0.0007 [-0.0021, +0.0037] |
| L1_snr0 | 500 | 1 | 0.8340 | 0.8340 | 0.8330 | 0.8320 | 0.8310 | 0.8020 | - | +0.0320 [+0.0160, +0.0480] | +0.0030 [-0.0020, +0.0080] | +0.0010 [-0.0040, +0.0050] |
| L1_snr15 | 500 | 1 | 0.9640 | 0.9640 | 0.9640 | 0.9640 | 0.9630 | 0.9550 | - | +0.0090 [-0.0010, +0.0180] | +0.0010 [-0.0030, +0.0060] | +0.0000 [-0.0040, +0.0040] |
| L2_snr0 | 500 | 2 | 0.7495 | 0.7475 | 0.7490 | 0.7530 | 0.7550 | 0.7361 | - | +0.0134 [-0.0041, +0.0306] | -0.0055 [-0.0125, +0.0005] | +0.0005 [-0.0015, +0.0025] |
| L2_snr15 | 500 | 2 | 0.9525 | 0.9455 | 0.9510 | 0.9535 | 0.9590 | 0.9479 | - | +0.0046 [-0.0052, +0.0143] | -0.0065 [-0.0120, -0.0015] | +0.0015 [-0.0015, +0.0050] |
| L4_snr0 | 500 | 4 | 0.6435 | 0.6395 | 0.6422 | 0.6508 | 0.6482 | 0.6250 | - | +0.0185 [+0.0022, +0.0341] | -0.0048 [-0.0115, +0.0018] | +0.0012 [-0.0015, +0.0043] |
| L4_snr15 | 500 | 4 | 0.9163 | 0.9107 | 0.9165 | 0.9255 | 0.9290 | 0.9106 | - | +0.0056 [-0.0043, +0.0155] | -0.0128 [-0.0195, -0.0060] | -0.0002 [-0.0035, +0.0025] |
| L5_snr0 | 500 | 5 | 0.6034 | 0.6020 | 0.6026 | 0.6054 | 0.6074 | 0.5841 | - | +0.0193 [+0.0053, +0.0331] | -0.0040 [-0.0106, +0.0022] | +0.0008 [-0.0022, +0.0038] |
| L5_snr15 | 500 | 5 | 0.9008 | 0.8934 | 0.9000 | 0.9134 | 0.9214 | 0.8937 | - | +0.0071 [-0.0045, +0.0181] | -0.0206 [-0.0288, -0.0126] | +0.0008 [-0.0020, +0.0040] |
| L6_snr0 | 500 | 6 | 0.5592 | 0.5592 | 0.5583 | 0.5652 | 0.5667 | 0.5459 | - | +0.0132 [-0.0017, +0.0282] | -0.0075 [-0.0142, -0.0010] | +0.0008 [-0.0022, +0.0038] |
| L6_snr15 | 500 | 6 | 0.8750 | 0.8665 | 0.8768 | 0.8945 | 0.8960 | 0.8598 | - | +0.0152 [+0.0036, +0.0266] | -0.0210 [-0.0292, -0.0135] | -0.0018 [-0.0052, +0.0013] |
| L7_snr0 | 500 | 7 | 0.5369 | 0.5371 | 0.5367 | 0.5441 | 0.5423 | 0.5151 | - | +0.0217 [+0.0088, +0.0347] | -0.0054 [-0.0117, +0.0007] | +0.0001 [-0.0020, +0.0023] |
| L7_snr15 | 500 | 7 | 0.8494 | 0.8411 | 0.8489 | 0.8716 | 0.8797 | 0.8407 | - | +0.0088 [-0.0039, +0.0209] | -0.0303 [-0.0390, -0.0223] | +0.0006 [-0.0019, +0.0030] |
| L8_snr0 | 500 | 8 | 0.4966 | 0.4964 | 0.4983 | 0.5028 | 0.5019 | 0.4746 | 0.4422 | +0.0220 [+0.0083, +0.0364] | -0.0052 [-0.0111, +0.0006] | -0.0016 [-0.0042, +0.0009] |
| L8_snr15 | 500 | 8 | 0.8349 | 0.8275 | 0.8343 | 0.8628 | 0.8651 | 0.8230 | 0.4689 | +0.0119 [-0.0002, +0.0232] | -0.0302 [-0.0383, -0.0224] | +0.0006 [-0.0026, +0.0045] |
| sep_10deg_snr15 | 500 | 2 | 0.9815 | 0.9775 | 0.9820 | 0.9955 | 0.9955 | 0.9725 | - | +0.0090 [-0.0034, +0.0208] | -0.0140 [-0.0240, -0.0055] | -0.0005 [-0.0030, +0.0020] |
| sep_1deg_snr15 | 500 | 2 | 0.4645 | 0.4650 | 0.4700 | 0.4795 | 0.4990 | 0.3214 | - | +0.1431 [-0.0492, +0.3162] | -0.0345 [-0.0450, -0.0245] | -0.0055 [-0.0095, -0.0020] |
| sep_20deg_snr15 | 500 | 2 | 0.9960 | 0.9955 | 0.9960 | 0.9960 | 0.9950 | 0.9955 | - | +0.0005 [-0.0040, +0.0051] | +0.0010 [+0.0000, +0.0025] | +0.0000 [+0.0000, +0.0000] |
| sep_2deg_snr15 | 500 | 2 | 0.3400 | 0.3375 | 0.3415 | 0.4550 | 0.5880 | 0.2917 | 0.2885 | +0.0483 [-0.0275, +0.1301] | -0.2480 [-0.2760, -0.2205] | -0.0015 [-0.0095, +0.0060] |
| sep_30deg_snr15 | 500 | 2 | 0.9955 | 0.9950 | 0.9945 | 0.9955 | 0.9950 | 0.9960 | - | -0.0005 [-0.0035, +0.0021] | +0.0005 [+0.0000, +0.0015] | +0.0010 [+0.0000, +0.0030] |
| sep_3deg_snr15 | 500 | 2 | 0.3070 | 0.2835 | 0.2990 | 0.5485 | 0.7115 | 0.5426 | - | -0.2356 [-0.3066, -0.1651] | -0.4045 [-0.4310, -0.3775] | +0.0080 [-0.0035, +0.0195] |
| sep_5deg_snr15 | 500 | 2 | 0.5520 | 0.4995 | 0.5280 | 0.8595 | 0.8970 | 0.8693 | 0.4435 | -0.3173 [-0.3554, -0.2785] | -0.3450 [-0.3760, -0.3140] | +0.0240 [+0.0110, +0.0375] |
| **tuning banks (used to tune other stages; no baseline caches)** | | | | | | | | | | | | |
| tune_L6 | 300 | 6 | 0.7167 | 0.7072 | 0.7147 | - | - | - | - | - | - | +0.0019 [-0.0006, +0.0047] |
| tune_clean | 600 | 3 | 0.7922 | 0.7828 | 0.7922 | - | - | - | - | - | - | +0.0000 [-0.0025, +0.0025] |
| tune_d5 | 300 | 3 | 0.8106 | 0.8050 | 0.8117 | - | - | - | - | - | - | -0.0011 [-0.0050, +0.0028] |
| tune_g3 | 300 | 3 | 0.8067 | 0.7972 | 0.8050 | - | - | - | - | - | - | +0.0017 [-0.0022, +0.0072] |

## Bank counts (95% CI of the paired difference: above 0 / below 0 / containing 0)

| bank group | difference | Pd paper-style (>0 / <0 / contains 0) | Pd strict (>0 / <0 / contains 0) | banks with CI > 0 (paper-style) |
|---|---|---|---|---|
| L=3 test banks | all-neural (SC on) - U-Net | 14 / 0 / 14 | 26 / 0 / 2 | 14 banks |
| L=3 test banks | all-neural (SC on) - NOMP-only | 1 / 12 / 15 | 1 / 12 / 15 | gain_g4_snr15 |
| L=3 test banks | all-neural (SC on) - IABR-Net classical heat decoder | 1 / 0 / 27 | 1 / 0 / 27 | phase_d5_snr15 |
| L=3 test banks | all-neural (SC off) - U-Net | 10 / 1 / 17 | 24 / 0 / 4 | 10 banks |
| L=3 test banks | all-neural (SC off) - NOMP-only | 1 / 17 / 10 | 1 / 17 / 10 | gain_g4_snr15 |
| L=3 test banks | all-neural (SC off) - IABR-Net classical heat decoder | 0 / 14 / 14 | 0 / 14 / 14 |  |
| L=3 test banks | all-neural (SC on) - all-neural (SC off) | 15 / 1 / 12 | 15 / 1 / 12 | 15 banks |
| L=3 test banks | all-neural (SC on) - ResNet (Lloria et al., port) | 4 / 0 / 0 | 4 / 0 / 0 | official, T2_d1, T2_d2, T2_d5 |
| L!=3 test banks | all-neural (SC on) - U-Net | 7 / 2 / 14 | 21 / 0 / 2 | 7 banks |
| L!=3 test banks | all-neural (SC on) - NOMP-only | 0 / 13 / 10 | 0 / 13 / 10 |  |
| L!=3 test banks | all-neural (SC on) - IABR-Net classical heat decoder | 1 / 1 / 21 | 1 / 1 / 21 | sep_5deg_snr15 |
| L!=3 test banks | all-neural (SC off) - U-Net | 5 / 2 / 16 | 20 / 1 / 2 | L10_snr0, L1_snr0, L5_snr0, L7_snr0, L8_snr0 |
| L!=3 test banks | all-neural (SC off) - NOMP-only | 0 / 16 / 7 | 0 / 16 / 7 |  |
| L!=3 test banks | all-neural (SC off) - IABR-Net classical heat decoder | 0 / 8 / 15 | 0 / 8 / 15 |  |
| L!=3 test banks | all-neural (SC on) - all-neural (SC off) | 9 / 0 / 14 | 9 / 0 / 14 | 9 banks |
| tuning banks (used to tune other stages) | all-neural (SC on) - IABR-Net classical heat decoder | 0 / 0 / 4 | 0 / 0 / 4 |  |
| tuning banks (used to tune other stages) | all-neural (SC off) - IABR-Net classical heat decoder | 0 / 4 / 0 | 0 / 4 / 0 |  |
| tuning banks (used to tune other stages) | all-neural (SC on) - all-neural (SC off) | 4 / 0 / 0 | 4 / 0 / 0 | tune_L6, tune_clean, tune_d5, tune_g3 |

Notes: L=3 test banks = official, T2_d1/2/5, gain_*, oodphase_*, phase_*, tail_* (28); L != 3 = L1..L10 and sep_* (23). A bank counts as "contains 0" when the CI touches 0. Where all-neural is below NOMP-only on L != 3 banks the deficit is largest at 15 dB with many paths (L5-L10: -0.02 to -0.04) and for closely spaced pairs (sep_2deg/3deg/5deg: -0.25 to -0.40, a limitation of 32x32 heatmap seeding shared with the classical heat decoder).

## Scene-level agreement with the classical heat-seeded decoder (`s6_agreement.log`)

| bank | SC on: scenes with max angle diff < 0.01 / 0.1 / 1 deg | SC off: same | scenes all paths ok: both / only all-neural / only classical / neither |
|---|---|---|---|
| official | 0.088 / 0.855 / 0.976 | 0.044 / 0.575 / 0.919 | 3463 / 36 / 45 / 4456 |
| T2_d1 | 0.088 / 0.863 / 0.977 | 0.042 / 0.580 / 0.916 | 3498 / 46 / 37 / 4419 |
| T2_d2 | 0.094 / 0.862 / 0.976 | 0.047 / 0.578 / 0.913 | 3364 / 43 / 36 / 4557 |
| T2_d5 | 0.099 / 0.861 / 0.976 | 0.045 / 0.558 / 0.911 | 3247 / 40 / 33 / 4680 |

## Latency (CPU, single thread; `s4_latency.log`)

Windows-11-10.0.26200-SP0, numpy 2.3.5, torch 2.7.1+cu118, torch threads 1; OMP/OPENBLAS/MKL/NUMEXPR = 1 before import. Official-bank scenes (L=3) drawn with np.random.default_rng(0) (same draw as verify/latency.py); 5 warm-up calls, then the median of 60 timed calls at batch 1 (60 distinct scenes) and 40 timed calls at batch 256 (4 distinct batches). No other heavy process was running.

| pipeline | batch 1: ms/scene (p10-p90) | batch 256: ms/call | batch 256: ms/scene |
|---|---|---|---|
| NOMP greedy, verify nomp_mine | 4.80 (4.62-5.08) | 194.7 | 0.760 |
| NOMP greedy, notebook nomp_greedy | 6.89 (6.47-7.57) | 378.7 | 1.479 |
| CNN forward only | 5.88 (5.60-6.48) | 550.6 | 2.151 |
| CNN + classical heat-seeded decoder (IABR-Net classical) | 12.27 (11.97-12.84) | 993.8 | 3.882 |
| CNN + learned decoder SC on (all-neural) | 101.41 (99.76-109.71) | 2083.7 | 8.139 |
| CNN + learned decoder SC off (all-neural) | 53.02 (51.77-60.57) | 1426.5 | 5.572 |
| learned decoder alone, SC on | 97.50 (93.96-126.56) | 1565.1 | 6.114 |

The absolute times are lower than in `verify/latency.log` (NOMP 8.5 ms, CNN+decoder 24.7 ms at batch 1), which was measured with the machine under other load; the ratios are what matters and both runs are internally consistent. The learned decoder costs ~95 ms at batch 1 because it is ~50 sequential small torch calls (pre-steps, 3 cyclic rounds x L paths x 2 steps, LS solves, 2 SC iterations with 2 more rounds), each with 5 score evaluations; it amortises well at batch 256 but is still slower than the classical decoder.

## Data, seeds and files

* Banks: all 52 banks with a `net__` cache and truth (`s0_inventory.log`; `dldoa_test` has no truth). No new data were generated; all decoders are deterministic. Bootstrap seed 0 (2000 resamples within SNR groups, same indices for all methods of a bank); latency scene draw `default_rng(0)`. No tuning was done in this stage (the learned decoder and lam come from `lurd.pt`; self-calibration constants are the decoder's own).
* Classical heat-seeded decoder predictions: `new_results/preds_heat/<bank>.npz` where present, `verify/cache/heat__<bank>.npz` for sep_2deg_snr15/phase_d0_*, otherwise recomputed by `s2_predict.py` with the heat_main recipe (`decode(Hc, L, heat, "heat", SC=(1.0, 5.0))`); the recipe was checked to reproduce `preds_heat/L8_snr0.npz` exactly (max |d| 0.00e+00 rad).
* Network-only predictions exist only for the banks listed with a value; U-Net, NOMP-only and IABR-v2 dual come from the notebook caches `pred__*` (verified in `verify/REPORT.md`).
* Large files (not on D:, regenerable from the scripts): predictions `C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder/<bank>__allneural_sc.npz`, `__allneural_nosc.npz`, `__heatcalc.npz`, `resnet__<bank>.npz` (+ `resnet_chunks/`), per-scene metric components `scene_stats/`.
* `results.json`: per bank, per method mean metrics (pd_paper, pd_strict, pd_source, rmse) with CIs, dropped counts, per-SNR metrics, all paired differences, per-SNR differences (8-SNR banks), source file of every method, and the bank counts.
* Logs from an earlier interrupted/partial pass are kept: `s3_stats_prelim_attempt1.log` (killed by a 590 s timeout), `s3_stats_prelim.log` (complete pass before the ResNet predictions existed; identical numbers for the other methods).

## Limitations

* Known L for every method (the published U-Net/ResNet also use known L for the top-L blobs). The learned decoder was trained only with L = 3; on L != 3 banks it still matches the classical decoder (21/23 CIs contain 0) but trails NOMP-only.
* All banks here are ideal-array or mild-impairment simulations; the stress results (position errors, low-resolution hardware, large impairments) are in `stress/` and were not re-run with the learned decoder.
* TSDCE and DFT-CEA from the published paper were not re-implemented; only their printed Table II values are quoted.
