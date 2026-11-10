# Code audit of the original IABR-Net v2 pipeline (read-only; 2026-11-10)

Scope: physics and dataset generation, labels, network design, training setup, decoder, metric. Files audited: `IABR_v2_extracted/IABR_v2/notebook_src.py`, `iabr2_sim.py`, `dldoa_dataset_generation.py`, `TVT_Blob_Inference.py`. Checks were run on the official bank (8000 scenes) unless stated. "[code]" = read from source, "[run]" = measured. Audit scripts: scratchpad `audit/a1.py` to `a4.py` (not saved in the repo).

**Bottom line: no bug in the physics, labels, metric or decoder.** The network-only score of 0.4507 comes from two design choices (no sub-cell supervision, and a probability-domain parabola read-out). Network + decoder sits below NOMP because the IABC corrector perturbs ideal-hardware data and because NOMP is already close to a single-path reference at 5 dB and above.

## Findings

| # | Item | Severity | Evidence | Fix / impact |
|---|---|---|---|---|
| 1a | Steering vectors, channel, codebooks | OK | Same formulas as the benchmark generator; W, F unitary to 2e-15 [run] | none |
| 1b | Noise / SNR definition | OK | LS residual at true angles gives noise variance 10.01 / 1.000 / 0.00316 at -10 / 0 / 25 dB = 10^(-SNR/10) [run]. SNR is pre-beamforming; matched-filter SNR of a path = SNR + 24.1 dB + 10 log10(abs(alpha)^2) | Paths with matched-filter SNR below 13 dB: 90% / 52% / 21% / 7% at -10 / -5 / 0 / 5 dB (Monte Carlo): the low-SNR floor for every method |
| 1c | Path gains CN(0,1/L), strongest first | OK | LS powers at >=20 dB: 0.608 / 0.273 / 0.107 vs theory 0.611 / 0.278 / 0.111 | none |
| 1d | Minimum-separation rule | OK (benchmark quirk) | 30 deg Euclidean in the (AoD, AoA) plane, not in beam space; 2.6% of paths have another within 2 heat cells; 27.9% share a row/column with another | feeds finding 5b |
| 1e | Endfire | OK, inherent | 11.5% of angle components within 10 deg of endfire; u=+pi and -pi are the same atom but 180 deg apart in the metric | all methods affected equally |
| 1f | Impairment model | MINOR (realism) | Fresh random per-antenna errors every scene; real arrays are static | train/evaluate with errors fixed per device |
| 2a | Angle -> cell -> angle mapping | OK | No half-cell bias, no axis swap, wrap correct | none |
| 2b | Heat target has no sub-cell information | **MAJOR design** | Gaussian centred on the rounded cell, identical for any position within +-0.5 cell. A perfect copy of the target scores 0.3353 at G=32 (0.158 at 16, 0.663 at 64, 0.833 at 128, 0.912 at 256) | caps network-only |
| 2c | Cell size vs the 1-degree tolerance | **MAJOR design** | One 32-grid cell = 3.58 deg at broadside, 7.2 deg at 30 deg, 20.6 deg at 10 deg from endfire | read-out needs about 0.05 cell rms for Pd 0.91 |
| 3a | Circular padding | OK | DFT beam index is periodic in u | none |
| 3b | Receptive field / input scaling | OK | 21 3x3 convs, RF 43 > 16, SE global pooling; scaling input x37 changes heat by 2e-6 | none |
| 3c | No offset head; focal loss on rounded cell only | **MAJOR design** | no regression term | same as 2b |
| 3d | Batch norm train/eval | OK | CPU eval reproduces the cached heat to 3e-4 | none |
| 3e | lam_imp = 50 with IABC | MINOR | impairment loss flat at 4.2e-3 from step 16k to 48k: IABC learns almost nothing | wasted, not harmful |
| 3f | IABC on ideal hardware | MAJOR (known) | Changes Ht by 0.11% energy. NOMP on Ht 0.7581; on Hc 0.7548; dual 0.7551; dual without IABC 0.7580 | -0.003; bypass or gate IABC |
| 3g | Capacity / training length | MINOR, open | heat loss 0.527 at 16k, 0.520 at 24k, 0.515 at 48k: last 32k steps bought almost nothing | more steps will not help; more capacity untested |
| 4a | Train vs test distribution | MAJOR design (cost not measured) | L uniform 1..9, SNR -15..24, 80% impaired: only about 1.9% of training samples match the test condition (L=3, ideal, -10..24 dB) | needs a retrain to quantify |
| 4b | Seeds | MINOR | worker RNG mixes in os.urandom, so training is not reproducible; overlap with test banks impossible in practice | use a fixed seed sequence |
| 5a | Read-out parabola fitted to probabilities | **MAJOR, free fix** | integer cell 0.2956; parabola on probability 0.4507 (reproduces the reported figure); on log-probability 0.5421; on logit 0.4565; 3x3 centroid 0.4806 | +0.091 by changing one line |
| 5b | 3x3 NMS and sidelobe masking | MAJOR for detection | at >=15 dB the network misses 89 of 9000 paths, NOMP 13. Miss rate with another path within 2 cells: 33.6% vs NOMP 1.9% | caps detection near 0.99 |
| 5c | NOMP maths | OK | derivatives and sign conventions checked by hand | none |
| 6a | Metric | OK | 1-degree threshold in angle, Hungarian matching | none |
| 6b | Paper-style Pd favours the U-Net | MINOR | scenes with fewer than L detections dropped: U-Net 0.7366 paper vs 0.6940 strict; network and NOMP always return L | quote strict Pd alongside |

## Decomposition of the network-only score (0.4507)

| Step | Pd |
|---|---|
| Perfect localisation of every true path | 1.000 |
| Oracle sub-cell offset on peaks within 1 cell of a true path (detection only) | 0.8879 |
| Actual read-out (parabola on probability) | 0.4507 |

Detection loss 0.112; localisation loss 0.437. The problem is resolution, not missed peaks.

| SNR (dB) | -10 | -5 | 0 | 5 | 10 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|
| Network detection within 1 cell | 0.501 | 0.767 | 0.906 | 0.959 | 0.977 | 0.991 | 0.989 | 0.991 |
| NOMP detection within 1 cell | 0.510 | 0.776 | 0.920 | 0.968 | 0.990 | 0.998 | 0.998 | 0.999 |
| Network-only Pd | 0.252 | 0.455 | 0.554 | 0.541 | 0.503 | 0.451 | 0.428 | 0.424 |
| Network sub-cell error (rms cells, log fit) | 0.275 | 0.201 | 0.160 | 0.150 | 0.153 | 0.171 | 0.179 | 0.184 |
| NOMP sub-cell error (rms cells) | 0.272 | 0.187 | 0.134 | 0.095 | 0.063 | 0.043 | 0.038 | 0.028 |

Why network + decoder cannot exceed NOMP: paths found only by the network are 0.7% of all paths (3.2% at -10 dB, 0.0% at 20-25 dB); paths found only by NOMP are 1.7%. IABC accounts for the whole dual deficit (0.7551 with, 0.7580 without, NOMP 0.7581). From 5 dB upward NOMP is close to an interference-free single-path reference (computed CRB-Gaussian figure: 0.800 mean vs NOMP 0.758; not a valid bound at low SNR).

## Five most valuable network-side changes

1. Fit the parabola to log-probability (notebook_src.py lines 658-661): network-only 0.4507 -> 0.5421, measured, no retraining.
2. Supervise the sub-cell position: offset-regression head (loss weighted by 1/sin(theta)) or a continuous-centre Gaussian on a 64 or 128 grid (ceiling 0.663 / 0.833). Expected network-only about 0.6-0.7 if the head reaches about 0.1 cell rms (a projection, not a measurement).
3. Fix close-path and sidelobe misses: detect-subtract-redetect instead of 3x3 NMS, plus an oversampled beam-image input channel (gain at most about +0.01 network-only).
4. Bypass or gate IABC on clean inputs: +0.003 (dual 0.7580 from the existing cache).
5. Match training to the task: raise the share of L=3, clean, -10..25 dB samples from 1.9%; stop at about 16k steps. Effect unmeasured.

Realistic expectation: network-only can reach perhaps 0.6-0.7 and network + decoder can reach parity with NOMP (0.758). Nothing found suggests a network-side change that beats NOMP on ideal hardware by more than a few thousandths.

## Not verified

The cost of the train/test mismatch (4a) and whether a larger trunk breaks the 0.515 loss plateau (3g); the benchmark paper's own SNR definition (only checked against the local generator copy); the U-Net port origin; self-calibration and impaired-hardware banks; GPU nondeterminism. The Pd estimates for changes 2, 3 and 5 are projections from measured ceilings, not trained results.
