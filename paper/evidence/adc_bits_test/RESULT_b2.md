# RESULT 2-bit ADC (peak AGC, L=3, ideal hardware)

Fresh scenes seeds 14000/14010/14020 (0/10/20 dB), 150 scenes per cell, same scenes for all methods. Pd = paper-style Pd (|err| <= 1 deg). Script: adc_bits.py, analyze.py.

## Per-SNR Pd (paper-style)

| method | 0 dB | 10 dB | 20 dB |
|---|---|---|---|
| NOMP | 0.387 | 0.343 | 0.258 |
| consistent NOMP | 0.422 | 0.529 | 0.362 |
| CNN read-out | 0.451 | 0.583 | 0.550 |
| ML known sigma | 0.478 | 0.661 | 0.711 |
| ML snr+5 | 0.440 | 0.578 | 0.633 |
| ML snr-5 | 0.428 | 0.586 | 0.638 |
| ML nominal 10dB | 0.356 | 0.661 | 0.507 |
| ML joint sigma | 0.441 | 0.618 | 0.652 |
| hybrid known sigma | 0.482 | 0.649 | 0.713 |
| hybrid snr+5 | 0.443 | 0.574 | 0.643 |
| hybrid snr-5 | 0.427 | 0.582 | 0.661 |
| hybrid nominal 10dB | 0.361 | 0.649 | 0.532 |
| hybrid joint sigma | 0.429 | 0.621 | 0.677 |

## Pooled unknown-SNR (all 450 scenes; paired bootstrap 95% CI, 2000 resamples, seed 0, stratified by SNR)

| method | Pd paper [95% CI] | Pd strict [95% CI] |
|---|---|---|
| NOMP | 0.329 [0.310,0.350] | 0.329 [0.310,0.350] |
| consistent NOMP | 0.438 [0.416,0.460] | 0.438 [0.416,0.460] |
| CNN read-out | 0.528 [0.507,0.548] | 0.528 [0.507,0.548] |
| ML known sigma | 0.617 [0.596,0.637] | 0.617 [0.596,0.637] |
| ML snr+5 | 0.550 [0.529,0.569] | 0.550 [0.529,0.569] |
| ML snr-5 | 0.550 [0.529,0.570] | 0.550 [0.529,0.570] |
| ML nominal 10dB | 0.508 [0.488,0.528] | 0.508 [0.488,0.528] |
| ML joint sigma | 0.570 [0.549,0.591] | 0.570 [0.549,0.591] |
| hybrid known sigma | 0.615 [0.593,0.636] | 0.615 [0.593,0.636] |
| hybrid snr+5 | 0.554 [0.533,0.573] | 0.554 [0.533,0.573] |
| hybrid snr-5 | 0.557 [0.537,0.576] | 0.557 [0.537,0.576] |
| hybrid nominal 10dB | 0.514 [0.494,0.534] | 0.514 [0.494,0.534] |
| hybrid joint sigma | 0.576 [0.555,0.596] | 0.576 [0.555,0.596] |

## Paired differences (pooled; * = CI excludes 0)

| a - b | paper | strict |
|---|---|---|
| CNN read-out - NOMP | +0.199 [+0.178,+0.220]* | +0.199 [+0.178,+0.220]* |
| hybrid known sigma - ML known sigma | -0.002 [-0.010,+0.006] | -0.002 [-0.010,+0.006] |
| hybrid nominal 10dB - ML known sigma | -0.103 [-0.121,-0.084]* | -0.103 [-0.121,-0.084]* |
| hybrid nominal 10dB - ML nominal 10dB | +0.006 [-0.001,+0.015] | +0.006 [-0.001,+0.015] |
| hybrid joint sigma - ML known sigma | -0.041 [-0.057,-0.025]* | -0.041 [-0.057,-0.025]* |
| hybrid joint sigma - ML joint sigma | +0.005 [-0.006,+0.018] | +0.005 [-0.006,+0.018] |
| hybrid snr+5 - ML snr+5 | +0.003 [-0.004,+0.011] | +0.003 [-0.004,+0.011] |
| hybrid snr-5 - ML snr-5 | +0.006 [-0.001,+0.014] | +0.006 [-0.001,+0.014] |
| ML snr+5 - ML known sigma | -0.066 [-0.081,-0.051]* | -0.066 [-0.081,-0.051]* |
| ML snr-5 - ML known sigma | -0.066 [-0.083,-0.050]* | -0.066 [-0.083,-0.050]* |
| ML nominal 10dB - ML known sigma | -0.109 [-0.126,-0.092]* | -0.109 [-0.126,-0.092]* |
| ML joint sigma - ML known sigma | -0.046 [-0.062,-0.031]* | -0.046 [-0.062,-0.031]* |
| CNN read-out - ML known sigma | -0.089 [-0.108,-0.068]* | -0.089 [-0.108,-0.068]* |
| CNN read-out - ML nominal 10dB | +0.020 [-0.000,+0.041] | +0.020 [-0.000,+0.041] |
| hybrid nominal 10dB - CNN read-out | -0.014 [-0.033,+0.005] | -0.014 [-0.033,+0.005] |

## Latency (seconds per scene, 1 thread, 12 scenes at 10 dB, batch of 12)

| item | s/scene |
|---|---|
| NOMP | 0.0021 |
| consistent NOMP | 0.0388 |
| CNN read-out | 0.0040 |
| ML starts only (13, no EM) | 0.0860 |
| ML starts only (13 + EM) | 0.1430 |
| ML known sigma LM only | 0.1617 |
| ML wrong/fixed sigma LM only (13 starts) | 0.1344 |
| hybrid LM only (CNN seeds) | 0.0152 |
| ML known sigma TOTAL | 0.3047 |
| ML wrong/fixed sigma TOTAL | 0.2204 |
| hybrid TOTAL (CNN + one LM) | 0.0192 |

## Reading (plain English)
- 2 bits: CNN read-out beats NOMP by +0.199* and is 0.089* below ML with known noise level, but is level with the SNR-blind fixed-10 dB ML (+0.020, CI [-0.000,+0.041], not significant). It needs no noise level; ML loses 0.046-0.109* without it.
- Hybrid with known sigma equals ML (-0.002, n.s.); under wrong/unknown sigma the hybrid is only +0.003 to +0.006 better than the matching ML (all n.s.). Hybrid with fixed 10 dB is 0.103* below ML with the true level.
- Strict Pd equals paper-style Pd (no dropped scenes).
