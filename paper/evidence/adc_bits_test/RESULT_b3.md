# RESULT 3-bit ADC (peak AGC, L=3, ideal hardware)

Fresh scenes seeds 14000/14010/14020 (0/10/20 dB), 150 scenes per cell, same scenes for all methods. Pd = paper-style Pd (|err| <= 1 deg). Script: adc_bits.py, analyze.py.

## Per-SNR Pd (paper-style)

| method | 0 dB | 10 dB | 20 dB |
|---|---|---|---|
| NOMP | 0.621 | 0.609 | 0.469 |
| consistent NOMP | 0.569 | 0.739 | 0.660 |
| CNN read-out | 0.556 | 0.591 | 0.618 |
| ML known sigma | 0.629 | 0.781 | 0.818 |
| ML snr+5 | 0.618 | 0.710 | 0.734 |
| ML snr-5 | 0.619 | 0.719 | 0.751 |
| ML nominal 10dB | 0.554 | 0.781 | 0.641 |
| ML joint sigma | 0.628 | 0.760 | 0.801 |
| hybrid known sigma | 0.626 | 0.771 | 0.821 |
| hybrid snr+5 | 0.607 | 0.713 | 0.744 |
| hybrid snr-5 | 0.617 | 0.716 | 0.757 |
| hybrid nominal 10dB | 0.557 | 0.771 | 0.658 |
| hybrid joint sigma | 0.621 | 0.752 | 0.801 |

## Pooled unknown-SNR (all 450 scenes; paired bootstrap 95% CI, 2000 resamples, seed 0, stratified by SNR)

| method | Pd paper [95% CI] | Pd strict [95% CI] |
|---|---|---|
| NOMP | 0.566 [0.543,0.588] | 0.566 [0.543,0.588] |
| consistent NOMP | 0.656 [0.635,0.676] | 0.656 [0.635,0.676] |
| CNN read-out | 0.588 [0.567,0.609] | 0.588 [0.567,0.609] |
| ML known sigma | 0.743 [0.723,0.762] | 0.743 [0.723,0.762] |
| ML snr+5 | 0.687 [0.666,0.708] | 0.687 [0.666,0.708] |
| ML snr-5 | 0.696 [0.677,0.716] | 0.696 [0.677,0.716] |
| ML nominal 10dB | 0.659 [0.640,0.678] | 0.659 [0.640,0.678] |
| ML joint sigma | 0.730 [0.710,0.750] | 0.730 [0.710,0.750] |
| hybrid known sigma | 0.739 [0.720,0.759] | 0.739 [0.720,0.759] |
| hybrid snr+5 | 0.688 [0.669,0.709] | 0.688 [0.669,0.709] |
| hybrid snr-5 | 0.696 [0.677,0.716] | 0.696 [0.677,0.716] |
| hybrid nominal 10dB | 0.662 [0.642,0.681] | 0.662 [0.642,0.681] |
| hybrid joint sigma | 0.725 [0.705,0.745] | 0.725 [0.705,0.745] |

## Paired differences (pooled; * = CI excludes 0)

| a - b | paper | strict |
|---|---|---|
| CNN read-out - NOMP | +0.022 [-0.001,+0.044] | +0.022 [-0.001,+0.044] |
| hybrid known sigma - ML known sigma | -0.003 [-0.010,+0.003] | -0.003 [-0.010,+0.003] |
| hybrid nominal 10dB - ML known sigma | -0.081 [-0.096,-0.066]* | -0.081 [-0.096,-0.066]* |
| hybrid nominal 10dB - ML nominal 10dB | +0.003 [-0.004,+0.010] | +0.003 [-0.004,+0.010] |
| hybrid joint sigma - ML known sigma | -0.018 [-0.030,-0.007]* | -0.018 [-0.030,-0.007]* |
| hybrid joint sigma - ML joint sigma | -0.005 [-0.013,+0.003] | -0.005 [-0.013,+0.003] |
| hybrid snr+5 - ML snr+5 | +0.001 [-0.006,+0.007] | +0.001 [-0.006,+0.007] |
| hybrid snr-5 - ML snr-5 | +0.000 [-0.006,+0.006] | +0.000 [-0.006,+0.006] |
| ML snr+5 - ML known sigma | -0.055 [-0.070,-0.041]* | -0.055 [-0.070,-0.041]* |
| ML snr-5 - ML known sigma | -0.046 [-0.058,-0.035]* | -0.046 [-0.058,-0.035]* |
| ML nominal 10dB - ML known sigma | -0.084 [-0.098,-0.069]* | -0.084 [-0.098,-0.069]* |
| ML joint sigma - ML known sigma | -0.013 [-0.023,-0.004]* | -0.013 [-0.023,-0.004]* |
| CNN read-out - ML known sigma | -0.154 [-0.175,-0.135]* | -0.154 [-0.175,-0.135]* |
| CNN read-out - ML nominal 10dB | -0.071 [-0.090,-0.052]* | -0.071 [-0.090,-0.052]* |
| hybrid nominal 10dB - CNN read-out | +0.074 [+0.055,+0.092]* | +0.074 [+0.055,+0.092]* |

## Latency (seconds per scene, 1 thread, 12 scenes at 10 dB, batch of 12)

| item | s/scene |
|---|---|
| NOMP | 0.0104 |
| consistent NOMP | 0.0486 |
| CNN read-out | 0.0049 |
| ML starts only (13, no EM) | 0.1128 |
| ML starts only (13 + EM) | 0.1887 |
| ML known sigma LM only | 0.2608 |
| ML wrong/fixed sigma LM only (13 starts) | 0.2056 |
| hybrid LM only (CNN seeds) | 0.0319 |
| ML known sigma TOTAL | 0.4495 |
| ML wrong/fixed sigma TOTAL | 0.3184 |
| hybrid TOTAL (CNN + one LM) | 0.0368 |

## Reading (plain English)
- 3 bits: as expected the classical side wins. The CNN read-out is only +0.022 over NOMP (n.s.) and below consistent NOMP (0.588 vs 0.656), 0.154* below ML with known sigma and 0.071* below even the fixed-10 dB ML.
- ML needs a noise level (wrong by 5 dB: -0.046 to -0.055*; fixed 10 dB: -0.084*) but the joint-sigma ML recovers most of it (-0.013*). The hybrid is indistinguishable from the matching ML in every arm (|diff| <= 0.005, n.s.), so the CNN seed adds nothing at 3 bits.
- Strict Pd equals paper-style Pd (no dropped scenes).
