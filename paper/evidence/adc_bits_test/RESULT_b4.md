# RESULT 4-bit ADC (peak AGC, L=3, ideal hardware)

Fresh scenes seeds 14000/14010/14020 (0/10/20 dB), 150 scenes per cell, same scenes for all methods. Pd = paper-style Pd (|err| <= 1 deg). Script: adc_bits.py, analyze.py.

## Per-SNR Pd (paper-style)

| method | 0 dB | 10 dB | 20 dB |
|---|---|---|---|
| NOMP | 0.700 | 0.823 | 0.734 |
| consistent NOMP | 0.687 | 0.803 | 0.849 |
| CNN read-out | 0.577 | 0.553 | 0.539 |
| ML known sigma | 0.703 | 0.853 | 0.899 |
| ML snr+5 | 0.704 | 0.816 | 0.848 |
| ML snr-5 | 0.699 | 0.833 | 0.834 |
| ML nominal 10dB | 0.692 | 0.853 | 0.779 |
| ML joint sigma | 0.702 | 0.839 | 0.887 |
| hybrid known sigma | 0.696 | 0.841 | 0.893 |
| hybrid snr+5 | 0.696 | 0.804 | 0.842 |
| hybrid snr-5 | 0.692 | 0.826 | 0.833 |
| hybrid nominal 10dB | 0.682 | 0.841 | 0.783 |
| hybrid joint sigma | 0.694 | 0.831 | 0.882 |

## Pooled unknown-SNR (all 450 scenes; paired bootstrap 95% CI, 2000 resamples, seed 0, stratified by SNR)

| method | Pd paper [95% CI] | Pd strict [95% CI] |
|---|---|---|
| NOMP | 0.753 [0.732,0.771] | 0.753 [0.732,0.771] |
| consistent NOMP | 0.780 [0.759,0.798] | 0.780 [0.759,0.798] |
| CNN read-out | 0.556 [0.536,0.576] | 0.556 [0.536,0.576] |
| ML known sigma | 0.819 [0.800,0.837] | 0.819 [0.800,0.837] |
| ML snr+5 | 0.789 [0.769,0.808] | 0.789 [0.769,0.808] |
| ML snr-5 | 0.789 [0.769,0.807] | 0.789 [0.769,0.807] |
| ML nominal 10dB | 0.775 [0.754,0.794] | 0.775 [0.754,0.794] |
| ML joint sigma | 0.809 [0.791,0.829] | 0.809 [0.791,0.829] |
| hybrid known sigma | 0.810 [0.791,0.830] | 0.810 [0.791,0.830] |
| hybrid snr+5 | 0.781 [0.760,0.801] | 0.781 [0.760,0.801] |
| hybrid snr-5 | 0.784 [0.764,0.803] | 0.784 [0.764,0.803] |
| hybrid nominal 10dB | 0.769 [0.748,0.788] | 0.769 [0.748,0.788] |
| hybrid joint sigma | 0.803 [0.783,0.821] | 0.803 [0.783,0.821] |

## Paired differences (pooled; * = CI excludes 0)

| a - b | paper | strict |
|---|---|---|
| CNN read-out - NOMP | -0.196 [-0.218,-0.174]* | -0.196 [-0.218,-0.174]* |
| hybrid known sigma - ML known sigma | -0.009 [-0.015,-0.003]* | -0.009 [-0.015,-0.003]* |
| hybrid nominal 10dB - ML known sigma | -0.050 [-0.060,-0.039]* | -0.050 [-0.060,-0.039]* |
| hybrid nominal 10dB - ML nominal 10dB | -0.006 [-0.013,+0.000] | -0.006 [-0.013,+0.000] |
| hybrid joint sigma - ML known sigma | -0.016 [-0.024,-0.009]* | -0.016 [-0.024,-0.009]* |
| hybrid joint sigma - ML joint sigma | -0.007 [-0.013,-0.000]* | -0.007 [-0.013,-0.000]* |
| hybrid snr+5 - ML snr+5 | -0.009 [-0.014,-0.003]* | -0.009 [-0.014,-0.003]* |
| hybrid snr-5 - ML snr-5 | -0.005 [-0.011,-0.001]* | -0.005 [-0.011,-0.001]* |
| ML snr+5 - ML known sigma | -0.029 [-0.039,-0.020]* | -0.029 [-0.039,-0.020]* |
| ML snr-5 - ML known sigma | -0.030 [-0.039,-0.021]* | -0.030 [-0.039,-0.021]* |
| ML nominal 10dB - ML known sigma | -0.044 [-0.054,-0.033]* | -0.044 [-0.054,-0.033]* |
| ML joint sigma - ML known sigma | -0.009 [-0.014,-0.004]* | -0.009 [-0.014,-0.004]* |
| CNN read-out - ML known sigma | -0.262 [-0.283,-0.242]* | -0.262 [-0.283,-0.242]* |
| CNN read-out - ML nominal 10dB | -0.219 [-0.241,-0.197]* | -0.219 [-0.241,-0.197]* |
| hybrid nominal 10dB - CNN read-out | +0.213 [+0.192,+0.234]* | +0.213 [+0.192,+0.234]* |

## Latency (seconds per scene, 1 thread, 12 scenes at 10 dB, batch of 12)

| item | s/scene |
|---|---|
| NOMP | 0.0025 |
| consistent NOMP | 0.0308 |
| CNN read-out | 0.0041 |
| ML starts only (13, no EM) | 0.0693 |
| ML starts only (13 + EM) | 0.1383 |
| ML known sigma LM only | 0.1458 |
| ML wrong/fixed sigma LM only (13 starts) | 0.1411 |
| hybrid LM only (CNN seeds) | 0.0146 |
| ML known sigma TOTAL | 0.2841 |
| ML wrong/fixed sigma TOTAL | 0.2104 |
| hybrid TOTAL (CNN + one LM) | 0.0188 |

## Reading (plain English)
- 4 bits: the classical decoder clearly wins. The CNN read-out (fine-tuned) is 0.196* BELOW plain NOMP (0.556 vs 0.753) and 0.262* below ML with known sigma; the network is not competitive here.
- Wrong/unknown noise level costs ML only 0.009-0.044* at 4 bits (joint-sigma ML loses just 0.009*), so the "network needs no noise level" argument is weak at 4 bits.
- Hybrid is slightly worse than the matching ML (-0.005 to -0.009*), so the CNN seed does not help. Strict Pd equals paper-style Pd (no dropped scenes).
