# RESULT 1-bit ADC (peak AGC, L=3, ideal hardware)

Fresh scenes seeds 14000/14010/14020 (0/10/20 dB), 150 scenes per cell, same scenes for all methods. Pd = paper-style Pd (|err| <= 1 deg). Script: adc_bits.py, analyze.py.

## Per-SNR Pd (paper-style)

| method | 0 dB | 10 dB | 20 dB |
|---|---|---|---|
| NOMP | 0.169 | 0.218 | 0.189 |
| consistent NOMP | 0.214 | 0.291 | 0.200 |
| CNN read-out | 0.232 | 0.416 | 0.382 |
| ML known sigma | 0.281 | 0.480 | 0.553 |
| ML snr+5 | 0.267 | 0.433 | 0.506 |
| ML snr-5 | 0.227 | 0.460 | 0.523 |
| ML nominal 10dB | 0.243 | 0.480 | 0.444 |
| ML joint sigma | 0.243 | 0.473 | 0.514 |
| hybrid known sigma | 0.279 | 0.490 | 0.600 |
| hybrid snr+5 | 0.269 | 0.439 | 0.556 |
| hybrid snr-5 | 0.228 | 0.464 | 0.573 |
| hybrid nominal 10dB | 0.251 | 0.490 | 0.469 |
| hybrid joint sigma | 0.254 | 0.476 | 0.554 |

## Pooled unknown-SNR (all 450 scenes; paired bootstrap 95% CI, 2000 resamples, seed 0, stratified by SNR)

| method | Pd paper [95% CI] | Pd strict [95% CI] |
|---|---|---|
| NOMP | 0.192 [0.176,0.208] | 0.192 [0.176,0.208] |
| consistent NOMP | 0.235 [0.219,0.252] | 0.235 [0.219,0.252] |
| CNN read-out | 0.343 [0.324,0.363] | 0.343 [0.324,0.363] |
| ML known sigma | 0.438 [0.416,0.459] | 0.438 [0.416,0.459] |
| ML snr+5 | 0.402 [0.382,0.422] | 0.402 [0.382,0.422] |
| ML snr-5 | 0.403 [0.383,0.423] | 0.403 [0.383,0.423] |
| ML nominal 10dB | 0.389 [0.371,0.410] | 0.389 [0.371,0.410] |
| ML joint sigma | 0.410 [0.389,0.431] | 0.410 [0.389,0.431] |
| hybrid known sigma | 0.456 [0.435,0.477] | 0.456 [0.435,0.477] |
| hybrid snr+5 | 0.421 [0.402,0.441] | 0.421 [0.402,0.441] |
| hybrid snr-5 | 0.422 [0.402,0.441] | 0.422 [0.402,0.441] |
| hybrid nominal 10dB | 0.403 [0.385,0.422] | 0.403 [0.385,0.422] |
| hybrid joint sigma | 0.428 [0.409,0.449] | 0.428 [0.409,0.449] |

## Paired differences (pooled; * = CI excludes 0)

| a - b | paper | strict |
|---|---|---|
| CNN read-out - NOMP | +0.151 [+0.132,+0.171]* | +0.151 [+0.132,+0.171]* |
| hybrid known sigma - ML known sigma | +0.018 [+0.006,+0.030]* | +0.018 [+0.006,+0.030]* |
| hybrid nominal 10dB - ML known sigma | -0.035 [-0.053,-0.017]* | -0.035 [-0.053,-0.017]* |
| hybrid nominal 10dB - ML nominal 10dB | +0.014 [+0.003,+0.026]* | +0.014 [+0.003,+0.026]* |
| hybrid joint sigma - ML known sigma | -0.010 [-0.026,+0.007] | -0.010 [-0.026,+0.007] |
| hybrid joint sigma - ML joint sigma | +0.018 [+0.006,+0.030]* | +0.018 [+0.006,+0.030]* |
| hybrid snr+5 - ML snr+5 | +0.019 [+0.009,+0.031]* | +0.019 [+0.009,+0.031]* |
| hybrid snr-5 - ML snr-5 | +0.019 [+0.007,+0.030]* | +0.019 [+0.007,+0.030]* |
| ML snr+5 - ML known sigma | -0.036 [-0.049,-0.024]* | -0.036 [-0.049,-0.024]* |
| ML snr-5 - ML known sigma | -0.035 [-0.048,-0.021]* | -0.035 [-0.048,-0.021]* |
| ML nominal 10dB - ML known sigma | -0.049 [-0.064,-0.033]* | -0.049 [-0.064,-0.033]* |
| ML joint sigma - ML known sigma | -0.028 [-0.041,-0.013]* | -0.028 [-0.041,-0.013]* |
| CNN read-out - ML known sigma | -0.095 [-0.116,-0.073]* | -0.095 [-0.116,-0.073]* |
| CNN read-out - ML nominal 10dB | -0.046 [-0.066,-0.026]* | -0.046 [-0.066,-0.026]* |
| hybrid nominal 10dB - CNN read-out | +0.060 [+0.041,+0.080]* | +0.060 [+0.041,+0.080]* |

## Latency (seconds per scene, 1 thread, 12 scenes at 10 dB, batch of 12)

| item | s/scene |
|---|---|
| NOMP | 0.0024 |
| consistent NOMP | 0.0298 |
| CNN read-out | 0.0062 |
| ML starts only (13, no EM) | 0.0677 |
| ML starts only (13 + EM) | 0.1136 |
| ML known sigma LM only | 0.1420 |
| ML wrong/fixed sigma LM only (13 starts) | 0.1438 |
| hybrid LM only (CNN seeds) | 0.0116 |
| ML known sigma TOTAL | 0.2556 |
| ML wrong/fixed sigma TOTAL | 0.2116 |
| hybrid TOTAL (CNN + one LM) | 0.0177 |

## Reading (plain English)
- 1 bit: the CNN read-out beats NOMP by +0.151* but stays 0.095* below quantiser ML with the true noise level, and 0.046* below ML that simply assumes a fixed 10 dB: the network alone does NOT beat noise-blind ML here.
- Wrong or unknown noise level costs ML 0.028-0.049 pooled Pd (all significant); the hybrid (ML started from CNN seeds) wins +0.014 to +0.019* over ML under the same wrong/unknown assumption, but hybrid with fixed 10 dB is still 0.035* below ML with known sigma.
- Strict Pd equals paper-style Pd in every method (no scene returned fewer than 3 paths, so no 'dropped' scenes).
