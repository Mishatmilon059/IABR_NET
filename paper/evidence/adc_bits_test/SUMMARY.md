# SUMMARY: pooled unknown-SNR Pd (paper-style; strict is identical, no dropped scenes), 450 fresh scenes per bit depth (150 at each of 0/10/20 dB, seeds 14000/14010/14020), L=3, peak-AGC ADC, ideal hardware

| method | 1 bit | 2 bits | 3 bits | 4 bits |
|---|---|---|---|---|
| NOMP | 0.192 | 0.329 | 0.566 | 0.753 |
| consistent NOMP | 0.235 | 0.438 | 0.656 | 0.780 |
| CNN read-out | 0.343 | 0.528 | 0.588 | 0.556 |
| ML known sigma | 0.438 | 0.617 | 0.743 | 0.819 |
| ML snr+5 | 0.402 | 0.550 | 0.687 | 0.789 |
| ML snr-5 | 0.403 | 0.550 | 0.696 | 0.789 |
| ML nominal 10dB | 0.389 | 0.508 | 0.659 | 0.775 |
| ML joint sigma | 0.410 | 0.570 | 0.730 | 0.809 |
| hybrid known sigma | 0.456 | 0.615 | 0.739 | 0.810 |
| hybrid snr+5 | 0.421 | 0.554 | 0.688 | 0.781 |
| hybrid snr-5 | 0.422 | 0.557 | 0.696 | 0.784 |
| hybrid nominal 10dB | 0.403 | 0.514 | 0.662 | 0.769 |
| hybrid joint sigma | 0.428 | 0.576 | 0.725 | 0.803 |

Pd loss when the assumed noise level is wrong (ML known sigma minus variant, pooled; positive = loss):

| variant | 1 bit | 2 bits | 3 bits | 4 bits |
|---|---|---|---|---|
| ML snr+5 | +0.036 | +0.066 | +0.055 | +0.029 |
| ML snr-5 | +0.035 | +0.066 | +0.046 | +0.030 |
| ML nominal 10dB | +0.049 | +0.109 | +0.084 | +0.044 |
| ML joint sigma | +0.028 | +0.046 | +0.013 | +0.009 |

Best method per bit depth (point estimate): 1 bit: hybrid known sigma 0.456; 2 bit: ML known sigma 0.617; 3 bit: ML known sigma 0.743; 4 bit: ML known sigma 0.819

## Reading
- CNN read-out beats NOMP only at 1-2 bits (+0.151*, +0.199*), ties at 3 bits (+0.022 n.s.) and loses at 4 bits (-0.196*). It is below quantiser ML with known sigma at every bit depth (-0.095*, -0.089*, -0.154*, -0.262*).
- Against SNR-blind ML (fixed 10 dB) the CNN read-out is worse at 1 bit (-0.046*), equal at 2 bits (+0.020, n.s.), worse at 3-4 bits: "network needs no noise level" helps only at ~2 bits.
- Hybrid (ML from CNN seeds) helps at 1 bit (+0.014 to +0.019* over matching ML), is neutral at 2-3 bits, and slightly hurts at 4 bits (-0.005 to -0.009*). 3-4 bits favour the classical decoder, as expected.
