"""Error IABC ADDS on clean (unimpaired) fresh scenes, from cache/a1__clean.npz (a1_replicate.py, seeds 8800+k). RMS over scenes of the
per-scene identifiable residual (= the IABC estimate itself, since the target is 0). usage: python a4_clean_added.py > a4_clean_added.log"""
import os, numpy as np
z = np.load(os.path.dirname(os.path.abspath(__file__)) + '/cache/a1__clean.npz'); snr = z['snr']
print(f'clean, all SNR: IABC adds phase RMS {np.sqrt(z["ph_iabc"].mean()):.3f} deg, gain RMS {np.sqrt(z["g_iabc"].mean()):.3f} dB (N={len(snr)})')
for s in sorted(set(snr.tolist())):
    m = snr == s
    print(f'  SNR {s:+5.0f}: phase {np.sqrt(z["ph_iabc"][m].mean()):.3f} deg, gain {np.sqrt(z["g_iabc"][m].mean()):.3f} dB | SC tuned adds {np.sqrt(z["ph_sc_tun"][m].mean()):.3f} deg / {np.sqrt(z["g_sc_tun"][m].mean()):.3f} dB')
print(f'mean of per-scene RMS (the metric of the original report): phase {np.sqrt(z["ph_iabc"]).mean():.3f} deg, gain {np.sqrt(z["g_iabc"]).mean():.3f} dB')
