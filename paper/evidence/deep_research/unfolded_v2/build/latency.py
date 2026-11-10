"""Latency (criterion S5): 1 thread, batch 1 (median over 60 scenes, after 5 warm-up calls) and batch 256 (ms per scene), on official-bank
scenes (ideal hardware, L = 3) and on 2-bit / 15 dB ADC scenes (M5_b2_snr15). Methods: A, B (float64 inference as evaluated), NOMP os4 rc3,
NOMP os8 rc6, classical skeleton, numpy quantised ML 13 starts + EM (advc.ml_quant, ADC only, 20 scenes batch 1).  stdout -> latency.log"""
import os
os.environ['THREADS'] = '1'
from ev import *
import statistics
torch.set_num_threads(1)
out = {}
for bank in ['official', 'M5_b2_snr15']:
    d = load(bank); Y = d['Y']; bits = d['bits']; snr = d['snr']; H = SS.to_ant(Y)
    sk = UNOMP('A'); sk.cfg.update(classical=True, hyps=['H1', 'H3', 'H4', 'H5', 'H6'], sc=False)
    fns = {'A': lambda ii: decode_np(net('A'), Y[ii], bits[ii], 3, snr=snr[ii], bs=256),
           'B': lambda ii: decode_np(net('B'), Y[ii], bits[ii], 3, snr=snr[ii], bs=256),
           'NOMP os4 rc3': lambda ii: C.nomp_greedy(H[ii], 3, 4, 3),
           'NOMP os8 rc6': lambda ii: C.nomp_greedy(H[ii], 3, 8, 6),
           'classical skeleton': lambda ii: decode_np(sk, Y[ii], bits[ii], 3, snr=snr[ii], bs=256)}
    res = {}
    for nm, f in fns.items():
        for i in range(5):
            f(np.array([i]))
        t = []
        for i in range(100, 160):
            t0 = time.perf_counter(); f(np.array([i])); t.append(1000 * (time.perf_counter() - t0))
        ii = np.arange(256); t0 = time.perf_counter(); f(ii); tb = 1000 * (time.perf_counter() - t0) / 256
        res[nm] = dict(batch1_median_ms=statistics.median(t), batch1_p90_ms=float(np.percentile(t, 90)), batch256_ms_per_scene=tb)
        print(f'{bank:12s} {nm:20s} batch 1 median {statistics.median(t):7.1f} ms (p90 {np.percentile(t, 90):7.1f}); batch 256 {tb:6.2f} ms/scene', flush=True)
    if bank.startswith('M5'):
        sys.path.insert(0, f'{DR}/finetune/adversarial'); import advc
        t = []
        for i in range(100, 120):
            t0 = time.perf_counter(); advc.ml_quant(Y[i:i + 1], 2, snr[i:i + 1], 3); t.append(1000 * (time.perf_counter() - t0))
        res['quantised ML 13 starts + EM (numpy)'] = dict(batch1_median_ms=statistics.median(t))
        print(f'{bank:12s} quantised ML 13 starts + EM (numpy): batch 1 median {statistics.median(t):.1f} ms', flush=True)
    out[bank] = res
save_json(out, f'{HERE}/latency.json')
