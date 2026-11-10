"""Reference decoders on the SAME test banks as run_gnomp.py (seeds 12000-12004, bits 1-4, SNR -10..25, 300 scenes/cell):
  'ML (all starts)'   quantiser maximum likelihood, 13 starts + EM, known sigma (kappa = 1, tuned value in tune.json),
                      advc.ml_quant (finetune/adversarial/advc.py, mlq.py);
  'consistent NOMP'   finetune/adc/acommon.consistent_nomp (8 iterations).
Stats are written next to the GNOMP stats on C: (same naming), so analyze.py can compare them scene by scene.
Usage: python run_ml.py   (resumable: finished cells are skipped). Heavy (about 0.5 s per scene, 1 thread): run alone."""
import os, sys, json, time
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/finetune/adversarial')
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/finetune/adc')
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/stress')
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np
import common as C
import stress_sim as SS
import acommon as A
import advc
CACHE = 'C:/Users/Milon/iabr_deep_research_cache/adc_program/baselines/gnomp/cache'   # same constants as run_gnomp.py
L = 3; NS = 300; CH = 100
SNRS = [-10, 0, 10, 15, 25]; BITS = [1, 2, 3, 4]
SEEDS = {s: 12000 + i for i, s in enumerate(SNRS)}


def stats_with_drops(psi, phi, bank):
    """identical to run_gnomp.stats_with_drops (NaN-padded scenes with fewer than L detections are dropped)."""
    full = ~np.isnan(psi).any(1) & ~np.isnan(phi).any(1)
    S = np.zeros((len(psi), 7)); S[~full] = [0, 0, 1, 0, L, 0, L]
    if full.any():
        sub = dict(psi=bank['psi'][full], phi=bank['phi'][full], L=bank['L'][full])
        S[full] = C.scene_stats([(psi[i], phi[i]) for i in np.where(full)[0]], sub)
    return S

summary_path = f'{HERE}/results_ml.json'
summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
for bits in BITS:
    for snr in SNRS:
        key = f'b{bits}_s{snr}'
        if key in summary:
            continue
        base = SS.base_scenes(NS, L, float(snr), SEEDS[snr]); Y = SS.generate(base, 'M5', bits)
        bank = dict(psi=base['psi'], phi=base['phi'], L=base['L'])
        out = {'ML (all starts)': ([], []), 'consistent NOMP': ([], [])}
        t0 = time.time()
        for c0 in range(0, NS, CH):
            sl = slice(c0, c0 + CH); Yc = Y[sl]
            o = advc.ml_quant(Yc, bits, float(snr), L, kappa=1.0, em_iters=6)
            pu = C.uv_to_angles(o['ML (all starts)'][0], o['ML (all starts)'][1])
            Uc, Vc, _ = A.consistent_nomp(Yc, bits, L, 8); pc = C.uv_to_angles(Uc, Vc)
            for name, p in [('ML (all starts)', pu), ('consistent NOMP', pc)]:
                out[name][0].append(np.asarray(p[0])[:, :L]); out[name][1].append(np.asarray(p[1])[:, :L])
        res = {}
        for name in out:
            psi = np.concatenate(out[name][0]); phi = np.concatenate(out[name][1])
            nfound = np.sum(~np.isnan(psi), 1)
            tag = name.replace(' ', '').replace('(', '').replace(')', '')
            np.savez(f'{CACHE}/{tag}_b{bits}_s{snr}.npz', psi=psi, phi=phi, n=nfound)
            S = stats_with_drops(psi, phi, bank)
            np.save(f'{CACHE}/stats_{tag}_b{bits}_s{snr}.npy', S)
            mt = C.metric_from_stats(S, np.arange(NS))
            res[name] = dict(pd_paper=float(mt['pd_paper']), pd_strict=float(mt['pd_strict']), rmse=float(mt['rmse']))
        summary[key] = res
        json.dump(summary, open(summary_path, 'w'), indent=1, default=float)
        print(f"{key}: ML {res['ML (all starts)']['pd_paper']:.4f}  consistentNOMP {res['consistent NOMP']['pd_paper']:.4f}  wall {time.time() - t0:.0f}s", flush=True)
print('done', flush=True)
