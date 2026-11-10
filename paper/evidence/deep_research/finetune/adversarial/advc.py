"""Shared setup for finetune/adversarial. Threads 6 (set before numpy/torch). Read-only use of finetune/adc/acommon.py (module A) and
finetune/imp/icom.py; nothing outside this folder is written (large arrays -> C: scratch, see PCACHE).
Seeds of this stage (grep-checked unused anywhere under D:/ai_ml_project before use, see s1_audit.log):
  ADC test banks         8700 (0 dB) / 8715 (15 dB), 1000 scenes, b = 1, 2 (also 3, 4 where noted)
  ADC tuning banks       8730 (0 dB) / 8745 (15 dB), 300 scenes (only for kappa / restart settings of the ML estimator)
  imperfection replicates 8760 (0 dB) / 8775 (15 dB), 1000 scenes
  fresh official-like     8780 + k (k = 0..7 for SNR -10, -5, ..., 25 dB), 250 scenes each, ideal hardware, L = 3
  bootstrap seed 0 (2000 resamples within SNR groups)."""
import os, sys, json, time
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = os.environ.get('THREADS', '6')
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/finetune/adc')
import acommon as A                      # noqa: E402
np, torch, C, SS = A.np, A.torch, A.C, A.SS
import mlq                               # noqa: E402
torch.set_num_threads(int(os.environ.get('THREADS', 6)))
PCACHE = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/finetune_adv'
os.makedirs(PCACHE, exist_ok=True)
TEST_SEEDS = {0: 8700, 15: 8715}
TUNE_SEEDS = {0: 8730, 15: 8745}
IMP_SEEDS = {0: 8760, 15: 8775}
OFF_SEED0 = 8780
MO = mlq.Model(SS.W_IDEAL, SS.F_IDEAL, C)


def sigma_of(snr):
    return np.sqrt(10 ** (-np.asarray(snr, float) / 10) / 2)


def boot(S, snr, pairs, B=2000):
    Bt = C.boot_metrics(S, snr, B=B, seed=0)
    pdm = {m: C.metric_by_snr(S[m], snr)['mean']['pd_paper'] for m in S}
    out = {}
    for a_, b_ in pairs:
        d = Bt[a_][:, 0] - Bt[b_][:, 0]
        out[f'{a_} - {b_}'] = [pdm[a_] - pdm[b_], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    return pdm, {m: list(C.ci(Bt[m][:, 0])[1:]) for m in S}, out


def fmt(d):
    return f'{d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}]' + ('*' if d[1] > 0 or d[2] < 0 else '')


def ml_quant(Yq, bits, snr, L=3, kappa=1.0, K=5, extra=None, em_iters=6, lm_iters=40, return_all=False):
    """Quantisation-aware classical ML (see mlq.py). Starts: consistent NOMP (acommon.consistent_nomp, 8 it), plain NOMP, consistent
    parabolic (FFT parabolic os16 -> consistent NOMP), all L-subsets of the top-K FFT local maxima (4x grid), and EM from the consistent
    NOMP start; every start -> convex ML gain fit -> joint LM; per scene the lowest exact NLL wins. extra: dict name -> (U, V) additional
    starts reported separately (e.g. oracle or network seeds; they are NOT part of the classical estimator)."""
    lo, hi = mlq.cells_ri(Yq, bits, A.adc_cells)
    s = (kappa * sigma_of(snr))[:, None] * np.ones((len(Yq), 1))
    H = A.to_ant(Yq)
    starts = {}
    Uc, Vc, _ = A.consistent_nomp(Yq, bits, L, 8); starts['consistent NOMP'] = (Uc, Vc)
    starts['NOMP'] = C.nomp_greedy(H, L, 4, 3)
    ps = A.parabolic_heat_decode(A.fft_heat(H, 16), L)
    U0 = np.array([np.pi * np.cos(p[0][:L]) for p in ps]); V0 = np.array([np.pi * np.cos(p[1][:L]) for p in ps])
    starts['consistent parabolic'] = A.consistent_nomp(Yq, bits, L, 8, U=U0, V=V0)[:2]
    Uk, Vk = mlq.fft_cands(H, K, 4)
    Uk, Vk = C.newton(H[:, None].repeat(K, 1).reshape(-1, 16, 16), Uk.ravel(), Vk.ravel()); Uk = Uk.reshape(-1, K); Vk = Vk.reshape(-1, K)
    for sub in __import__('itertools').combinations(range(K), L):
        starts[f'fft{sub}'] = (Uk[:, list(sub)], Vk[:, list(sub)])
    res = {}
    for nm, (U, V) in starts.items():
        X = mlq.init_gains(MO, U, V, Yq, lo, hi, s)
        res[nm] = mlq.lm(MO, U, V, X, lo, hi, s, iters=lm_iters)
    if em_iters:
        U, V = starts['consistent NOMP']; X = mlq.init_gains(MO, U, V, Yq, lo, hi, s)
        Ue, Ve, Xe, _ = mlq.em(MO, U, V, X, lo, hi, s, L, iters=em_iters)
        res['EM'] = mlq.lm(MO, Ue, Ve, Xe, lo, hi, s, iters=lm_iters)
    out = {}
    U, V, X, F, k = mlq.pick(list(res.values())); out['ML (all starts)'] = (U, V, F, k)
    out['ML from consistent NOMP'] = res['consistent NOMP'][0], res['consistent NOMP'][1], res['consistent NOMP'][3], None
    if em_iters:
        out['EM + ML'] = res['EM'][0], res['EM'][1], res['EM'][3], None
    for nm, (U, V) in (extra or {}).items():
        X = mlq.init_gains(MO, U, V, Yq, lo, hi, s); r = mlq.lm(MO, U, V, X, lo, hi, s, iters=lm_iters)
        out[f'ML from {nm}'] = (r[0], r[1], r[3], None)
    out['_start_names'] = list(res.keys())
    if return_all:
        out['_res'] = res
    return out
