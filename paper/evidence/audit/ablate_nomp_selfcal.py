"""Missing ablation row: network-free NOMP + self-calibration, scored with the paper's own evaluator (prepare_for_metric / get_ang_difference).
Sanity check: the greedy-only decode must reproduce the cached 'NOMP_no_network' predictions of the original run.
Self-calibration setting (lam, tau) = (1.0, 5.0) is the one tuned for the hybrid on separate tuning banks; it was NOT re-tuned for the network-free decoder."""
import sys, os, ast, json, time, numpy as np, scipy.ndimage
sys.path.insert(0, 'D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA'); sys.path.insert(0, 'D:/ai_ml_project/IABR_Net_TestSuite')
import cv2
EVAL = 'D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA/src/TVT_Blob_Inference.py'
tree = ast.parse(open(EVAL, encoding='utf-8').read()); ns = dict(np=np, cv2=cv2)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['prepare_for_metric', 'get_ang_difference']], type_ignores=[]), EVAL, 'exec'), ns)
ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
src = open(f'{ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
g = dict(np=np, scipy=scipy, CFG=dict(nomp_os=4, nomp_rc=3)); exec('\n'.join(src[527:647]), g); decode = g['decode']
GEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/generated_banks'


def load_bank(name):
    if name == 'official':
        b = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz'); return b['feat'][:, 0], b['feat'][:, 1], b['meta'][:, 0].astype(int), b['meta'][:, 1]
    p = f'{ROOT}/data/banks/{name}.npz'
    b = np.load(p if os.path.exists(p) else f'{GEN}/{name}.npz'); return b['psi'], b['phi'], b['L'].astype(int), b['snr']


def per_sample(psi_p, phi_p, psi_t, phi_t, Ls, n_det=None):
    out = np.zeros(len(Ls))
    for i, L in enumerate(Ls):
        n = L if n_det is None else int(n_det[i])
        gt, p = ns['prepare_for_metric']((psi_p[i, :n], phi_p[i, :n]), np.stack([psi_t[i, :L], phi_t[i, :L]]).astype(np.float64))
        out[i] = (np.abs(ns['get_ang_difference'](gt, p, return_flat=False)) <= 1.0).sum() / (2 * L)
    return out


def run(Ht, L, mode, sc, chunk=1000):
    P, F = [], []
    for s in range(0, len(Ht), chunk):
        a, b = decode(Ht[s:s + chunk].astype(np.complex128), L, None, mode, sc); P.append(a); F.append(b)
    return np.concatenate(P), np.concatenate(F)


def boot(a, b, B=2000, seed=0):
    r = np.random.default_rng(seed); d = a - b; n = len(d)
    m = np.array([d[r.integers(0, n, n)].mean() for _ in range(B)]); return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


banks = sys.argv[1].split(',') if len(sys.argv) > 1 else ['official', 'T2_d5', 'gain_g4_snr15', 'gain_g2_snr0', 'oodphase_d15_snr15', 'L8_snr0', 'L8_snr15', 'sep_5deg_snr15']
res = {}
for name in banks:
    t0 = time.time(); psi_t, phi_t, Ls, snr = load_bank(name); L = int(Ls[0]); assert (Ls == L).all()
    net = np.load(f'{ROOT}/outputs/cache/net__{name}.npz'); Ht = net['Ht']
    z_iabr = np.load(f'{ROOT}/outputs/cache/pred__IABR-v2__{name}.npz'); z_nomp = np.load(f'{ROOT}/outputs/cache/pred__NOMP_no_network__{name}.npz')
    pd_iabr = per_sample(z_iabr['psi'], z_iabr['phi'], psi_t, phi_t, Ls, z_iabr['n'])
    pd_nomp_cached = per_sample(z_nomp['psi'], z_nomp['phi'], psi_t, phi_t, Ls, z_nomp['n'])
    p0, f0 = run(Ht, L, 'greedy', None); pd_nomp_re = per_sample(p0, f0, psi_t, phi_t, Ls)
    p1, f1 = run(Ht, L, 'greedy', (1.0, 5.0)); pd_nomp_sc = per_sample(p1, f1, psi_t, phi_t, Ls)
    d_sc_iabr = boot(pd_nomp_sc, pd_iabr); d_sc_nomp = boot(pd_nomp_sc, pd_nomp_cached)
    res[name] = dict(n=len(Ls), L=L, IABR_v2=pd_iabr.mean(), NOMP_cached=pd_nomp_cached.mean(), NOMP_recomputed=pd_nomp_re.mean(), NOMP_selfcal=pd_nomp_sc.mean(),
                     recompute_matches_cache=bool(np.allclose(pd_nomp_re, pd_nomp_cached)), diff_selfcalNOMP_minus_IABR=d_sc_iabr, diff_selfcalNOMP_minus_NOMP=d_sc_nomp)
    r = res[name]
    print(f'{name:22s} n={r["n"]:5d} L={L:2d} | IABR-v2 {r["IABR_v2"]:.4f} | NOMP(cache) {r["NOMP_cached"]:.4f} | NOMP(recomputed) {r["NOMP_recomputed"]:.4f} (identical: {r["recompute_matches_cache"]}) | '
          f'NOMP+selfcal {r["NOMP_selfcal"]:.4f} | (NOMP+sc minus IABR-v2) {d_sc_iabr[0]:+.4f} [{d_sc_iabr[1]:+.4f},{d_sc_iabr[2]:+.4f}] | {time.time()-t0:.0f}s', flush=True)
json.dump(res, open('ablate_nomp_selfcal_results.json', 'w'), indent=1, default=float)
