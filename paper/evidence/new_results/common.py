"""Shared helpers for the new_results experiments (E13/E6, E4, E5, E8).
Nothing here modifies any existing file. The evaluator functions are compiled from TVT_Blob_Inference.py (as in the audit script),
the decoder is compiled from notebook_src.py lines 528-647 (1-based; python slice [527:647]), metrics() is compiled verbatim from lines 315-343."""
import sys, os, ast, json, time, numpy as np, scipy, scipy.ndimage
sys.path.insert(0, 'D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA'); sys.path.insert(0, 'D:/ai_ml_project/IABR_Net_TestSuite')
import cv2
EVAL = 'D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA/src/TVT_Blob_Inference.py'
_tree = ast.parse(open(EVAL, encoding='utf-8').read()); ns = dict(np=np, cv2=cv2)
exec(compile(ast.Module(body=[n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in ['prepare_for_metric', 'get_ang_difference']], type_ignores=[]), EVAL, 'exec'), ns)
prepare_for_metric, get_ang_difference = ns['prepare_for_metric'], ns['get_ang_difference']
ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
CACHE = f'{ROOT}/outputs/cache'
GEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/generated_banks'
OUT = 'D:/ai_ml_project/paper/evidence/new_results'
_src = open(f'{ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
CFG = dict(nomp_os=4, nomp_rc=3)
g = dict(np=np, scipy=scipy, CFG=CFG); exec('\n'.join(_src[527:647]), g)
decode, nomp_greedy, selfcal, cyclic, newton, ls_gains, residual_energy, uv_to_angles, atoms = [g[k] for k in ['decode', 'nomp_greedy', 'selfcal', 'cyclic', 'newton', 'ls_gains', 'residual_energy', 'uv_to_angles', 'atoms']]
SC = (1.0, 5.0)

# metrics()/by_snr() verbatim from notebook_src.py lines 315-343
_m = dict(np=np, prepare_for_metric=prepare_for_metric, get_ang_difference=get_ang_difference)
exec('\n'.join(_src[314:343]), _m)
nb_metrics, nb_by_snr = _m['metrics'], _m['by_snr']


def load_bank(name):
    """returns dict(psi,phi,L,snr) as the notebook's load_bank (same sources)"""
    if name == 'official':
        b = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
        return dict(psi=b['feat'][:, 0, :], phi=b['feat'][:, 1, :], L=b['meta'][:, 0].astype(int), snr=b['meta'][:, 1].astype(np.float32))
    p = f'{ROOT}/data/banks/{name}.npz'
    z = np.load(p if os.path.exists(p) else f'{GEN}/{name}.npz')
    return dict(psi=z['psi'], phi=z['phi'], L=z['L'].astype(int), snr=z['snr'].astype(np.float32))


def bank_exists(name):
    return name == 'official' or os.path.exists(f'{ROOT}/data/banks/{name}.npz') or os.path.exists(f'{GEN}/{name}.npz')


def load_preds(path):
    z = np.load(path); return [(z['psi'][i, :z['n'][i]], z['phi'][i, :z['n'][i]]) for i in range(len(z['n']))]


def scene_stats(preds, bank):
    """per-scene components of metrics() (identical logic): good, bad, dropped, src_ok, src_tot, sumsq(good errs), L"""
    N = len(preds); S = np.zeros((N, 7))
    for i in range(N):
        L = int(bank['L'][i])
        feat = np.stack([bank['psi'][i, :L], bank['phi'][i, :L]]).astype(np.float64)
        gt, pr = prepare_for_metric(preds[i], feat)
        if np.isnan(pr).any():
            S[i] = [0, 0, 1, 0, L, 0, L]; continue
        dif = get_ang_difference(gt, pr, return_flat=False); ok = np.abs(dif) <= 1.0
        S[i] = [ok.sum(), (~ok).sum(), 0, ok.all(0).sum(), L, np.square(dif[ok]).sum(), L]
    return S   # cols: good, bad, dropped, src_ok, src_tot(=L), sumsq, L


def metric_from_stats(S, idx):
    """metrics() quantities for scene subset idx"""
    s = S[idx]; good, bad, drop, sok, stot, ssq, L = s.sum(0)
    tot = good + bad; strict_bad = 2 * (s[:, 2] * s[:, 6]).sum()
    return dict(pd_paper=good / tot if tot else np.nan, pd_strict=good / (tot + strict_bad) if tot + strict_bad else np.nan,
                pd_source=sok / stot if stot else np.nan, rmse=float(np.sqrt(ssq / good)) if good else np.nan, n=len(idx), dropped=int(drop))


KEYS = ['pd_paper', 'pd_strict', 'pd_source', 'rmse']


def metric_by_snr(S, snr):
    groups = {s: np.where(snr.astype(int) == s)[0] for s in sorted(set(snr.astype(int).tolist()))}
    per = {s: metric_from_stats(S, ix) for s, ix in groups.items()}
    per['mean'] = {k: float(np.nanmean([per[s][k] for s in groups])) for k in KEYS}
    return per


def boot_metrics(S_dict, snr, B=2000, seed=0):
    """Paired bootstrap: resample scenes within each SNR group (same indices for all methods); returns {method: (B,4) array of
    mean-over-SNR metrics [pd_paper, pd_strict, pd_source, rmse]}"""
    rng = np.random.default_rng(seed); snr = snr.astype(int)
    groups = [np.where(snr == s)[0] for s in sorted(set(snr.tolist()))]
    out = {m: np.zeros((B, 4)) for m in S_dict}; acc = {m: np.zeros((B, 4)) for m in S_dict}
    for ix in groups:
        n = len(ix); R = rng.integers(0, n, (B, n))
        for m, S in S_dict.items():
            s = S[ix]
            good = s[:, 0][R].sum(1); bad = s[:, 1][R].sum(1); sb = 2 * (s[:, 2] * s[:, 6])[R].sum(1); sok = s[:, 3][R].sum(1); stot = s[:, 4][R].sum(1); ssq = s[:, 5][R].sum(1)
            tot = good + bad
            with np.errstate(all='ignore'):
                acc[m] += np.stack([good / tot, good / (tot + sb), sok / stot, np.sqrt(ssq / good)], 1)
    for m in S_dict:
        out[m] = acc[m] / len(groups)
    return out


def ci(x):
    return float(np.mean(x)), float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))
