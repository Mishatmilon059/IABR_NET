"""Independent implementation of the TVT evaluator metric (definitions read from TVT_Blob_Inference.py:
prepare_for_metric / get_ang_difference; implemented here without calling them).

Per scene with L true paths (psi_l, phi_l):
  * if fewer than L predictions are returned -> the scene is 'dropped' (paper-style Pd ignores it; strict Pd counts 2L misses).
  * otherwise the first L predicted pairs are matched one-to-one to the true pairs by the assignment that minimises the sum
    of Euclidean distances in the raw (psi, phi) plane (radians, no wrap) -- solved here by exhaustive enumeration of the L!
    permutations (the evaluator uses the Hungarian algorithm; both give the optimum).
  * component errors e = wrap(true - pred) in degrees (wrap to (-180, 180]); a component is 'good' if |e| <= 1 deg.
Aggregates per SNR group: pd_paper = good/(good+bad); pd_strict = good/(good+bad+2L*dropped);
pd_source = paths with both components good / all paths (dropped count as misses); rmse = sqrt(mean e^2 over good components).
Reported value = mean over SNR groups."""
import itertools
import numpy as np

COLS = ['good', 'bad', 'dropped', 'src_ok', 'L', 'sumsq']


def scene_table(pred_psi, pred_phi, n_pred, true_psi, true_phi, L):
    """pred_* (N, >=1) arrays padded with NaN, n_pred (N,), true_* (N, >=L), L (N,) -> per-scene stats (N, 6)."""
    N = len(L); S = np.zeros((N, 6))
    S[:, 4] = L
    pred_psi = np.asarray(pred_psi, np.float64); pred_phi = np.asarray(pred_phi, np.float64)
    true_psi = np.asarray(true_psi, np.float64); true_phi = np.asarray(true_phi, np.float64)
    nan_scenes = 0
    for Lv in np.unique(L):
        Lv = int(Lv); idx = np.where(L == Lv)[0]
        enough = n_pred[idx] >= Lv
        if enough.any():
            ok_idx = idx[enough]
            pp = np.stack([pred_psi[ok_idx, :Lv], pred_phi[ok_idx, :Lv]], -1)       # (M, L, 2)
            nanrow = np.isnan(pp).any((1, 2))
            nan_scenes += int(nanrow.sum())
            S[ok_idx[nanrow], 2] = 1
            ok_idx = ok_idx[~nanrow]; pp = pp[~nanrow]
            tt = np.stack([true_psi[ok_idx, :Lv], true_phi[ok_idx, :Lv]], -1)      # (M, L, 2)
            perms = np.array(list(itertools.permutations(range(Lv))))             # (P, L)
            ch = max(1, 200000 // len(perms))
            for s in range(0, len(ok_idx), ch):
                t = tt[s:s + ch]; p = pp[s:s + ch]
                D = np.sqrt(((t[:, :, None, :] - p[:, None, :, :]) ** 2).sum(-1))   # (m, L_true, L_pred)
                cost = D[:, np.arange(Lv)[None, :], perms].sum(-1)                 # (m, P)
                best = perms[np.argmin(cost, 1)]                                    # true l -> pred best[l]
                pm = np.take_along_axis(p, best[:, :, None], 1)                     # (m, L, 2)
                e = np.rad2deg(np.mod(t - pm + np.pi, 2 * np.pi) - np.pi)          # (m, L, 2) degrees
                g = np.abs(e) <= 1.0
                rows = ok_idx[s:s + len(t)]
                S[rows, 0] = g.sum((1, 2)); S[rows, 1] = (~g).sum((1, 2))
                S[rows, 3] = g.all(2).sum(1); S[rows, 5] = np.where(g, e ** 2, 0).sum((1, 2))
        S[idx[~enough], 2] = 1
    return S, nan_scenes


def agg(S):
    good, bad, drop, sok, L, ssq = S.sum(0)
    sb = 2 * (S[:, 2] * S[:, 4]).sum()
    with np.errstate(all='ignore'):
        return dict(pd_paper=good / (good + bad), pd_strict=good / (good + bad + sb), pd_source=sok / L, rmse=np.sqrt(ssq / good),
                    dropped=int(drop), n=len(S))


KEYS = ['pd_paper', 'pd_strict', 'pd_source', 'rmse']


def by_snr(S, snr):
    snr = np.asarray(snr).astype(int); out = {}
    for s in sorted(set(snr.tolist())):
        out[s] = agg(S[snr == s])
    out['mean'] = {k: float(np.nanmean([out[s][k] for s in out])) for k in KEYS}
    out['mean']['dropped'] = int(sum(out[s]['dropped'] for s in out if s != 'mean'))
    return out


def paired_bootstrap(S_dict, snr, B=2000, seed=0):
    """Resample scenes with replacement within each SNR group (the same indices for every method).
    Returns {method: (B, 4) mean-over-SNR [pd_paper, pd_strict, pd_source, rmse]}."""
    rng = np.random.default_rng(seed); snr = np.asarray(snr).astype(int)
    groups = [np.where(snr == s)[0] for s in sorted(set(snr.tolist()))]
    acc = {m: np.zeros((B, 4)) for m in S_dict}
    for ix in groups:
        R = rng.integers(0, len(ix), (B, len(ix)))
        for m, S in S_dict.items():
            T = S[ix]
            good = T[:, 0][R].sum(1); bad = T[:, 1][R].sum(1); sb = (2 * T[:, 2] * T[:, 4])[R].sum(1)
            sok = T[:, 3][R].sum(1); Lt = T[:, 4][R].sum(1); ssq = T[:, 5][R].sum(1)
            with np.errstate(all='ignore'):
                acc[m] += np.stack([good / (good + bad), good / (good + bad + sb), sok / Lt, np.sqrt(ssq / good)], 1)
    return {m: a / len(groups) for m, a in acc.items()}


def pct_ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
