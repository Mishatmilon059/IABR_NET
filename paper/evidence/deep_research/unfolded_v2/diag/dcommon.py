"""Shared helpers for unfolded_v2/diag (read-only use of existing caches). Threads limited to 2."""
import os, sys
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '2'
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import numpy as np
from common import (load_bank, scene_stats, metric_by_snr, boot_metrics, ci, decode, nomp_greedy, selfcal, cyclic, newton,
                    ls_gains, residual_energy, uv_to_angles, atoms, CFG, prepare_for_metric, get_ang_difference)
import common as C
heat_peaks = C.g["heat_peaks"]
import scipy.ndimage
from itertools import combinations, permutations
ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
CACHE = f'{ROOT}/outputs/cache'
HERE = os.path.dirname(os.path.abspath(__file__))
BIG = 'C:/Users/Milon/iabr_deep_research_cache/dr_cache/unfolded_v2/diag'
os.makedirs(BIG, exist_ok=True)
SEP = np.pi / 6


def subset(bank, per=250):
    """first `per` scenes of every SNR group (deterministic, no RNG)"""
    snr = bank['snr'].astype(int)
    return np.concatenate([np.where(snr == s)[0][:per] for s in sorted(set(snr.tolist()))])


def load_official(per=250):
    b = load_bank('official'); idx = subset(b, per)
    z = np.load(f'{CACHE}/net__official.npz')
    H = z['Ht'][idx].astype(np.complex128); heat = z['heat'][idx].astype(np.float64)
    tb = {k: v[idx] for k, v in b.items()}
    p = np.load(f'{CACHE}/pred__NOMP_no_network__official.npz')
    U0 = np.pi * np.cos(p['psi'][idx].astype(np.float64)); V0 = np.pi * np.cos(p['phi'][idx].astype(np.float64))
    return idx, tb, H, heat, U0, V0


def true_uv(tb, L=3):
    return np.pi * np.cos(tb['psi'][:, :L].astype(np.float64)), np.pi * np.cos(tb['phi'][:, :L].astype(np.float64))


def comp_errors(psi_t, phi_t, psi_e, phi_e):
    """(L,) truth vs (L,) estimate, matched as the metric (prepare_for_metric). Returns |err| deg (2,L) after matching and the
    estimate permutation."""
    feat = np.stack([psi_t, phi_t]).astype(np.float64)
    gt, pr = prepare_for_metric((psi_e, phi_e), feat)
    dif = get_ang_difference(gt, pr, return_flat=False)
    return np.abs(dif), gt, pr


def n_correct(psi_t, phi_t, U, V):
    """number of correct components (|err|<=1 deg) for one scene, estimates given in (u,v)"""
    pe, fe = uv_to_angles(U[None], V[None])
    e, _, _ = comp_errors(psi_t, phi_t, pe[0], fe[0])
    return int((e <= 1.0).sum()), float(e.sum())


def sep_violation(U, V):
    """(B,K) -> min pairwise Euclidean distance in (psi,phi) [rad]"""
    ps, ph = uv_to_angles(U, V); K = U.shape[1]; d = np.full(len(U), np.inf)
    for i in range(K):
        for j in range(i + 1, K):
            d = np.minimum(d, np.hypot(ps[:, i] - ps[:, j], ph[:, i] - ph[:, j]))
    return d


def sigma2(snr):
    return 10 ** (-np.asarray(snr, float) / 10)


def bayes_terms(y, U, V, snr, L=3):
    """per scene: residual energy (LS), marginal log-likelihood with x~CN(0,I/L) (gains integrated out), log angle prior
    (uniform psi,phi -> Jacobian in (u,v)), separation indicator. noise var = 10^(-snr/10) (verified in probe)."""
    s2 = sigma2(snr)
    x, A = ls_gains(y, U, V)
    res = (np.abs(y.reshape(len(y), -1) - np.einsum('bik,bk->bi', A, x)) ** 2).sum(1)
    # marginal: C = A A^H / L + s2 I ; use Woodbury: y^H C^-1 y = (|y|^2 - y^H A (L s2 I + A^H A)^-1 A^H y)/s2
    AH = np.conj(np.transpose(A, (0, 2, 1))); G = AH @ A; yv = y.reshape(len(y), -1, 1)
    K = U.shape[1]; Mx = G + (L * s2)[:, None, None] * np.eye(K)[None]
    b = AH @ yv; q = np.real(np.conj(b[..., 0]) * np.linalg.solve(Mx, b)[..., 0]).sum(1)
    quad = ((np.abs(y) ** 2).sum((1, 2)) - q) / s2
    logdet = 256 * np.log(s2) + np.linalg.slogdet(np.eye(K)[None] + G / (L * s2)[:, None, None])[1]
    mll = -quad - logdet
    uu = np.clip(np.angle(np.exp(1j * U)), -np.pi + 1e-6, np.pi - 1e-6); vv = np.clip(np.angle(np.exp(1j * V)), -np.pi + 1e-6, np.pi - 1e-6)
    lprior = -0.5 * np.log(np.pi ** 2 - uu ** 2).sum(1) - 0.5 * np.log(np.pi ** 2 - vv ** 2).sum(1)
    sep = sep_violation(U, V)
    return dict(res=res, mll=mll, lprior=lprior, sep=sep, gains=x)


PERMS = np.array(list(permutations(range(3))))


def vmatch(pt, ft, pe, fe):
    """Vectorised metric matching for L=3 (same rule as prepare_for_metric: permutation of the estimates with minimum total
    Euclidean distance in raw (psi,phi)). pt,ft: (B,3) truth; pe,fe: (...,B,3) or (B,K,3) estimates broadcastable as (B,K,3).
    Returns abs component errors (B,K,2,3) in deg aligned with truth order, and the chosen permutation index."""
    if pe.ndim == 2:
        pe, fe = pe[:, None], fe[:, None]
    P = PERMS
    pp = pe[..., P]; fp = fe[..., P]                      # (B,K,6,3)
    d = np.hypot(pt[:, None, None, :] - pp, ft[:, None, None, :] - fp).sum(-1)   # (B,K,6)
    k = d.argmin(-1)
    pb = np.take_along_axis(pp, k[..., None, None], 2)[:, :, 0]; fb = np.take_along_axis(fp, k[..., None, None], 2)[:, :, 0]
    ep = np.abs(np.rad2deg(np.angle(np.exp(1j * (pt[:, None] - pb))))); ef = np.abs(np.rad2deg(np.angle(np.exp(1j * (ft[:, None] - fb)))))
    return np.stack([ep, ef], 2), k


def score_uv(pt, ft, U, V):
    """U,V (B,K,3) -> n correct components (B,K), sum of abs errors (B,K)"""
    pe, fe = uv_to_angles(U, V)
    e, _ = vmatch(pt, ft, pe, fe)
    return (e <= 1.0).sum((2, 3)), e.sum((2, 3))


def bayes_terms_s2(y, U, V, s2, L=3):
    """as bayes_terms but with an explicit per-scene noise variance s2 (true or estimated)"""
    x, A = ls_gains(y, U, V)
    res = (np.abs(y.reshape(len(y), -1) - np.einsum('bik,bk->bi', A, x)) ** 2).sum(1)
    AH = np.conj(np.transpose(A, (0, 2, 1))); G = AH @ A; yv = y.reshape(len(y), -1, 1)
    K = U.shape[1]; Mx = G + (L * s2)[:, None, None] * np.eye(K)[None]
    b = AH @ yv; q = np.real(np.conj(b[..., 0]) * np.linalg.solve(Mx, b)[..., 0]).sum(1)
    quad = ((np.abs(y) ** 2).sum((1, 2)) - q) / s2
    logdet = 256 * np.log(s2) + np.linalg.slogdet(np.eye(K)[None] + G / (L * s2)[:, None, None])[1]
    uu = np.clip(np.angle(np.exp(1j * U)), -np.pi + 1e-6, np.pi - 1e-6); vv = np.clip(np.angle(np.exp(1j * V)), -np.pi + 1e-6, np.pi - 1e-6)
    lprior = -0.5 * np.log(np.pi ** 2 - uu ** 2).sum(1) - 0.5 * np.log(np.pi ** 2 - vv ** 2).sum(1)
    return dict(res=res, mll=-quad - logdet, lprior=lprior, sep=sep_violation(U, V), gains=x)
