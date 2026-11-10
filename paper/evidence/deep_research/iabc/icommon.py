"""Shared code for the IABC evidence scripts (iabc/). Nothing outside iabc/ is modified.
Sources of truth (exec-ed, not copied):
  decoder        notebook_src.py slice [527:647] via new_results/common.py (nomp_greedy, cyclic, selfcal, heat_peaks, newton, ...)
  CNN + IABC     notebook_src.py slice [419:522], weights outputs/checkpoints/IABRv2/final.pt (CPU)
  simulator      iabr2_sim.py (sample_paths, simulate, impairment_targets, project_impairment)
  metrics        new_results/common.py (scene_stats, metric_from_stats, metric_by_snr, boot_metrics, ci)
Threads: OMP/OPENBLAS/MKL = 6 and torch.set_num_threads(6) (resume run; attempt 1 used 2 -- thread count does not change any per-scene result)."""
import os, sys, json, time
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = '6'
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = f'{HERE}/cache'
os.makedirs(CACHE, exist_ok=True)
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results'); sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np, scipy, scipy.ndimage, torch, torch.nn as nn, torch.nn.functional as TF
import common as C
import iabr2_sim as S

torch.set_num_threads(6)
_src = open(f'{C.ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
_gm = dict(np=np, torch=torch, nn=nn, TF=TF, W_IDEAL=S.W_IDEAL, F_IDEAL=S.F_IDEAL, P=16, Q=16)
exec('\n'.join(_src[419:522]), _gm)
cnn = _gm['IABRv2'](32, 10, 32)
cnn.load_state_dict(torch.load(f'{C.ROOT}/outputs/checkpoints/IABRv2/final.pt', map_location='cpu')); cnn.eval()
DB = 20.0 / np.log(10.0)          # natural-log amplitude gain -> dB
_nc = np.arange(16.) - 7.5
TUNED = (1.0, 2.0, 4)             # lam, tau, iters  (stress/adversarial/replicate.py tuning result)
DEFAULT = (1.0, 5.0, 2)           # notebook SC = (1, 5), iters 2


def to_ant(Y):
    """Ht = W_IDEAL Y F_IDEAL^H (notebook line 274 / IABRv2.front)."""
    return np.matmul(np.matmul(S.W_IDEAL[None], Y), S.F_IDEAL.conj().T[None])


def net(Y, bs=250):
    """CNN forward on CPU -> sigmoid heat (B,32,32), Hc (B,16,16), imp (B,32,2) [identifiable phase rad, log-gain]."""
    X = np.ascontiguousarray(S.to_ri(Y).transpose(0, 3, 1, 2)); hs, hc, im = [], [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            o = cnn(torch.from_numpy(X[s:s + bs]))
            hs.append(torch.sigmoid(o['heat']).double().numpy()); hc.append(o['Hc'].numpy().astype(complex)); im.append(o['imp'].double().numpy())
    return np.concatenate(hs), np.concatenate(hc), np.concatenate(im)


def selfcal_ret(y, U, V, lam, tau, rc=2, iters=2, record=(), fix_angles=False):
    """Line-for-line copy of notebook selfcal() that ALSO returns the per-antenna correction it applied.
    Correction convention (same as the notebook and IABRv2.front): yc = y * exp(-lg_r - j ph_r)[rows] * exp(-lg_t - j ph_t)[cols],
    so the estimate of the identifiable distortion is est = (ph, lg) with rows 0-15 Rx, 16-31 Tx, (B,32,2), zero where the gate is shut.
    The estimate at iteration k is computed from the ORIGINAL y (not incremental), so the cumulative correction = the last iteration's.
    record: iteration counts (1-based) at which to snapshot (U, V, est, gate). fix_angles=True skips the cyclic angle refinement
    (oracle ALS with true angles). Returns U, V, est, gate, snaps."""
    B = len(y); yc = y.copy(); U, V = U.copy(), V.copy(); snaps = {}
    est = np.zeros((B, 32, 2)); gate = np.zeros(B, bool)
    for it in range(iters):
        x, A = C.ls_gains(yc, U, V); Hm = np.einsum('bik,bk->bi', A, x).reshape(y.shape)
        snr_est = (np.abs(Hm) ** 2).sum((1, 2)) / ((np.abs(yc - Hm) ** 2).sum((1, 2)) + 1e-12)
        dt = np.ones((B, 16), complex)
        for _ in range(3):
            M = Hm * dt[:, None, :]
            dr = (y * np.conj(M)).sum(2) / ((np.abs(M) ** 2).sum(2) + 1e-12)
            M = Hm * dr[:, :, None]
            dt = (y * np.conj(M)).sum(1) / ((np.abs(M) ** 2).sum(1) + 1e-12)

        def clean(d):
            lg = np.log(np.abs(d) + 1e-12); ph = np.unwrap(np.angle(d), axis=1)
            lg = lg - lg.mean(1, keepdims=True); ph = ph - ph.mean(1, keepdims=True)
            ph = ph - (ph @ _nc)[:, None] / (_nc @ _nc) * _nc
            return lam * lg, lam * ph
        lr, pr = clean(dr); lt, pt = clean(dt)
        g = (snr_est > tau)[:, None]
        cr = np.where(g, np.exp(-lr - 1j * pr), 1.0); ct = np.where(g, np.exp(-lt - 1j * pt), 1.0)
        yc = y * cr[:, :, None] * ct[:, None, :]
        if not fix_angles:
            U, V = C.cyclic(yc, U, V, rc)
        gate = g[:, 0]
        est = np.where(g[:, :, None], np.stack([np.concatenate([pr, pt], 1), np.concatenate([lr, lt], 1)], -1), 0.0)
        if it + 1 in record:
            snaps[it + 1] = (U.copy(), V.copy(), est.copy(), gate.copy())
    return U, V, est, gate, snaps


def heat_seed(Hc, heat, L, rc=3):
    """decode(..., mode='heat') up to (not including) self-cal: heat peaks -> Newton -> cyclic (notebook decode lines)."""
    Uh, Vh = C.g['heat_peaks'](heat, L)
    Uh, Vh = C.newton(Hc[:, None].repeat(L, 1).reshape(-1, 16, 16), Uh.ravel(), Vh.ravel())
    return C.cyclic(Hc, Uh.reshape(-1, L), Vh.reshape(-1, L), rc)


def wrap(x):
    return np.angle(np.exp(1j * x))


def ident_err(target, est):
    """identifiable error (B,32,2): phase wrapped then re-projected per array (rows/cols), log-gain mean-removed per array."""
    e = target - est
    ph = wrap(e[..., 0]); ph = np.concatenate([S.project_impairment(ph[:, :16]), S.project_impairment(ph[:, 16:])], 1)
    lg = e[..., 1]; lg = np.concatenate([lg[:, :16] - lg[:, :16].mean(1, keepdims=True), lg[:, 16:] - lg[:, 16:].mean(1, keepdims=True)], 1)
    return np.stack([ph, lg], -1)


def rms_deg_db(e):
    """per-scene RMS over the 32 antennas: phase in degrees, gain in dB."""
    return np.rad2deg(np.sqrt((e[..., 0] ** 2).mean(1))), DB * np.sqrt((e[..., 1] ** 2).mean(1))


def unident_part(D_r, D_t):
    """RMS (over 32 antennas) of the part no method can remove: phase mean + linear ramp (deg), log-gain mean (dB)."""
    raw_ph = np.concatenate([-np.angle(D_r), np.angle(D_t)], 1)
    up = np.concatenate([raw_ph[:, :16] - S.project_impairment(raw_ph[:, :16]), raw_ph[:, 16:] - S.project_impairment(raw_ph[:, 16:])], 1)
    lg = np.concatenate([np.log(np.abs(D_r)), np.log(np.abs(D_t))], 1)
    ug = np.concatenate([np.repeat(lg[:, :16].mean(1, keepdims=True), 16, 1), np.repeat(lg[:, 16:].mean(1, keepdims=True), 16, 1)], 1)
    return np.rad2deg(np.sqrt((up ** 2).mean(1))), DB * np.sqrt((ug ** 2).mean(1))


def true_uv(psi, phi):
    return np.pi * np.cos(psi), np.pi * np.cos(phi)


def stats(ang, psi, phi, L):
    pred = [(ang[0][i], ang[1][i]) for i in range(len(ang[0]))]
    return C.scene_stats(pred, dict(psi=psi, phi=phi, L=L))
