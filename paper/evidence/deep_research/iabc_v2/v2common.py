"""Shared code for iabc_v2/. Nothing outside iabc_v2/ is modified.
Re-uses (imports, read-only) ../iabc/icommon.py, which execs the notebook sources:
  decoder (nomp_greedy, cyclic, selfcal, ...)  notebook_src.py [527:647] via new_results/common.py
  CNN + old IABC                               notebook_src.py [419:522], weights outputs/checkpoints/IABRv2/final.pt (read only)
  simulator                                    iabr2_sim.py
Threads: OMP/OPENBLAS/MKL = 6, torch 6 (set in icommon before numpy/torch import)."""
import os, sys
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = '6'
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc')
from icommon import *          # noqa  (np, torch, nn, TF, C, S, cnn, net, selfcal_ret, ident_err, rms_deg_db, heat_seed, stats, to_ant, ...)
import icommon as IC

V2 = os.path.dirname(os.path.abspath(__file__))
V2CACHE = f'{V2}/cache'
os.makedirs(V2CACHE, exist_ok=True)
SNRS = [-10, -5, 0, 5, 10, 15, 20, 25]
NC = np.arange(16.) - 7.5


def proj_np(x):
    return S.project_impairment(x)


def ident_vec(ph, lg):
    """(B,32) phase rad, (B,32) log-gain -> identifiable (B,32,2): phase projected per array, gain mean-removed per array."""
    p = np.concatenate([proj_np(ph[:, :16]), proj_np(ph[:, 16:])], 1)
    g = np.concatenate([lg[:, :16] - lg[:, :16].mean(1, keepdims=True), lg[:, 16:] - lg[:, 16:].mean(1, keepdims=True)], 1)
    return np.stack([p, g], -1)


def apply_corr(Ht, est):
    """same convention as IABRv2.front: Hc = Ht * c_r * c_t, c = exp(-lg - j ph), est (B,32,2) [phase, log-gain]."""
    c = np.exp(-est[..., 1] - 1j * est[..., 0])
    return Ht * c[:, :16, None] * c[:, None, 16:]


def full_true_corr(Ht, D_r, D_t):
    """remove the FULL distortion (incl. unidentifiable mean / linear ramp): Ht ~ conj(D_r) H D_t -> divide."""
    return Ht / np.conj(D_r)[:, :, None] / D_t[:, None, :]


def make_scenes(seed, n, L, snr, pdeg, gdb, Lvar=None):
    """Test-scene generator (same structure as iabc/s2_sweep.py): angles/alpha from default_rng(seed), noise from [seed,1],
    impairment uniforms from [seed,3]. Lvar: optional array of per-scene L (else constant L)."""
    rng = np.random.default_rng(seed)
    Ls = np.full(n, L) if Lvar is None else Lvar
    psi, phi, alpha = S.sample_paths(Ls, rng)
    g = np.random.default_rng([seed, 1]); s = np.sqrt(10 ** (-snr / 10) / 2)
    noise = s * (g.standard_normal((n, 16, 16)) + 1j * g.standard_normal((n, 16, 16)))
    o = S.simulate(psi, phi, alpha, float(snr), np.random.default_rng([seed, 3]), phase_deg_max=pdeg, gain_db_max=gdb, noise=noise)
    return dict(psi=psi, phi=phi, alpha=alpha, L=Ls, snr=np.full(n, float(snr)), Y=o['Y'], D_r=o['D_r'], D_t=o['D_t'], H=o['H'],
                tgt=S.impairment_targets(o['D_r'], o['D_t']).astype(np.float64))


def err_rms(tgt, est):
    """per-scene identifiable RMS error (phase deg, gain dB) and per-scene MSE (rad^2, neper^2)."""
    e = ident_err(tgt, est)
    return np.rad2deg(np.sqrt((e[..., 0] ** 2).mean(1))), DB * np.sqrt((e[..., 1] ** 2).mean(1)), (e[..., 0] ** 2).mean(1), (e[..., 1] ** 2).mean(1)


def boot_removed(mse_before, mse_after, snr, B=2000, seed=0):
    """% of pooled MSE removed, paired bootstrap within SNR groups -> (point, lo, hi)."""
    rng = np.random.default_rng(seed); snr = snr.astype(int); vals = np.zeros(B); nb = np.zeros(B); na = np.zeros(B)
    for s in sorted(set(snr.tolist())):
        ix = np.where(snr == s)[0]; R = rng.integers(0, len(ix), (B, len(ix)))
        nb += mse_before[ix][R].sum(1); na += mse_after[ix][R].sum(1)
    vals = 100 * (1 - na / np.maximum(nb, 1e-30))
    pt = 100 * (1 - mse_after.sum() / max(mse_before.sum(), 1e-30))
    return pt, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def hybrid_inputs(Ht, L, eA, iters=4):
    """IABC-v2b stage-2 inputs. H1 = Ht corrected by the blind estimate eA; NOMP(H1, L); un-shrunk gate-open ALS (selfcal_ret,
    lam 1, tau 0, 4 iterations, cyclic angle refinement) on H1 -> e_ALS (relative to H1); model fit on H1 corrected by e_ALS gives the
    SNR estimate and per-antenna residual / model energies. Returns tok (B,32,6), glob (B,3), and the NOMP angles on H1 (reused)."""
    H1 = apply_corr(Ht, eA)
    U, V = C.nomp_greedy(H1, L, 4, 3)
    U2, V2, eALS, _, _ = selfcal_ret(H1, U, V, 1.0, 0.0, iters=iters)
    H2 = apply_corr(H1, eALS)
    x, A = C.ls_gains(H2, U2, V2); Hm = np.einsum('bik,bk->bi', A, x).reshape(H2.shape); R = H2 - Hm
    er = np.abs(R) ** 2; em = np.abs(Hm) ** 2
    rr = np.concatenate([er.sum(2), er.sum(1)], 1); mr = np.concatenate([em.sum(2), em.sum(1)], 1)
    rr = np.log(rr / (rr.mean(1, keepdims=True) + 1e-12) + 1e-6); mr = np.log(mr / (mr.mean(1, keepdims=True) + 1e-12) + 1e-6)
    snr_est = em.sum((1, 2)) / (er.sum((1, 2)) + 1e-12)
    tok = np.concatenate([eA, eA + eALS, rr[..., None], mr[..., None]], -1).astype(np.float32)
    glob = np.stack([np.log10(snr_est + 1e-6), np.full(len(Ht), L / 9.0), np.log10(er.sum((1, 2)) / (np.abs(H1) ** 2).sum((1, 2)) + 1e-9)], 1).astype(np.float32)
    return tok, glob, (U, V)


def toeplitz_est(Ht, iters=2):
    """Classical blind structure-based estimator (no angles). For an ideal ULA with (nearly) uncorrelated paths, R_r = Ht Ht^H is
    Toeplitz; per-antenna distortion d breaks it: R[n,k] = d_n conj(d_k) T[n-k].
      gain : weighted LS of log|R[n,k]| = a_n + a_k + t_|n-k|  (weights |R[n,k]|)
      phase: shift-invariance, z = R[n+1,k+1] conj(R[n,k]) = |T|^2 exp(j(delta_n - delta_k)), delta_n = b_{n+1} - b_n;
             weighted LS of angle(z) over pairs n<k (weights |z|), b = cumsum(delta), projected.
    Same on Ht^H for the Tx side (phase sign flipped back). Iterated `iters` times (correct, re-estimate, accumulate).
    Returns identifiable estimate (B,32,2) in the IABC convention."""
    B = len(Ht); est = np.zeros((B, 32, 2))
    n = np.arange(16); NN, KK = np.meshgrid(n, n, indexing='ij')
    Dg = np.zeros((256, 32)); Dg[np.arange(256), NN.ravel()] += 1; Dg[np.arange(256), KK.ravel()] += 1
    Dg[np.arange(256), 16 + np.abs(NN - KK).ravel()] += 1
    pr = [(i, k) for i in range(15) for k in range(i + 1, 15)]
    Dp = np.zeros((len(pr), 15))
    for r_, (i, k) in enumerate(pr):
        Dp[r_, i] = 1; Dp[r_, k] = -1
    ii = np.array([p[0] for p in pr]); kk = np.array([p[1] for p in pr])

    def side(M):
        R = M @ np.conj(np.transpose(M, (0, 2, 1)))
        w = np.abs(R).reshape(B, 256) + 1e-12; y = np.log(w)
        Aw = Dg[None] * w[..., None]
        G = np.transpose(Aw, (0, 2, 1)) @ Dg[None] + 1e-6 * np.eye(32)[None]
        sol = np.linalg.solve(G, np.transpose(Aw, (0, 2, 1)) @ y[..., None])[..., 0]
        a = sol[:, :16]; a = a - a.mean(1, keepdims=True)
        z = R[:, ii + 1, kk + 1] * np.conj(R[:, ii, kk]); wz = np.abs(z) + 1e-12; ph = np.angle(z)
        Aw = Dp[None] * wz[..., None]
        G = np.transpose(Aw, (0, 2, 1)) @ Dp[None] + 1e-6 * np.eye(15)[None]
        dl = np.linalg.solve(G, np.transpose(Aw, (0, 2, 1)) @ ph[..., None])[..., 0]
        b = np.concatenate([np.zeros((B, 1)), np.cumsum(dl, 1)], 1)
        return proj_np(b), a

    H = Ht.copy()
    for _ in range(iters):
        br, ar = side(H); bt, at = side(np.conj(np.transpose(H, (0, 2, 1))))
        e = np.stack([np.concatenate([br, -bt], 1), np.concatenate([ar, at], 1)], -1)
        est = ident_vec(est[..., 0] + e[..., 0], est[..., 1] + e[..., 1])
        H = apply_corr(Ht, est)
    return est


def trunk_heat(Yc, bs=250):
    """IABR-Net trunk + heat head applied to a given (already corrected) beamspace Yc -- identical to IABRv2.forward after front()
    (notebook lines 497-520); used to swap a different corrector into IABR-Net with the trunk frozen. Returns sigmoid heat."""
    out = []
    with torch.no_grad():
        for s in range(0, len(Yc), bs):
            Y = torch.from_numpy(Yc[s:s + bs].astype(np.complex64))
            Yn = Y / (Y.abs().pow(2).mean((1, 2), keepdim=True).sqrt() + 1e-8)
            x = torch.stack([Yn.real, Yn.imag, torch.log(Yn.abs() ** 2 + 1e-3)], 1)
            x = torch.relu(cnn.stem(TF.pad(x, (1, 1, 1, 1), mode='circular')))
            for b in cnn.blocks:
                x = b(x)
            h = TF.pixel_shuffle(cnn.head2(torch.relu(cnn.head1(x))), cnn.r)[:, 0]
            out.append(torch.sigmoid(h).double().numpy())
    return np.concatenate(out)


def to_beam(Hc):
    """inverse of to_ant: Y = W^H Hc F."""
    return np.matmul(np.matmul(S.W_IDEAL.conj().T[None], Hc), S.F_IDEAL[None])
