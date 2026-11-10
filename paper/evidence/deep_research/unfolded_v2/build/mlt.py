"""Classical quantisation-aware ML family, batched in torch (same algorithm as finetune/adversarial/mlq.py lm / init_gains /
advc.ml_quant; verified against the numpy original in mlt_check.py / mlt_check.log). No learning.
  starts_classical(Y, bits, L, s_em): 13 classical starts (consistent NOMP, NOMP, consistent parabolic, 10 L-subsets of the top-5 FFT
     maxima after Newton) + the EM start (mlq.em from consistent NOMP, using noise std s_em; omitted when s_em is None).
  lm_t(...): Levenberg-Marquardt on the exact interval-censored likelihood (mlq.lm: lam 1e-2, /3 accept, x4 reject, step clip 0.1).
  ml_multi(...): every start -> convex gain fit (15 it) -> joint LM (40 it) -> lowest NLL per scene.
Variants used as baselines (all classical):
  'ML'          true sigma (13 starts + EM)                     [= advc.ml_quant, the pre-specified quantised ML]
  'annealed ML' true sigma, kappa 3 (15 it) -> 1.7 (15 it) -> 1 (40 it), same starts (G0b definition)
  'ML snr+5' / 'ML snr-5'  sigma mis-specified by -/+5 dB SNR (13 classical starts, no EM start)
  'ML joint sigma' noise std unknown: log sigma estimated jointly (alternating 1-D Newton on log sigma after every LM step, accepted only
     if the exact NLL decreases), init sigma0 = 0.1 x RMS of one real component, 13 classical starts, no EM start; the deployable
     SNR-blind classical pipeline of criterion S1b.
  'ML nominal 10dB' SNR-blind fixed nominal noise std (SNR 10 dB), 13 classical starts (added: joint sigma is unidentifiable at 1 bit).
  'hybrid ML from FT seeds' (not classical): ML (true sigma) started from the ADC-fine-tuned CNN read-out (prior best system)."""
from bcommon import *
import ucore as U_

K64 = None


def kk():
    global K64
    if K64 is None:
        K64 = U_.K_(torch.float64)
    return K64


def nll_q(mu, lo, hi, s):
    isq = torch.ones(mu.shape[0], dtype=torch.bool)
    return U_.nll_terms_t(mu, lo, hi, s, isq)


def lm_t(U, V, X, lo, hi, s, iters=40, which='all', clip=0.1, lam0=1e-2):
    k = kk(); R, K = U.shape
    lam = torch.full((R,), lam0, dtype=torch.float64)
    mu, J = U_.jac_t(U, V, X, k); f, g, h = nll_q(mu, lo, hi, s); F0 = f.sum(1)
    n = 4 * K if which == 'all' else 2 * K
    eye = torch.eye(n, dtype=torch.float64)[None]
    for _ in range(iters):
        Js = J if which == 'all' else J[:, :, 2 * K:]
        Am = torch.einsum('bik,bi,bil->bkl', Js, h, Js); gr = torch.einsum('bik,bi->bk', Js, g)
        D = torch.diagonal(Am, dim1=1, dim2=2)
        d = -torch.linalg.solve(Am + (lam[:, None] * (D + 1e-9))[:, :, None] * eye, gr[..., None])[..., 0]
        if which == 'all':
            Un = U + torch.clamp(d[:, :K], -clip, clip); Vn = V + torch.clamp(d[:, K:2 * K], -clip, clip)
            Xn = X + torch.complex(d[:, 2 * K:3 * K], d[:, 3 * K:])
        else:
            Un, Vn, Xn = U, V, X + torch.complex(d[:, :K], d[:, K:])
        mun, Jn = U_.jac_t(Un, Vn, Xn, k); fn, gn, hn = nll_q(mun, lo, hi, s); Fn = fn.sum(1)
        ok = Fn < F0; o1 = ok[:, None]
        U = torch.where(o1, Un, U); V = torch.where(o1, Vn, V); X = torch.where(o1, Xn, X)
        J = torch.where(ok[:, None, None], Jn, J); g = torch.where(o1, gn, g); h = torch.where(o1, hn, h); F0 = torch.where(ok, Fn, F0)
        lam = torch.clamp(torch.where(ok, lam / 3, lam * 4), 1e-7, 1e7)
    return U, V, X, F0


def lm_joint_sigma(U, V, X, lo, hi, s0, iters=40, clip=0.1):
    """LM on (u, v, x) alternating with a damped 1-D Newton step on log sigma (finite-difference curvature), accept-if-better."""
    k = kk(); ls = torch.log(s0[:, 0])
    def F(U, V, X, ls):
        return nll_q(U_.mu_t(U, V, X, k), lo, hi, torch.exp(ls)[:, None])[0].sum(1)
    for it in range(iters // 4):
        U, V, X, F0 = lm_t(U, V, X, lo, hi, torch.exp(ls)[:, None], iters=4, clip=clip, lam0=1e-2 if it == 0 else 1e-3)
        e = 0.05
        Fp, Fm = F(U, V, X, ls + e), F(U, V, X, ls - e)
        gq = (Fp - Fm) / (2 * e); hq = (Fp - 2 * F0 + Fm) / e ** 2
        step = torch.where(hq > 0, -gq / hq.clamp_min(1e-9), -torch.sign(gq) * 0.5)
        step = torch.clamp(step, -0.5, 0.5)
        ln = ls + step; Fn = F(U, V, X, ln)
        ls = torch.where(Fn < F0, ln, ls)
    return U, V, X, F(U, V, X, ls), torch.exp(ls)


def init_gains_t(U, V, Yq, lo, hi, s):
    k = kk()
    H = U_.to_ant_t(torch.from_numpy(Yq), k)
    X, _ = U_.ls_gains_t(H, U, V, k)
    return lm_t(U, V, X, lo, hi, s, iters=15, which='gain')[2]


def ml_multi(Yq, lo, hi, s, starts, iters=40, schedule=None):
    """starts: list of (U, V) numpy (B, L). s: (B,1) numpy std. Returns U, V (B,L) numpy of the lowest-NLL start, NLL, index."""
    B = len(Yq); S = len(starts)
    Uc = torch.from_numpy(np.concatenate([u for u, _ in starts])); Vc = torch.from_numpy(np.concatenate([v for _, v in starts]))
    rep = lambda a: torch.from_numpy(np.concatenate([a] * S))
    lo_t, hi_t = rep(lo), rep(hi); Yr = np.concatenate([Yq] * S)
    st = torch.from_numpy(np.concatenate([s] * S))
    if schedule is None:
        X = init_gains_t(Uc, Vc, Yr, lo_t, hi_t, st)
        Uo, Vo, Xo, F = lm_t(Uc, Vc, X, lo_t, hi_t, st, iters=iters)
    else:
        X = init_gains_t(Uc, Vc, Yr, lo_t, hi_t, schedule[0][0] * st)
        Uo, Vo, Xo = Uc, Vc, X
        for kap, it in schedule:
            Uo, Vo, Xo, F = lm_t(Uo, Vo, Xo, lo_t, hi_t, kap * st, iters=it)
    F = F.reshape(S, B); i = F.argmin(0); ar = torch.arange(B)
    Uo = Uo.reshape(S, B, -1)[i, ar]; Vo = Vo.reshape(S, B, -1)[i, ar]
    return Uo.numpy(), Vo.numpy(), F[i, ar].numpy(), i.numpy()


def starts_classical(Yq, bits, L, s_em=None, K=5):
    H = A.to_ant(Yq); st = []
    Uc, Vc, _ = A.consistent_nomp(Yq, bits, L, 8); st.append((Uc, Vc))
    st.append(C.nomp_greedy(H, L, 4, 3))
    ps = A.parabolic_heat_decode(A.fft_heat(H, 16), L)
    U0 = np.array([np.pi * np.cos(p[0][:L]) for p in ps]); V0 = np.array([np.pi * np.cos(p[1][:L]) for p in ps])
    st.append(A.consistent_nomp(Yq, bits, L, 8, U=U0, V=V0)[:2])
    Uk, Vk = mlq.fft_cands(H, K, 4)
    Uk, Vk = C.newton(H[:, None].repeat(K, 1).reshape(-1, 16, 16), Uk.ravel(), Vk.ravel()); Uk = Uk.reshape(-1, K); Vk = Vk.reshape(-1, K)
    for sb in itertools.combinations(range(K), L):
        st.append((Uk[:, list(sb)], Vk[:, list(sb)]))
    em = None
    if s_em is not None:
        lo, hi = mlq.cells_ri(Yq, bits, A.adc_cells)
        X = mlq.init_gains(MO, Uc, Vc, Yq, lo, hi, s_em); Ue, Ve, _, _ = mlq.em(MO, Uc, Vc, X, lo, hi, s_em, L, iters=6); em = (Ue, Ve)
    return st, em


def ml_family(Yq, bits, snr, L, ft_seeds=None):
    """all classical ML variants on one chunk (B scenes, same bits). Returns dict name -> (U, V)."""
    lo, hi = mlq.cells_ri(Yq, bits, A.adc_cells); s = sigma_of(snr)[:, None] * np.ones((len(Yq), 1))
    st, em = starts_classical(Yq, bits, L, s)
    out = {}
    U, V, _, _ = ml_multi(Yq, lo, hi, s, st + [em]); out['ML'] = (U, V)
    U, V, _, _ = ml_multi(Yq, lo, hi, s, st + [em], schedule=[(3.0, 15), (1.7, 15), (1.0, 40)]); out['annealed ML'] = (U, V)
    for nm, dsnr in [('ML snr+5', 5.0), ('ML snr-5', -5.0)]:
        U, V, _, _ = ml_multi(Yq, lo, hi, sigma_of(snr + dsnr)[:, None] * np.ones((len(Yq), 1)), st); out[nm] = (U, V)
    # joint sigma (SNR-blind): every classical start, LM + log-sigma Newton, lowest NLL
    rms = np.sqrt((np.abs(Yq) ** 2).mean((1, 2)) / 2)[:, None]; s0 = 0.1 * rms
    S = len(st); B = len(Yq)
    Uc = torch.from_numpy(np.concatenate([u for u, _ in st])); Vc = torch.from_numpy(np.concatenate([v for _, v in st]))
    lo_t = torch.from_numpy(np.concatenate([lo] * S)); hi_t = torch.from_numpy(np.concatenate([hi] * S)); s0t = torch.from_numpy(np.concatenate([s0] * S))
    X = init_gains_t(Uc, Vc, np.concatenate([Yq] * S), lo_t, hi_t, s0t)
    Uo, Vo, Xo, F, sg = lm_joint_sigma(Uc, Vc, X, lo_t, hi_t, s0t)
    F = F.reshape(S, B); i = F.argmin(0); ar = torch.arange(B)
    out['ML joint sigma'] = (Uo.reshape(S, B, -1)[i, ar].numpy(), Vo.reshape(S, B, -1)[i, ar].numpy())
    U, V, _, _ = ml_multi(Yq, lo, hi, sigma_of(np.full(len(Yq), 10.0))[:, None], st); out['ML nominal 10dB'] = (U, V)
    if ft_seeds is not None:
        U, V, _, _ = ml_multi(Yq, lo, hi, s, [ft_seeds]); out['hybrid ML from FT seeds'] = (U, V)
    return out
