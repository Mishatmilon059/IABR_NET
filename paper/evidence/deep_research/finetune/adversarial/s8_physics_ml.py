"""Step 3 follow-up: physics-model classical estimators for the two imperfection families where the imp-FT 'learned seeds' path won on
new seeds in s4 (M6 per-measurement phase noise, M2 antenna-position error). No learning, nothing tuned: every constant is the known
impairment level (sigma_phi for M6, sigma_pos for M2) or fixed a priori (30 LM iterations, step clip 0.1).
  PN-ML  (M6): Gaussian likelihood with the exact second moments of Y = mu e^{j Phi} + n, Phi ~ N(0, s^2): per sample, residual split
               into the component along the model value m (radial, var sn2 + c_r |m|^2) and across it (tangential, var sn2 + c_t |m|^2),
               c_r = (1 - e^{-s^2})^2 / (2 e^{-s^2}), c_t = (1 - e^{-2 s^2}) / (2 e^{-s^2}) (mean shrink e^{-s^2/2} absorbed in the gains);
               Levenberg-Marquardt on (u, v, gains), Gauss-Newton weights from the current m, steps accepted on the exact objective.
  POS-MAP (M2): [main version: position errors restricted to the complement of {constant, ramp} of each array, since those two
               components are absorbed by the path gains / angles; the first version with full d is also reported - it fell below NOMP (0.571 vs 0.600)
               on the SMOKE seed 9435, which motivated the restriction; no test seed was looked at]  joint LM on (u, v, gains, per-element position errors d_r (16), d_t (16)) with the model of stress_sim.generate(M2)
               (x_n = n + d_n on both arrays) and a Gaussian prior d ~ N(0, sigma_pos^2); white noise of known variance.
  Starts (classical): NOMP (os4 rc3), NOMP os8 rc6, FFT-peak seeds (top-3 4x FFT maxima -> Newton -> cyclic rc3), multi-start K6;
  'all starts' keeps, per scene, the lowest objective (no ground truth).  Robustness: same with the impairment level mis-specified
  (x0.5).  Hybrid (reported separately, not classical): the same estimator started from the imp-FT learned seeds.
Learned reference = 'FT seeds on Ht+SC' exactly as s4_imp.py (imp_ft.pt).  FRESH seeds 9415 (15 dB) / 9425 (0 dB), 1000 scenes,
SMOKE seed 9435 (no tuning done on it, sanity only). All methods on the same Y, true L = 3. Paired bootstrap 2000 within SNR, seed 0.
usage: python s8_physics_ml.py  (stdout -> s8_physics_ml.log; writes s8_physics_ml.json; per-chunk cache in PCACHE)"""
from advc import *
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/finetune/imp')
import icom as I                        # noqa: E402

SMOKE = int(os.environ.get('SMOKE', 0))
SEEDS = {15: 9415, 0: 9425}
N, CH, L, RC = (SMOKE if SMOKE else 1000), 200, 3, 3
SFX = '_smoke' if SMOKE else ''
CELLS = [('M6', 30.0, 15), ('M2', 0.2, 15), ('M6', 20.0, 15), ('M6', 30.0, 0)]
ft = I.load_cnn(I.FT_CKPT)
WH, FT_ = MO.WH, MO.FT
_n = np.arange(16.)


def heat_ref(H, heat):
    Uh, Vh = C.g['heat_peaks'](heat, L)
    Uh, Vh = C.newton(H[:, None].repeat(L, 1).reshape(-1, 16, 16), Uh.ravel(), Vh.ravel())
    return C.cyclic(H, Uh.reshape(-1, L), Vh.reshape(-1, L), RC)


def scd(H, UV):
    return C.uv_to_angles(*C.selfcal(H, UV[0], UV[1], *A.SC))


def sct(H, UV):
    return C.uv_to_angles(*C.selfcal(H, UV[0], UV[1], 1.0, 2.0, rc=2, iters=4))


# ---------------------------------------------------------------- M6 phase-noise ML
def pn_eval(U, V, X, y, sn2, cr, ct):
    mu, J = MO.jac(U, V, X)
    m = mu[:, :256] + 1j * mu[:, 256:]
    e = np.exp(-1j * np.angle(m)); r = (y - m) * e; a2 = np.abs(m) ** 2
    vr = sn2 + cr * a2; vt = sn2 + ct * a2
    F = 0.5 * (r.real ** 2 / vr + r.imag ** 2 / vt + np.log(vr) + np.log(vt)).sum(1)
    Jc = (J[:, :256] + 1j * J[:, 256:]) * e[:, :, None]
    Am = np.einsum('bik,bi,bil->bkl', Jc.real, 1 / vr, Jc.real) + np.einsum('bik,bi,bil->bkl', Jc.imag, 1 / vt, Jc.imag)
    g = np.einsum('bik,bi->bk', Jc.real, r.real / vr) + np.einsum('bik,bi->bk', Jc.imag, r.imag / vt)
    return F, Am, g


def lm_pn(U, V, y, sn2, sphi, iters=30, clip=0.1):
    q = np.exp(-sphi ** 2)
    cr = (1 - q) ** 2 / (2 * q); ct = (1 - q ** 2) / (2 * q)
    H = A.to_ant(y.reshape(-1, 16, 16)); X = C.ls_gains(H, U, V)[0]
    U, V = U.copy(), V.copy(); B, K = U.shape; lam = np.full(B, 1e-2); eye = np.eye(4 * K)[None]
    F0, Am, g = pn_eval(U, V, X, y, sn2, cr, ct)
    for _ in range(iters):
        D = np.einsum('bkk->bk', Am)
        d = np.linalg.solve(Am + (lam[:, None] * (D + 1e-9))[:, :, None] * eye, g[..., None])[..., 0]
        Un = U + np.clip(d[:, :K], -clip, clip); Vn = V + np.clip(d[:, K:2 * K], -clip, clip); Xn = X + d[:, 2 * K:3 * K] + 1j * d[:, 3 * K:]
        Fn, An, gn = pn_eval(Un, Vn, Xn, y, sn2, cr, ct); ok = Fn < F0
        U = np.where(ok[:, None], Un, U); V = np.where(ok[:, None], Vn, V); X = np.where(ok[:, None], Xn, X)
        Am = np.where(ok[:, None, None], An, Am); g = np.where(ok[:, None], gn, g); F0 = np.where(ok, Fn, F0)
        lam = np.clip(np.where(ok, lam / 3, lam * 4), 1e-7, 1e7)
    return U, V, F0


# ---------------------------------------------------------------- M2 position-error MAP
def pos_jac(U, V, X, dr, dt):
    xr = _n + dr; xt = _n + dt
    r = np.exp(-1j * U[..., None] * xr[:, None, :]); t = np.exp(1j * V[..., None] * xt[:, None, :])
    rho = np.einsum('qn,bkn->bkq', WH, r); tau = np.einsum('pm,bkm->bkp', FT_, t)
    drho = np.einsum('qn,bkn->bkq', WH, -1j * xr[:, None, :] * r); dtau = np.einsum('pm,bkm->bkp', FT_, 1j * xt[:, None, :] * t)
    At = rho[:, :, :, None] * tau[:, :, None, :]
    mu = np.einsum('bk,bkqp->bqp', X, At)
    Ju = X[:, :, None, None] * drho[:, :, :, None] * tau[:, :, None, :]
    Jv = X[:, :, None, None] * rho[:, :, :, None] * dtau[:, :, None, :]
    Ar = np.einsum('bk,bkn,bkp->bnp', X * (-1j * U), r, tau); Jdr = np.einsum('qn,bnp->bnqp', WH, Ar)
    Atq = np.einsum('bk,bkq,bkm->bmq', X * (1j * V), rho, t); Jdt = np.einsum('pm,bmq->bmqp', FT_, Atq)
    B = len(U)
    Jc = np.concatenate([Ju, Jv, At, 1j * At, Jdr, Jdt], 1).reshape(B, -1, 256)
    J = np.concatenate([Jc.real, Jc.imag], 2).transpose(0, 2, 1)
    m = mu.reshape(B, 256)
    return np.concatenate([m.real, m.imag], 1), J


_Bq = np.linalg.qr(np.stack([np.ones(16), _n - 7.5], 1), mode='complete')[0][:, 2:]   # (16,14) basis orthogonal to {1, n}
QR = np.zeros((32, 28)); QR[:16, :14] = _Bq; QR[16:, 14:] = _Bq


def pos_eval(U, V, X, dr, dt, yri, sn2, sp2, noramp=False):
    """noramp: position errors restricted to the complement of {constant, linear ramp} per array (those two components are not
    identifiable separately from the path gains / angles); parameters z (28) with d = QR z, prior z ~ N(0, sp2)."""
    K = U.shape[1]
    if noramp:
        z = np.concatenate([dr, dt], 1); d = z @ QR.T; mu, J = pos_jac(U, V, X, d[:, :16], d[:, 16:])
        J = np.concatenate([J[:, :, :4 * K], J[:, :, 4 * K:] @ QR], 2); dd = z
    else:
        mu, J = pos_jac(U, V, X, dr, dt); dd = np.concatenate([dr, dt], 1)
    r = yri - mu; P = dd.shape[1]
    F = 0.5 * (r ** 2).sum(1) / sn2[:, 0] + 0.5 * (dd ** 2).sum(1) / sp2
    Am = np.einsum('bik,bil->bkl', J, J) / sn2[:, :, None]
    Am[:, 4 * K:, 4 * K:] += np.eye(P)[None] / sp2
    g = np.einsum('bik,bi->bk', J, r) / sn2
    g[:, 4 * K:] -= dd / sp2
    return F, Am, g


def lm_pos(U, V, y, sn2, spos, iters=30, clip=0.1, noramp=False):
    sp2 = spos ** 2; yri = np.concatenate([y.real, y.imag], 1)
    H = A.to_ant(y.reshape(-1, 16, 16)); X = C.ls_gains(H, U, V)[0]
    nd = 14 if noramp else 16
    U, V = U.copy(), V.copy(); B, K = U.shape; dr = np.zeros((B, nd)); dt = np.zeros((B, nd))
    lam = np.full(B, 1e-2); P = 4 * K + 2 * nd; eye = np.eye(P)[None]
    F0, Am, g = pos_eval(U, V, X, dr, dt, yri, sn2, sp2, noramp)
    for _ in range(iters):
        D = np.einsum('bkk->bk', Am)
        d = np.linalg.solve(Am + (lam[:, None] * (D + 1e-9))[:, :, None] * eye, g[..., None])[..., 0]
        Un = U + np.clip(d[:, :K], -clip, clip); Vn = V + np.clip(d[:, K:2 * K], -clip, clip); Xn = X + d[:, 2 * K:3 * K] + 1j * d[:, 3 * K:4 * K]
        drn = dr + np.clip(d[:, 4 * K:4 * K + nd], -clip, clip); dtn = dt + np.clip(d[:, 4 * K + nd:], -clip, clip)
        Fn, An, gn = pos_eval(Un, Vn, Xn, drn, dtn, yri, sn2, sp2, noramp); ok = Fn < F0
        w = ok[:, None]
        U = np.where(w, Un, U); V = np.where(w, Vn, V); X = np.where(w, Xn, X); dr = np.where(w, drn, dr); dt = np.where(w, dtn, dt)
        Am = np.where(ok[:, None, None], An, Am); g = np.where(w, gn, g); F0 = np.where(ok, Fn, F0)
        lam = np.clip(np.where(ok, lam / 3, lam * 4), 1e-7, 1e7)
    return U, V, F0


def pick3(c):
    Fm = np.stack([x[2] for x in c], 1); k = Fm.argmin(1); i = np.arange(len(k))
    return np.stack([x[0] for x in c], 1)[i, k], np.stack([x[1] for x in c], 1)[i, k], k


def run(Y, kind, sev, snr):
    H = SS.to_ant(Y); P = {}; B = len(Y); y = Y.reshape(B, 256)
    g = C.nomp_greedy(H, L, 4, RC); g8 = C.nomp_greedy(H, L, 8, 6); fh = heat_ref(H, A.fft_heat(H, 4)); ms = A.multistart(H, L, 6)[:2]
    P['NOMP'] = C.uv_to_angles(*g); P['NOMP os8 rc6'] = C.uv_to_angles(*g8); P['NOMP+SC tuned'] = sct(H, g)
    P['FFT-peak seeds+SC'] = scd(H, fh); P['multi-start K6'] = C.uv_to_angles(*ms)
    hf, _, _ = I.net_forward(ft, Y); ls = heat_ref(H, hf)
    P['FT seeds on Ht+SC'] = scd(H, ls)
    sn2 = (sigma_of(snr) ** 2)[:, None]
    if kind == 'M6':
        est = lambda U, V, f: lm_pn(U, V, y, sn2, np.deg2rad(sev) * f)
    else:
        est = lambda U, V, f: lm_pos(U, V, y, sn2, sev * f, noramp=True)
    tag = 'PN-ML' if kind == 'M6' else 'POS-MAP'
    t0 = time.time(); tsec = 0.0
    for f, sfx in [(1.0, ''), (0.5, ' (level x0.5)')]:
        res = [est(U, V, f) for U, V in [g, g8, fh, ms]]
        Ub, Vb, k = pick3(res)
        P[f'{tag} all starts{sfx}'] = C.uv_to_angles(Ub, Vb)
        if f == 1.0:
            tsec = time.time() - t0
            P[f'{tag} from NOMP'] = C.uv_to_angles(res[0][0], res[0][1])
            r = est(ls[0], ls[1], 1.0); P[f'{tag} from learned seeds'] = C.uv_to_angles(r[0], r[1])
            Uh, Vh, _ = pick3(res + [r]); P[f'{tag} all starts + learned seeds'] = C.uv_to_angles(Uh, Vh)
            if kind == 'M2':     # first (a-priori) version: full 16+16 position errors (ramp included)
                rf = [lm_pos(U, V, y, sn2, sev) for U, V in [g, g8, fh, ms]]; Uf, Vf, _ = pick3(rf)
                P['POS-MAP full d (with ramp) all starts'] = C.uv_to_angles(Uf, Vf)
    return P, tsec


out = {}
jp = f'{HERE}/s8_physics_ml{SFX}.json'
if os.path.exists(jp):
    out = json.load(open(jp))
for kind, sev, snr in CELLS:
    name = f'{kind}_{SS.sev_label(kind, sev)}_snr{snr}'
    if name in out:
        print('skip', name, flush=True); continue
    t0 = time.time(); seed = 9435 if SMOKE else SEEDS[snr]
    base = SS.base_scenes(N, 3, float(snr), seed); Y = SS.generate(base, kind, sev)
    S = {}; T = 0.0
    for c0 in range(0, N, CH):
        f = f'{PCACHE}/s8_{name}_{c0}{SFX}.npz'
        sl = slice(c0, c0 + CH); bc = {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim and len(v) == N else v) for k, v in base.items()}
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True); Sc = {k: z[k] for k in z.files if k != 'T'}; tc = float(z['T'])
        else:
            P, tc = run(Y[sl], kind, sev, bc['snr'])
            Sc = {m: A.stats(v, bc) for m, v in P.items()}
            np.savez(f, T=tc, **Sc)
            print(f'  {name} chunk {c0} done [{time.time() - t0:.0f}s]', flush=True)
        for m, v in Sc.items():
            S.setdefault(m, []).append(v)
        T += tc
    S = {m: np.concatenate(v) for m, v in S.items()}
    tag = 'PN-ML' if kind == 'M6' else 'POS-MAP'
    cls = ['NOMP', 'NOMP os8 rc6', 'NOMP+SC tuned', 'FFT-peak seeds+SC', 'multi-start K6', f'{tag} all starts', f'{tag} from NOMP']
    pd0 = {m: A.pd(S[m]) for m in S}; best = max(cls, key=lambda m: pd0[m]); bestold = max(cls[:5], key=lambda m: pd0[m])
    LS_ = 'FT seeds on Ht+SC'
    pairs = [(LS_, best), (LS_, bestold), (f'{tag} all starts', bestold), (f'{tag} all starts (level x0.5)', bestold),
             (LS_, f'{tag} all starts (level x0.5)'), (f'{tag} from learned seeds', f'{tag} all starts'),
             (f'{tag} all starts + learned seeds', f'{tag} all starts'), (f'{tag} all starts + learned seeds', LS_)]
    pairs = list(dict.fromkeys((a, b) for a, b in pairs if a != b))
    pdm, ci_, D = boot(S, base['snr'], pairs)
    out[name] = dict(pd=pdm, ci=ci_, diffs=D, best_classical=best, best_classical_without_physics=bestold, seed=seed, n=N,
                     physics_sec_per_scene_all_starts_6thr=T / N)
    print(f'=== {name} (n={N}, seed {seed}) best classical {best} {pdm[best]:.4f}; without the physics estimator {bestold} {pdm[bestold]:.4f} '
          f'[{time.time() - t0:.0f}s; physics estimator {T / N * 1000:.0f} ms/scene, 6 threads, batch {CH}]', flush=True)
    for m in S:
        print(f'   {m:40s} Pd {pdm[m]:.4f} [{ci_[m][0]:.4f},{ci_[m][1]:.4f}]', flush=True)
    for k, d in D.items():
        print(f'   {k:70s} {fmt(d)}', flush=True)
    json.dump(out, open(jp, 'w'), indent=1, default=float)
print('done', flush=True)
