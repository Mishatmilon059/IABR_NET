"""Independent re-check of M7 (unknown number of paths).
 A. Consistency on the cached M7 test banks (stress/cache/m7_test_snr*.npz): recompute L_hat of rules (a), (a'), blind CFAR and MDL from the
    stored detection statistics T / residual energies E with my own code, compare with the stored L_hat; recompute P(L_hat = L), recall,
    precision from the stored per-scene arrays and compare with m7_order.log.
 B. FRESH banks (1000 scenes per SNR, seeds 9600 / 9615, L ~ U{1..8} from default_rng([seed, 7]), my own draw scheme from adv_common) with
    my own step-wise NOMP loop (checked against notebook nomp_greedy), the constants FROZEN from the original run (t* = 0.45, kappa_b = 13.0,
    kappa_MC = 12.32 -- also re-derived here by Monte Carlo with a different seed), and my own matching: maximum-cardinality assignment of
    predicted to true paths where a pair counts only if |d psi| <= 1 deg AND |d phi| <= 1 deg (linear_sum_assignment on -ok).
    Also variant (a16): analytic kappa = 10.145 applied to the critical 16x16 grid statistic (the grid for which the formula is exact).
 Paired bootstrap 2000, seed 0.  usage: python m7_check.py > m7_check.log"""
import scipy.optimize
from adv_common import *

ST = 'D:/ai_ml_project/paper/evidence/deep_research/stress'
R = json.load(open(f'{ST}/m7_results.json')); LMAX = 10
KAPPA = -np.log(1 - (1 - 1e-2) ** (1 / 256)); KMC = R['kappa_mc']; KB = R['kappa_blind']; TH = R['t_heat']
print(f'frozen constants from m7_results.json: kappa_mc {KMC:.4f}, kappa_blind {KB}, t_heat {TH}; my analytic kappa {KAPPA:.4f}')


def first_below(T, thr):
    b = T < thr[:, None]
    return np.where(b.any(1), b.argmax(1), LMAX)


print('A. cached test banks')
for snr in [15, 0]:
    z = np.load(f'{ST}/cache/m7_test_snr{snr}.npz'); L = z['L']; T = z['T']; E = z['E']; s2 = 10 ** (-snr / 10); n = len(L)
    mine = {'NOMP CFAR (a) oracle sigma2': first_below(T, np.full(n, KAPPA * s2)),
            "NOMP CFAR (a') oracle sigma2, MC kappa": first_below(T, np.full(n, KMC * s2)),
            'NOMP blind CFAR (kappa_b tuned)': first_below(T, KB * E[:, LMAX] / (256 - 2 * LMAX)),
            'NOMP blind MDL': np.argmin(256 * np.log(E) + 2 * np.arange(LMAX + 1)[None] * np.log(512), 1),
            'NOMP blind AIC': np.argmin(256 * np.log(E) + 4 * np.arange(LMAX + 1)[None], 1)}
    for m, l in mine.items():
        st = z[f'Lhat__{m}']; Sx = z[f'S__{m}']
        print(f'  SNR {snr:2d} {m:40s} L_hat identical to stored: {np.array_equal(st, l)}; P(L_hat=L) {np.mean(l == L):.4f}; '
              f'recall {Sx[:, 1].sum() / Sx[:, 2].sum():.4f}, precision {Sx[:, 1].sum() / Sx[:, 3].sum():.4f}; AIC L_hat histogram {np.bincount(l, minlength=11).tolist() if "AIC" in m else ""}')
    for m in ['IABR-Net heat count', 'NOMP known L', 'IABR-Net heat known L']:
        Sx = z[f'S__{m}']; l = z[f'Lhat__{m}']
        print(f'  SNR {snr:2d} {m:40s} P(L_hat=L) {np.mean(l == L):.4f}; recall {Sx[:, 1].sum() / Sx[:, 2].sum():.4f}, precision {Sx[:, 1].sum() / Sx[:, 3].sum():.4f}')

# ---------------------------------------------------------------- B. fresh banks
rng = np.random.default_rng(777); Z = (rng.standard_normal((20000, 16, 16)) + 1j * rng.standard_normal((20000, 16, 16))) / np.sqrt(2)
H0 = to_ant(Z)
g4 = (np.abs(np.fft.fft2(H0, s=(64, 64))) ** 2).reshape(20000, -1).max(1) / 256; g1 = (np.abs(np.fft.fft2(H0)) ** 2).reshape(20000, -1).max(1) / 256
print(f'B. MC re-derivation (seed 777, 20000 noise-only scenes): 0.99-quantile of the 4x-grid statistic {np.quantile(g4, 0.99):.4f} (orig {KMC:.4f}); '
      f'realised Pfa of analytic kappa: 4x grid {np.mean(g4 > KAPPA):.4f}, 16x16 grid {np.mean(g1 > KAPPA):.4f}')


def nomp_states(y):
    B = len(y); U = np.zeros((B, 0)); V = np.zeros((B, 0)); r = y.copy(); T4 = np.zeros((B, LMAX + 1)); T1 = T4.copy(); E = T4.copy(); st = [(U, V)]
    for k in range(LMAX + 1):
        X = np.abs(np.fft.fft2(r, s=(64, 64))).reshape(B, -1)
        T4[:, k] = X.max(1) ** 2 / 256; T1[:, k] = (np.abs(np.fft.fft2(r)) ** 2).reshape(B, -1).max(1) / 256; E[:, k] = (np.abs(r) ** 2).sum((1, 2))
        if k == LMAX:
            break
        i = X.argmax(1); u, v = C.newton(r, -2 * np.pi * (i // 64) / 64, 2 * np.pi * (i % 64) / 64)
        U, V = C.cyclic(y, np.c_[U, u], np.c_[V, v], 3)
        x, A = C.ls_gains(y, U, V); r = y - np.einsum('bik,bk->bi', A, x).reshape(y.shape); st.append((U, V))
    return st, T4, T1, E


def match(pred, psi, phi, L):
    """per scene [L_hat == L, matched, L, L_hat]"""
    out = np.zeros((len(L), 4))
    for i in range(len(L)):
        gp, gf = np.rad2deg(psi[i, :L[i]]), np.rad2deg(phi[i, :L[i]]); pp, pf = np.rad2deg(np.asarray(pred[i][0])), np.rad2deg(np.asarray(pred[i][1]))
        m = 0
        if len(pp):
            ok = (np.abs(gp[:, None] - pp[None]) <= 1) & (np.abs(gf[:, None] - pf[None]) <= 1)
            r, c = scipy.optimize.linear_sum_assignment(-ok.astype(float)); m = int(ok[r, c].sum())
        out[i] = [len(pp) == L[i], m, L[i], len(pp)]
    return out


def bootm(Sd, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(next(iter(Sd.values()))); Ri = rng.integers(0, n, (B, n)); o = {}
    for m, S_ in Sd.items():
        s = S_[Ri].sum(1); o[m] = np.stack([s[:, 0] / n, s[:, 1] / s[:, 2], s[:, 1] / s[:, 3]], 1)
    return o


res = {}
for snr in [15, 0]:
    t0 = time.time(); seed = 9600 + snr; n = 1000
    L = np.random.default_rng([seed, 7]).integers(1, 9, n)
    rng = np.random.default_rng(seed); psi, phi, alpha = S.sample_paths(L, rng, lmax=8)
    g = np.random.default_rng([seed, 901]); s = np.sqrt(10 ** (-snr / 10) / 2); noise = s * (g.standard_normal((n, 16, 16)) + 1j * g.standard_normal((n, 16, 16)))
    Y = forward(psi, phi, alpha, noise); H = to_ant(Y); heat, Hc = net(Y)
    st, T4, T1, E = nomp_states(H)
    if snr == 15:
        dm = max(np.abs(np.r_[(st[k][0][:30] - Uk).ravel(), (st[k][1][:30] - Vk).ravel()]).max() for k in [2, 5] for Uk, Vk in [C.nomp_greedy(H[:30], k, 4, 3)])
        print(f'  my step-wise NOMP state vs notebook nomp_greedy (k=2,5, 30 scenes): max|diff| {dm:.1e}')
    s2 = 10 ** (-snr / 10)
    Lh = {'(a) oracle CFAR, 4x grid': first_below(T4, np.full(n, KAPPA * s2)), "(a') oracle CFAR MC kappa": first_below(T4, np.full(n, KMC * s2)),
          '(a16) oracle CFAR, 16x16 grid': first_below(T1, np.full(n, KAPPA * s2)), 'blind CFAR kappa_b 13': first_below(T4, KB * E[:, LMAX] / (256 - 2 * LMAX)),
          'MDL': np.argmin(256 * np.log(E) + 2 * np.arange(LMAX + 1)[None] * np.log(512), 1), 'AIC': np.argmin(256 * np.log(E) + 4 * np.arange(LMAX + 1)[None], 1)}
    mx = scipy.ndimage.maximum_filter(heat, size=(1, 3, 3), mode='wrap'); Lh['heat count t 0.45'] = np.minimum(((heat >= mx) & (heat > TH)).reshape(n, -1).sum(1), LMAX)
    Lh['NOMP known L'] = L.copy()
    P = {}
    for m, l in Lh.items():
        P[m] = [((C.uv_to_angles(st[k][0][i:i + 1], st[k][1][i:i + 1])[0][0], C.uv_to_angles(st[k][0][i:i + 1], st[k][1][i:i + 1])[1][0]) if k else (np.zeros(0), np.zeros(0)))
                for i, k in enumerate(l)]
    for nm, l in [('IABR-Net heat @ heat count', Lh['heat count t 0.45']), ('IABR-Net heat known L', L)]:
        pr = [(np.zeros(0), np.zeros(0))] * n
        for k in sorted(set(l.tolist()) - {0}):
            ii = np.where(l == k)[0]; a, c = C.decode(Hc[ii], int(k), heat[ii], 'heat', C.SC)
            for j, i in enumerate(ii):
                pr[i] = (a[j], c[j])
        P[nm] = pr; Lh[nm] = l
    Sd = {m: match(P[m], psi, phi, L) for m in P}; Bt = bootm(Sd); res[snr] = {}
    print(f'SNR {snr} fresh bank (seed {seed}, {n} scenes, L hist {np.bincount(L, minlength=9)[1:].tolist()}) [{time.time() - t0:.0f}s]')
    for m, S_ in Sd.items():
        s_ = S_.sum(0); ci = [np.percentile(Bt[m][:, j], [2.5, 97.5]) for j in range(3)]
        res[snr][m] = dict(order_acc=s_[0] / n, recall=s_[1] / s_[2], precision=s_[1] / s_[3], ci=[c.tolist() for c in ci], mean_Lhat_minus_L=float((S_[:, 3] - S_[:, 2]).mean()))
        print(f'  {m:32s} P(L_hat=L) {s_[0] / n:.4f} [{ci[0][0]:.4f},{ci[0][1]:.4f}] recall {s_[1] / s_[2]:.4f} precision {s_[1] / s_[3]:.4f} mean(L_hat-L) {(S_[:, 3] - S_[:, 2]).mean():+.3f}')
    for a_, b_ in [('IABR-Net heat @ heat count', '(a) oracle CFAR, 4x grid'), ('IABR-Net heat @ heat count', 'blind CFAR kappa_b 13'),
                   ('IABR-Net heat @ heat count', "(a') oracle CFAR MC kappa"), ('blind CFAR kappa_b 13', '(a) oracle CFAR, 4x grid')]:
        d = Bt[a_] - Bt[b_]; sa, sb = Sd[a_].sum(0), Sd[b_].sum(0)
        pt = [sa[0] / n - sb[0] / n, sa[1] / sa[2] - sb[1] / sb[2], sa[1] / sa[3] - sb[1] / sb[3]]
        print(f'  diff {a_} - {b_}: ' + ' '.join(f'{k} {pt[j]:+.4f} [{np.percentile(d[:, j], 2.5):+.4f},{np.percentile(d[:, j], 97.5):+.4f}]'
                                                for j, k in enumerate(['order_acc', 'recall', 'precision'])))
        res[snr][f'diff {a_} - {b_}'] = [[pt[j], float(np.percentile(d[:, j], 2.5)), float(np.percentile(d[:, j], 97.5))] for j in range(3)]
json.dump(res, open(f'{HERE}/m7_check.json', 'w'), indent=1, default=float)
