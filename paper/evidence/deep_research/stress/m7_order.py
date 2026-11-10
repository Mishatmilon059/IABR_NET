"""M7: unknown number of paths. L ~ U{1..8} per scene, impairment-free model otherwise, SNR 0 and 15 dB.
Test banks: 1000 scenes per SNR, seeds 6000 / 6015.  Tuning banks (used ONLY to pick constants): 500 per SNR, seeds 6100 / 6115.
L drawn by default_rng([seed, 1]).integers(1, 9); paths/noise from stress_sim.base_scenes(seed) (lmax 8).

NOMP with stopping rules (one NOMP run per scene up to Lmax = 10 steps; the solution for L_hat = k is the state after k steps, which is
exactly notebook nomp_greedy(y, k) -- checked below on a subset). Detection statistic before step k+1:
  T_k = max over the 4x-oversampled (64x64) FFT grid of |<a(u,v), r_k>|^2 / 256,  r_k = residual after k paths (a = unit-modulus atom, |a|^2 = 256),
  so for a noise-only residual each critical-grid value is sigma^2 Exp(1).
 (a)  oracle-noise CFAR (Mamandipoor, Ramasamy, Madhow 2016): P(max of N=256 iid sigma^2 Exp(1) > tau) = 1 - (1 - e^{-tau/sigma^2})^N = Pfa
      -> tau = -sigma^2 ln(1 - (1 - Pfa)^{1/N}) = 10.145 sigma^2 for Pfa = 1e-2.  L_hat = first k with T_k < tau (else Lmax).
      sigma^2 = 10^(-SNR/10) (true noise variance per antenna-domain entry; to_ant is unitary).
 (a') same, kappa calibrated by Monte Carlo so that the max over the 4x OVERSAMPLED grid of noise-only data exceeds kappa sigma^2 with prob. 1e-2
      (20000 noise-only scenes, seed 321; no data with paths involved).
 (b)  blind (sigma^2 unknown), three candidates, the one with the best mean P(L_hat = L) on the TUNING banks is the primary blind rule:
      MDL: L_hat = argmin_k 256 ln E_k + 2 k ln 512   (E_k = ||r_k||^2, k = 0..Lmax; 4 real parameters per path, 512 real observations)
      AIC: L_hat = argmin_k 256 ln E_k + 4 k
      blind CFAR: sigma_hat^2 = E_Lmax / (256 - 2 Lmax), L_hat = first k with T_k < kappa_b sigma_hat^2; kappa_b tuned on the tuning banks.
IABR-Net: L_hat = number of 3x3 (wrap) local maxima of the 32x32 sigmoid heatmap with value > t (cap 10); t tuned on the tuning banks;
      then decode(Hc, L_hat, heat, 'heat', SC) (heat-seeded IABR-Net with that count).  Also 'NOMP @ heat count' (NOMP state at the heat count).
References with the TRUE L: NOMP known L, IABR-Net heat known L.
Metrics (per SNR): P(L_hat = L); path recall = matched / sum L; precision = matched / sum L_hat; a path is matched when both |d psi| and |d phi|
<= 1 deg after Hungarian assignment (cost = 1000 [max(|d psi|,|d phi|) > 1 deg] + Euclidean distance in deg); RMSE (deg) over the psi and phi
errors of matched paths. Paired bootstrap 2000 resamples within the SNR group, seed 0.
usage: python m7_order.py   (stdout -> m7_order.log; writes m7_results.json, cache/m7_*.npz)"""
import scipy.optimize
from scommon import *

LMAX = 10; N_TEST, N_TUNE = 1000, 500
SEED_TEST = {0: 6000, 15: 6015}; SEED_TUNE = {0: 6100, 15: 6115}
PFA = 1e-2; KAPPA = -np.log(1 - (1 - PFA) ** (1 / 256))


def make(n, snr, seed):
    L = np.random.default_rng([seed, 1]).integers(1, 9, n)
    b = SS.base_scenes(n, L, float(snr), seed, lmax=8)
    return b, SS.generate(b, 'clean')


def nomp_path(y):
    """returns states U[k], V[k] (k = 0..LMAX), T (B, LMAX+1), E (B, LMAX+1) -- the notebook nomp_greedy loop, recorded step by step."""
    B = len(y); NF = 64
    U = np.zeros((B, 0)); V = np.zeros((B, 0)); res = y.copy(); T = np.zeros((B, LMAX + 1)); E = np.zeros((B, LMAX + 1)); st = [(U, V)]
    for k in range(LMAX + 1):
        X = np.fft.fft2(res, s=(NF, NF)); A2 = np.abs(X).reshape(B, -1)
        T[:, k] = A2.max(1) ** 2 / 256; E[:, k] = (np.abs(res) ** 2).sum((1, 2))
        if k == LMAX:
            break
        idx = A2.argmax(1)
        u, v = C.newton(res, -2 * np.pi * (idx // NF) / NF, 2 * np.pi * (idx % NF) / NF)
        U = np.c_[U, u]; V = np.c_[V, v]
        U, V = C.cyclic(y, U, V, C.CFG['nomp_rc'])
        x, A = C.ls_gains(y, U, V); res = y - np.einsum('bik,bk->bi', A, x).reshape(y.shape)
        st.append((U.copy(), V.copy()))
    return st, T, E


def first_below(T, thr):
    below = T < thr[:, None]
    return np.where(below.any(1), below.argmax(1), LMAX)


def heat_count(heat, t):
    mx = scipy.ndimage.maximum_filter(heat, size=(1, 3, 3), mode='wrap')
    return np.minimum(((heat >= mx) & (heat > t)).reshape(len(heat), -1).sum(1), LMAX)


def preds_from_states(st, Lh):
    out = [None] * len(Lh)
    for i, k in enumerate(Lh):
        a = C.uv_to_angles(st[k][0][i:i + 1], st[k][1][i:i + 1]) if k > 0 else (np.zeros((1, 0)), np.zeros((1, 0)))
        out[i] = (a[0][0], a[1][0])
    return out


def heat_decode(Hc, heat, Lh):
    out = [(np.zeros(0), np.zeros(0))] * len(Lh)
    for k in sorted(set(Lh.tolist()) - {0}):
        ii = np.where(Lh == k)[0]
        a, c = C.decode(Hc[ii], int(k), heat[ii], 'heat', C.SC)
        for j, i in enumerate(ii):
            out[i] = (a[j], c[j])
    return out


def match_stats(pred, b, Lh):
    """per scene: [order correct, matched, L, L_hat, sumsq matched errors (deg^2, psi and phi)]"""
    n = len(Lh); S = np.zeros((n, 5))
    for i in range(n):
        L = int(b['L'][i]); gp = np.rad2deg(b['psi'][i, :L]); gf = np.rad2deg(b['phi'][i, :L])
        pp = np.rad2deg(np.asarray(pred[i][0], float)); pf = np.rad2deg(np.asarray(pred[i][1], float)); K = len(pp)
        m = 0; ss = 0.0
        if K:
            dp = gp[:, None] - pp[None]; df = gf[:, None] - pf[None]
            ok = (np.abs(dp) <= 1) & (np.abs(df) <= 1)
            cost = 1000.0 * (~ok) + np.hypot(dp, df)
            r, c = scipy.optimize.linear_sum_assignment(cost)
            g = ok[r, c]; m = int(g.sum()); ss = float((dp[r, c][g] ** 2 + df[r, c][g] ** 2).sum())
        S[i] = [int(K == L), m, L, K, ss]
    return S


def summ(S):
    s = S.sum(0)
    return dict(order_acc=s[0] / len(S), recall=s[1] / s[2], precision=s[1] / s[3] if s[3] else np.nan,
                rmse=float(np.sqrt(s[4] / (2 * s[1]))) if s[1] else np.nan, mean_Lhat_minus_L=float((S[:, 3] - S[:, 2]).mean()))


def boot(Sd, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(next(iter(Sd.values()))); R = rng.integers(0, n, (B, n)); out = {}
    for m, S in Sd.items():
        s = S[R].sum(1)            # (B,5)
        with np.errstate(all='ignore'):
            out[m] = np.stack([s[:, 0] / n, s[:, 1] / s[:, 2], s[:, 1] / s[:, 3], np.sqrt(s[:, 4] / (2 * s[:, 1]))], 1)
    return out


MK = ['order_acc', 'recall', 'precision', 'rmse']


def run_bank(b, Y):
    t0 = time.time(); Hant = SS.to_ant(Y); heat, Hc = cnn_forward(Y)
    st, T, E = nomp_path(Hant)
    print(f'   NOMP path to Lmax={LMAX}: {time.time() - t0:.0f}s', flush=True)
    return dict(Hant=Hant, heat=heat, Hc=Hc, st=st, T=T, E=E)


t00 = time.time()
# ---- (a') Monte-Carlo kappa for the oversampled statistic
rng = np.random.default_rng(321); Z = (rng.standard_normal((20000, 16, 16)) + 1j * rng.standard_normal((20000, 16, 16))) / np.sqrt(2)
g4 = (np.abs(np.fft.fft2(SS.to_ant(Z), s=(64, 64))) ** 2).reshape(20000, -1).max(1) / 256
KAPPA_MC = float(np.quantile(g4, 1 - PFA))
print(f'(a) analytic kappa = {KAPPA:.4f} (realised Pfa on the 4x grid, noise only: {np.mean(g4 > KAPPA):.4f}); (a\') MC kappa = {KAPPA_MC:.4f}', flush=True)

# ---- check: state after k steps == notebook nomp_greedy(y, k)
b, Y = make(40, 15, 999); H = SS.to_ant(Y); st, _, _ = nomp_path(H)
dmax = max(float(np.abs(np.r_[(st[k][0] - U).ravel(), (st[k][1] - V).ravel()]).max()) for k in [1, 3, 6] for U, V in [C.nomp_greedy(H, k, 4, 3)])
print(f'check nomp_path state k == nomp_greedy(y, k), k in 1,3,6, 40 scenes: max |diff| {dmax:.1e} rad', flush=True)

# ---- tuning
print('TUNING banks (seeds 6100/6115, 500 scenes per SNR)', flush=True)
tune = {}
for snr in [0, 15]:
    bt, Yt = make(N_TUNE, snr, SEED_TUNE[snr]); tune[snr] = (bt, run_bank(bt, Yt))
TGRID = np.round(np.arange(0.05, 0.951, 0.025), 3); KGRID = np.round(np.arange(4.0, 30.01, 0.5), 2)
acc_t = np.array([np.mean([np.mean(heat_count(tune[s][1]['heat'], t) == tune[s][0]['L']) for s in [0, 15]]) for t in TGRID])
T_BEST = float(TGRID[acc_t.argmax()])


def blind_cfar(r, kb):
    s2 = r['E'][:, LMAX] / (256 - 2 * LMAX); return first_below(r['T'], kb * s2)


def mdl(r):
    return np.argmin(256 * np.log(r['E']) + 2 * np.arange(LMAX + 1)[None] * np.log(512), 1)


def aic(r):
    return np.argmin(256 * np.log(r['E']) + 4 * np.arange(LMAX + 1)[None], 1)


acc_k = np.array([np.mean([np.mean(blind_cfar(tune[s][1], k) == tune[s][0]['L']) for s in [0, 15]]) for k in KGRID])
K_BEST = float(KGRID[acc_k.argmax()])
tune_acc = {'heat count t*': float(acc_t.max()), 'blind CFAR kappa_b*': float(acc_k.max()),
            'MDL': float(np.mean([np.mean(mdl(tune[s][1]) == tune[s][0]['L']) for s in [0, 15]])),
            'AIC': float(np.mean([np.mean(aic(tune[s][1]) == tune[s][0]['L']) for s in [0, 15]])),
            '(a) CFAR oracle': float(np.mean([np.mean(first_below(tune[s][1]['T'], np.full(N_TUNE, KAPPA * 10 ** (-s / 10))) == tune[s][0]['L']) for s in [0, 15]])),
            "(a') CFAR oracle MC": float(np.mean([np.mean(first_below(tune[s][1]['T'], np.full(N_TUNE, KAPPA_MC * 10 ** (-s / 10))) == tune[s][0]['L']) for s in [0, 15]]))}
BLIND = max(['blind CFAR kappa_b*', 'MDL', 'AIC'], key=lambda k: tune_acc[k])
print(f'  heat threshold grid {TGRID[0]}..{TGRID[-1]} step 0.025 -> t* = {T_BEST} (tuning P(L_hat=L) {acc_t.max():.4f}); '
      f'blind-CFAR kappa_b grid {KGRID[0]}..{KGRID[-1]} -> kappa_b* = {K_BEST} ({acc_k.max():.4f})', flush=True)
print('  tuning-bank mean P(L_hat = L): ' + ', '.join(f'{k} {v:.4f}' for k, v in tune_acc.items()) + f'  -> primary blind rule: {BLIND}', flush=True)
del tune

# ---- test
res = dict(Lmax=LMAX, pfa=PFA, kappa_analytic=KAPPA, kappa_mc=KAPPA_MC, t_heat=T_BEST, kappa_blind=K_BEST, blind_primary=BLIND, tuning_acc=tune_acc,
           tune_grid=dict(t=TGRID.tolist(), acc_t=acc_t.tolist(), kappa=KGRID.tolist(), acc_k=acc_k.tolist()), snr={})
for snr in [15, 0]:
    print(f'TEST bank SNR {snr} (seed {SEED_TEST[snr]}, {N_TEST} scenes)', flush=True)
    b, Y = make(N_TEST, snr, SEED_TEST[snr]); r = run_bank(b, Y); s2 = 10 ** (-snr / 10)
    Lh = {'NOMP CFAR (a) oracle sigma2': first_below(r['T'], np.full(N_TEST, KAPPA * s2)),
          "NOMP CFAR (a') oracle sigma2, MC kappa": first_below(r['T'], np.full(N_TEST, KAPPA_MC * s2)),
          'NOMP blind MDL': mdl(r), 'NOMP blind AIC': aic(r), 'NOMP blind CFAR (kappa_b tuned)': blind_cfar(r, K_BEST),
          'IABR-Net heat count': heat_count(r['heat'], T_BEST)}
    Lh['NOMP @ heat count'] = Lh['IABR-Net heat count']
    Lh['NOMP known L'] = b['L'].copy(); Lh['IABR-Net heat known L'] = b['L'].copy()
    P = {}
    for m, l in Lh.items():
        if m.startswith('IABR-Net'):
            P[m] = heat_decode(r['Hc'], r['heat'], l)
        else:
            P[m] = preds_from_states(r['st'], l)
    Sd = {m: match_stats(P[m], b, Lh[m]) for m in P}
    np.savez_compressed(f'{MYCACHE}/m7_test_snr{snr}.npz', L=b['L'], **{f'Lhat__{m}': v for m, v in Lh.items()},
                        **{f'S__{m}': v for m, v in Sd.items()}, T=r['T'], E=r['E'])
    Bt = boot(Sd)
    out = dict(seed=SEED_TEST[snr], n=N_TEST, L_hist=np.bincount(b['L'], minlength=9)[1:].tolist(), methods={}, diffs={}, by_L={})
    for m in Sd:
        sm = summ(Sd[m])
        out['methods'][m] = dict(sm, ci={k: list(C.ci(Bt[m][:, j])[1:]) for j, k in enumerate(MK)})
        out['by_L'][m] = {int(L): float(np.mean(Lh[m][b['L'] == L] == L)) for L in range(1, 9)}
        print(f'  {m:40s} P(L_hat=L) {sm["order_acc"]:.4f} [{out["methods"][m]["ci"]["order_acc"][0]:.4f},{out["methods"][m]["ci"]["order_acc"][1]:.4f}] '
              f'recall {sm["recall"]:.4f} precision {sm["precision"]:.4f} rmse {sm["rmse"]:.4f} mean(L_hat-L) {sm["mean_Lhat_minus_L"]:+.3f}', flush=True)
    blind_name = {'blind CFAR kappa_b*': 'NOMP blind CFAR (kappa_b tuned)', 'MDL': 'NOMP blind MDL', 'AIC': 'NOMP blind AIC'}[BLIND]
    pairs = [('IABR-Net heat count', 'NOMP CFAR (a) oracle sigma2'), ('IABR-Net heat count', blind_name),
             ('IABR-Net heat count', "NOMP CFAR (a') oracle sigma2, MC kappa"), ('NOMP @ heat count', 'NOMP CFAR (a) oracle sigma2'),
             ('NOMP @ heat count', blind_name), ('NOMP CFAR (a) oracle sigma2', blind_name), ('IABR-Net heat known L', 'NOMP known L')]
    for a_, b_ in pairs:
        d = Bt[a_] - Bt[b_]; sa, sb = summ(Sd[a_]), summ(Sd[b_])
        out['diffs'][f'{a_} - {b_}'] = {k: [sa[k] - sb[k], *C.ci(d[:, j])[1:]] for j, k in enumerate(MK)}
        dd = out['diffs'][f'{a_} - {b_}']
        print(f'  diff {a_} - {b_}: ' + ' '.join(f'{k} {dd[k][0]:+.4f} [{dd[k][1]:+.4f},{dd[k][2]:+.4f}]' for k in MK), flush=True)
    res['snr'][str(snr)] = out
json.dump(res, open(f'{HERE}/m7_results.json', 'w'), indent=1, default=float)
print(f'total {time.time() - t00:.0f}s')
