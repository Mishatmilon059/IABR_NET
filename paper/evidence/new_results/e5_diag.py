"""E5: diagnostics (diag.py / diag2.py / diag3.py quantities) re-run for the NETWORK-FREE decoder on the first 250 scenes per SNR of the official
bank (2000 scenes), input = raw antenna-domain channel Ht (net__official.npz), L = 3.
 (a) greedy only (nomp_greedy os=4, rc=3)    (b) greedy + selfcal (lam,tau)=(1.0,5.0)
METRIC NAME: 'component-wise score of spec 10.2' (fraction of the 6 components within 1 deg after L1 assignment over 3! permutations); NOT paper-style Pd.
The single-path Cramer-Rao approximation of diag3.py is REPLACED by the numerically computed JOINT Fisher information (Slepian-Bangs) of
y = sum_l x_l a(u_l,v_l) + w; the single-path numbers are also printed for contrast."""
import sys; sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
import itertools
from scipy.stats import norm, chi2
PERMS = list(itertools.permutations(range(3))); NTEST = 250; t00 = time.time()
b = load_bank('official'); n = np.load(f'{CACHE}/net__official.npz'); snr_all = b['snr']
idx = np.concatenate([np.where(snr_all == s)[0][:NTEST] for s in sorted(set(snr_all.tolist()))])
Ht = n['Ht'][idx].astype(np.complex128); PT, FT, snr = b['psi'][idx][:, :3].astype(np.float64), b['phi'][idx][:, :3].astype(np.float64), snr_all[idx]
UT, VT = np.pi * np.cos(PT), np.pi * np.cos(FT); N = len(idx); SNRS = sorted(set(snr.tolist()))
out = {}
OUTTXT = []


def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); OUTTXT.append(s)


P('E5 diagnostics, network-free decoder, official bank, first 250 scenes per SNR (%d scenes), input Ht (no IABC), L=3' % N)
P('Metric = component-wise score of spec 10.2 (NOT paper-style Pd). nomp_os=4, nomp_rc=3, selfcal (lam,tau)=(1.0,5.0)')


def flags(U, V):
    ps, ph = uv_to_angles(U, V); ps, ph = np.degrees(ps), np.degrees(ph); tp, tf = np.degrees(PT), np.degrees(FT)
    ok = np.zeros((len(U), 6), bool); err = np.zeros((len(U), 6))
    for k in range(len(U)):
        best = min(PERMS, key=lambda p: sum(abs(ps[k, q] - tp[k, j]) + abs(ph[k, q] - tf[k, j]) for j, q in enumerate(p)))
        e = [abs(ps[k, q] - tp[k, j]) for j, q in enumerate(best)] + [abs(ph[k, q] - tf[k, j]) for j, q in enumerate(best)]
        err[k] = e; ok[k] = np.array(e) <= 1
    return ok, err


_nc = np.arange(16.) - 7.5


def selfcal_ext(y, U, V, lam, tau, rc=2, iters=2):
    """verbatim copy of selfcal() (notebook_src.py lines 597-620) that additionally returns the final calibrated data yc"""
    B = len(y); yc = y.copy(); U, V = U.copy(), V.copy()
    for _ in range(iters):
        x, A = ls_gains(yc, U, V); Hm = np.einsum('bik,bk->bi', A, x).reshape(y.shape)
        snr_est = (np.abs(Hm) ** 2).sum((1, 2)) / ((np.abs(yc - Hm) ** 2).sum((1, 2)) + 1e-12)
        dt = np.ones((B, 16), complex)
        for _ in range(3):
            M = Hm * dt[:, None, :]; dr = (y * np.conj(M)).sum(2) / ((np.abs(M) ** 2).sum(2) + 1e-12)
            M = Hm * dr[:, :, None]; dt = (y * np.conj(M)).sum(1) / ((np.abs(M) ** 2).sum(1) + 1e-12)

        def clean(d):
            lg = np.log(np.abs(d) + 1e-12); ph = np.unwrap(np.angle(d), axis=1)
            lg = lg - lg.mean(1, keepdims=True); ph = ph - ph.mean(1, keepdims=True)
            ph = ph - (ph @ _nc)[:, None] / (_nc @ _nc) * _nc
            return lam * lg, lam * ph
        lr, pr = clean(dr); lt, pt = clean(dt); gate = (snr_est > tau)[:, None]
        cr = np.where(gate, np.exp(-lr - 1j * pr), 1.0); ct = np.where(gate, np.exp(-lt - 1j * pt), 1.0)
        yc = y * cr[:, :, None] * ct[:, None, :]; U, V = cyclic(yc, U, V, rc)
    return U, V, yc


# ---------------- estimates
t0 = time.time(); UA, VA = nomp_greedy(Ht, 3, 4, 3)
cached = np.load(f'{CACHE}/pred__NOMP_no_network__official.npz'); pa = uv_to_angles(UA, VA)
P('greedy recompute equals cached NOMP_no_network on these scenes:', bool(np.allclose(pa[0], cached['psi'][idx][:, :3]) and np.allclose(pa[1], cached['phi'][idx][:, :3])), '[%.0fs]' % (time.time() - t0))
US, VS, YC = selfcal_ext(Ht, UA, VA, 1.0, 5.0); U2, V2 = selfcal(Ht, UA, VA, 1.0, 5.0)
P('selfcal_ext equals selfcal():', bool(np.allclose(US, U2) and np.allclose(VS, V2)))
if os.path.exists(f'{OUT}/preds/selfcal_pred__official.npz'):
    sc_cached = np.load(f'{OUT}/preds/selfcal_pred__official.npz'); ps_ = uv_to_angles(US, VS)
    P('selfcal estimate equals e13 selfcal preds on these scenes:', bool(np.allclose(ps_[0], sc_cached['psi'][idx][:, :3]) and np.allclose(ps_[1], sc_cached['phi'][idx][:, :3])))
EST = {'greedy': (UA, VA, Ht), 'greedy+selfcal': (US, VS, YC)}


def refine_from(U0, V0, hc, pre, rc):
    Un, Vn = newton(hc[:, None].repeat(3, 1).reshape(-1, 16, 16), U0.ravel(), V0.ravel(), iters=pre); return cyclic(hc, Un.reshape(-1, 3), Vn.reshape(-1, 3), rc)


# ---------------- oracle starts and iteration controls
UC, VC = refine_from(UT.copy(), VT.copy(), Ht, 4, 3)                       # diag C1 for the raw-data objective
UCs, VCs = selfcal(Ht, UC, VC, 1.0, 5.0)                                    # oracle + selfcal
UB, VB = refine_from(UA, VA, Ht, 30, 15)                                    # diag B: 10x more iterations from the greedy estimate
UBg, VBg = nomp_greedy(Ht, 3, 4, 15)                                        # greedy with rc=15 inside every stage
UBs, VBs = selfcal(Ht, UA, VA, 1.0, 5.0, rc=10, iters=6)                    # selfcal with 3x iters and 5x rounds
okA, errA = flags(UA, VA); okS, errS = flags(US, VS); okC, errC = flags(UC, VC); okCs, _ = flags(UCs, VCs)
okB, _ = flags(UB, VB); okBg, _ = flags(UBg, VBg); okBs, _ = flags(UBs, VBs)


def mv(U1, U0):
    return 100 * np.mean(np.degrees(np.abs(np.angle(np.exp(1j * (U1 - U0))))).max(1) > 0.05)


P('\n=== 1. iteration-count control (component-wise score, all 2000 scenes) ===')
P('greedy                                       %.4f' % okA.mean())
P('greedy, 10x more newton/cyclic iterations from its estimate (pre=30, rc=15): %.4f ; scenes whose u moved >0.05 deg: %.1f%% ; v: %.1f%%' % (okB.mean(), mv(UB, UA), mv(VB, VA)))
P('greedy with rc=15 inside every stage          %.4f' % okBg.mean())
P('start at the true angles (C1, newton4+cyclic3 on Ht)           %.4f' % okC.mean())
P('greedy+selfcal                                %.4f' % okS.mean())
P('greedy+selfcal with iters=6, rc=10 (3x/5x)     %.4f ; scenes whose u moved >0.05 deg vs standard selfcal: %.1f%%' % (okBs.mean(), mv(UBs, US)))
P('start at truth + selfcal                       %.4f' % okCs.mean())
out['iteration_control'] = dict(greedy=okA.mean(), greedy_10x=okB.mean(), greedy_rc15=okBg.mean(), truth_start=okC.mean(), selfcal=okS.mean(), selfcal_3x5x=okBs.mean(), truth_start_selfcal=okCs.mean(),
                                moved_u_pct_greedy10x=mv(UB, UA), moved_u_pct_selfcal_more=mv(UBs, US))

# ---------------- objective vs truth taxonomy (diag2)
x_t, _ = ls_gains(Ht, UT, VT); amp2 = np.abs(x_t) ** 2
RT_raw = residual_energy(Ht, UT, VT); RC_raw = residual_energy(Ht, UC, VC)
sig2_d2 = RT_raw / (256 - 9); sig2_d3 = RT_raw / (256 - 3)
chi99 = float(chi2.ppf(0.99, 6))
P('\n=== 2. objective-vs-truth taxonomy (diag2 statistic dE = (RA-RT)/sig2, sig2 = RT/(256-9); chi2(6) 99%% = %.2f) ===' % chi99)
hi = snr >= 15
T0 = 2 * (RT_raw - RC_raw) / sig2_d3
P('null-scale check (scenes SNR>=15, ML started at the truth): quantiles of T=2*(RT-RC)/sigma2 at 5/50/95/99%%: %s ; chi2(6): %s' % (np.round(np.percentile(T0[hi], [5, 50, 95, 99]), 2), np.round(chi2.ppf([.05, .5, .95, .99], 6), 2)))
out['null_scale_check'] = dict(T_quantiles_5_50_95_99=np.percentile(T0[hi], [5, 50, 95, 99]).tolist(), chi2_6=chi2.ppf([.05, .5, .95, .99], 6).tolist())
TAX = {}
for nm, (U, V, data) in EST.items():
    ok = okA if nm == 'greedy' else okS; sA = ok.all(1); sC = okC.all(1) if nm == 'greedy' else okCs.all(1)
    RA = residual_energy(data, U, V); RT = residual_energy(data, UT, VT)
    sg = RT / (256 - 9); dE = (RA - RT) / sg
    fail = ~sA; c1 = fail & (dE <= 0); c2 = fail & (dE > 0) & (dE <= chi99); c3 = fail & (dE > chi99)
    Tt = 2 * (RA - RT) / (RT / (256 - 3)); d1 = fail & (Tt <= 0); d2 = fail & (Tt > 0) & (Tt <= chi99); d3 = fail & (Tt > chi99)
    P('\n[%s] data used for the objective: %s' % (nm, 'raw Ht' if nm == 'greedy' else 'self-calibrated yc (final selfcal data)'))
    P('  scenes failing (any of 6 components >1 deg): %d of %d (%.1f%%) ; sample success %.3f' % (fail.sum(), N, 100 * fail.mean(), sA.mean()))
    P('  diag2 convention (threshold %.1f on dE):   fits BETTER than truth (dE<=0): %d (%.0f%%) | inside band: %d (%.0f%%) | clearly worse (optimiser miss): %d (%.0f%%)' % (
        chi99, c1.sum(), 100 * c1.sum() / fail.sum(), c2.sum(), 100 * c2.sum() / fail.sum(), c3.sum(), 100 * c3.sum() / fail.sum()))
    P('  corrected scale (T=2dE, sigma2=RT/253):   fits BETTER than truth: %d (%.0f%%) | inside chi2(6) 99%% band: %d (%.0f%%) | outside (optimiser miss): %d (%.0f%%)' % (
        d1.sum(), 100 * d1.sum() / fail.sum(), d2.sum(), 100 * d2.sum() / fail.sum(), d3.sum(), 100 * d3.sum() / fail.sum()))
    P('  failed here but succeeded with the truth start: %d' % (fail & sC).sum())
    P('  per SNR: sample-success | truth-start | optimiser misses %% of all scenes (diag2 conv.) | (corrected)')
    rows = []
    for s_ in SNRS:
        m = snr == s_; P('   %4d   %.3f   %.3f   %5.1f   %5.1f' % (int(s_), sA[m].mean(), sC[m].mean(), 100 * (c3 & m).sum() / m.sum(), 100 * (d3 & m).sum() / m.sum()))
        rows.append(dict(snr=int(s_), success=float(sA[m].mean()), truth_start=float(sC[m].mean()), miss_pct_diag2=float(100 * (c3 & m).sum() / m.sum()), miss_pct_corrected=float(100 * (d3 & m).sum() / m.sum())))
    TAX[nm] = dict(fail=int(fail.sum()), better=int(c1.sum()), inband=int(c2.sum()), miss=int(c3.sum()), c_better=int(d1.sum()), c_inband=int(d2.sum()), c_miss=int(d3.sum()), recovered_by_truth_start=int((fail & sC).sum()), per_snr=rows)
    if nm != 'greedy':
        RAr = residual_energy(Ht, U, V); dEr = (RAr - RT_raw) / sig2_d2; c3r = fail & (dEr > chi99); cb = fail & (dEr <= 0)
        P('  [raw-Ht objective for the self-cal estimate] fits better than truth on raw Ht: %d (%.0f%%); outside band (dE>%.1f): %d (%.0f%%)' % (cb.sum(), 100 * cb.sum() / fail.sum(), chi99, c3r.sum(), 100 * c3r.sum() / fail.sum()))
        TAX[nm]['raw_better'] = int(cb.sum()); TAX[nm]['raw_miss'] = int(c3r.sum())
out['taxonomy'] = TAX

# ---------------- stratification (diag2 end-fire/high-SNR; diag3 weakest-path)
endf_c = np.concatenate([np.abs(np.cos(PT)) > 0.995, np.abs(np.cos(FT)) > 0.995], 1); endf_s = endf_c.any(1)
tp, tf = np.degrees(PT), np.degrees(FT)
minsep = np.array([min(np.hypot(tp[k, a] - tp[k, c], tf[k, a] - tf[k, c]) for a, c in [(0, 1), (0, 2), (1, 2)]) for k in range(N)])
hs = snr >= 10; STRAT = {}
P('\n=== 3. stratification ===')
P('HIGH-SNR (>=10 dB) scenes: %d ; sample success (all 6 comps)' % hs.sum())
for nm, ok in [('greedy', okA), ('greedy+selfcal', okS)]:
    s = ok.all(1); d = {}
    for name, grp in [('has an end-fire path (|cos|>0.995)', endf_s), ('no end-fire path', ~endf_s), ('no end-fire, closest pair >=20 deg', ~endf_s & (minsep >= 20))]:
        m = hs & grp; d[name] = (int(m.sum()), float(s[m].mean())); P('  [%s] %-38s n=%4d success %.3f' % (nm, name, m.sum(), s[m].mean()))
    STRAT[nm] = d
rk = np.argsort(np.argsort(amp2, 1), 1)


def comp_tables(ok_c, label):
    for r_, rn in [(0, 'weakest'), (1, 'middle'), (2, 'strongest')]:
        m6 = np.concatenate([rk == r_, rk == r_], 1)
        P('  [%s] %-9s path: observed %.3f (n=%d comps), median true-path SNR %.1f dB' % (label, rn, ok_c[m6].mean(), m6.sum(), 10 * np.log10(np.median((amp2 / sig2_d3[:, None])[rk == r_]))))


P('component success vs path-power rank (all SNRs):'); comp_tables(okA.astype(float), 'greedy'); comp_tables(okS.astype(float), 'greedy+selfcal')
out['stratification'] = STRAT


# ---------------- joint Fisher information
def joint_crb(U, V, x, sig2):
    """U,V,x: (B,3); sig2: (B,) per-entry noise energy E|w|^2. Parameters per path: Re x, Im x, u, v. Returns CRB std of u (B,3), v (B,3), cond numbers, D."""
    A = atoms(U, V)                                                  # (B,256,3)
    nn_ = np.arange(16.)[:, None].repeat(16, 1).ravel(); mm_ = np.arange(16.)[None, :].repeat(16, 0).ravel()
    Du = x[:, None, :] * (-1j * nn_[None, :, None]) * A; Dv = x[:, None, :] * (1j * mm_[None, :, None]) * A
    D = np.concatenate([A, 1j * A, Du, Dv], 2)                       # (B,256,12) order: xr(3), xi(3), u(3), v(3)
    FIM = (2 / sig2)[:, None, None] * np.real(np.conj(np.transpose(D, (0, 2, 1))) @ D)
    cond = np.linalg.cond(FIM); CRB = np.linalg.inv(FIM)
    d = np.diagonal(CRB, axis1=1, axis2=2)
    return np.sqrt(d[:, 6:9]), np.sqrt(d[:, 9:12]), cond, D


def model(U, V, x): return (atoms(U, V) @ x[..., None])[..., 0]


k = 7; sd_u1, sd_v1, cnd, Dk = joint_crb(UT[k:k + 1], VT[k:k + 1], x_t[k:k + 1], sig2_d3[k:k + 1]); eps = 1e-6; fd = []
for j in range(3):
    Up = UT[k:k + 1].copy(); Up[0, j] += eps; Um = UT[k:k + 1].copy(); Um[0, j] -= eps
    fd.append((model(Up, VT[k:k + 1], x_t[k:k + 1]) - model(Um, VT[k:k + 1], x_t[k:k + 1]))[0] / (2 * eps))
fd_err = float(np.abs(np.stack(fd, 1) - Dk[0, :, 6:9]).max() / np.abs(Dk[0, :, 6:9]).max())
P('\nFIM check: max |analytic dy/du - finite difference| / max|dy/du| = %.2e' % fd_err)
sdu, sdv, cond, _ = joint_crb(UT, VT, x_t, sig2_d3)
P('joint FIM condition number: median %.3g, max %.3g ; non-finite CRB entries: %d' % (np.median(cond), cond.max(), int((~np.isfinite(sdu)).sum() + (~np.isfinite(sdv)).sum())))
rng = np.random.default_rng(1); sel = np.where((snr >= 20) & ~endf_s & (minsep >= 20))[0][:40]; R = 100; ratios_u = []; ratios_v = []
for k in sel:
    mu = model(UT[k:k + 1], VT[k:k + 1], x_t[k:k + 1])[0]
    w = (rng.standard_normal((R, 16, 16)) + 1j * rng.standard_normal((R, 16, 16))) * np.sqrt(sig2_d3[k] / 2)
    y = mu.reshape(16, 16)[None] + w
    Ue, Ve = refine_from(np.repeat(UT[k:k + 1], R, 0), np.repeat(VT[k:k + 1], R, 0), y, 4, 3)
    du_ = np.angle(np.exp(1j * (Ue - UT[k]))); dv_ = np.angle(np.exp(1j * (Ve - VT[k])))
    ratios_u.append(du_.std(0) / sdu[k]); ratios_v.append(dv_.std(0) / sdv[k])
ratios = np.concatenate([np.concatenate(ratios_u), np.concatenate(ratios_v)])
P('Monte-Carlo check of the joint CRB (%d scenes with SNR>=20, no end-fire, pairs>=20deg, %d noise draws each, ML started at truth): empirical std / CRB std over %d (path,angle) cells: median %.3f, 10-90%% range %.3f-%.3f' % (
    len(sel), R, len(ratios), np.median(ratios), *np.percentile(ratios, [10, 90])))
out['crb_validation'] = dict(fd_rel_err=fd_err, mc_ratio_median=float(np.median(ratios)), mc_ratio_p10_p90=np.percentile(ratios, [10, 90]).tolist(), cond_median=float(np.median(cond)), cond_max=float(cond.max()))

d1 = np.radians(1.0)


def p_within(theta, sd, clamp):
    mu = np.pi * np.cos(theta); lo_t, hi_t = np.minimum(theta + d1, np.pi), np.maximum(theta - d1, 0.0)
    u_lo, u_hi = np.pi * np.cos(lo_t), np.pi * np.cos(hi_t)
    if clamp:
        u_lo = np.where(theta + d1 >= np.pi, -np.inf, u_lo); u_hi = np.where(theta - d1 <= 0, np.inf, u_hi)
    return norm.cdf((u_hi - mu) / sd) - norm.cdf((u_lo - mu) / sd)


sd1 = np.sqrt(sig2_d3[:, None] / (2 * amp2 * 5440))                  # diag3 single-path approximation
PJw = np.concatenate([p_within(PT, sdu, False), p_within(FT, sdv, False)], 1); PJc = np.concatenate([p_within(PT, sdu, True), p_within(FT, sdv, True)], 1)
PSw = np.concatenate([p_within(PT, sd1, False), p_within(FT, sd1, False)], 1); PSc = np.concatenate([p_within(PT, sd1, True), p_within(FT, sd1, True)], 1)
obsA = (errA <= 1).astype(float); obsS = (errS <= 1).astype(float)
P('\n=== 4. component-wise score (spec 10.2, NOT paper-style Pd): observed vs CRB-limit prediction ===')
P('joint = numerically computed joint FIM (L=3, 12 real parameters); single = diag3 single-path approximation (for contrast). wrap = physical (u beyond +-pi wraps to the other end-fire = fold); clamp = hypothetical fix.')
P('SNR  obs(greedy) obs(greedy+sc) | joint-wrap joint-clamp | single-wrap single-clamp')
CR = []
for s_ in SNRS:
    m = snr == s_; r = dict(snr=int(s_), obs_greedy=float(obsA[m].mean()), obs_selfcal=float(obsS[m].mean()), joint_wrap=float(PJw[m].mean()), joint_clamp=float(PJc[m].mean()), single_wrap=float(PSw[m].mean()), single_clamp=float(PSc[m].mean())); CR.append(r)
    P('%4d   %.3f       %.3f        | %.3f      %.3f      | %.3f       %.3f' % (r['snr'], r['obs_greedy'], r['obs_selfcal'], r['joint_wrap'], r['joint_clamp'], r['single_wrap'], r['single_clamp']))
r = dict(snr='all', obs_greedy=float(obsA.mean()), obs_selfcal=float(obsS.mean()), joint_wrap=float(PJw.mean()), joint_clamp=float(PJc.mean()), single_wrap=float(PSw.mean()), single_clamp=float(PSc.mean())); CR.append(r)
P('all    %.3f       %.3f        | %.3f      %.3f      | %.3f       %.3f' % (r['obs_greedy'], r['obs_selfcal'], r['joint_wrap'], r['joint_clamp'], r['single_wrap'], r['single_clamp']))
out['crb_prediction'] = CR
big_A = errA > 90; big_S = errS > 90; wrong_A = ~(errA <= 1); wrong_S = ~(errS <= 1)
P('\n=== 5. end-fire loss (components with true |cos|>0.995) ===')
EF = {}
for name, m in [('end-fire comps', endf_c), ('other comps', ~endf_c)]:
    d = dict(n=int(m.sum()), obs_greedy=float(obsA[m].mean()), obs_selfcal=float(obsS[m].mean()), joint_wrap=float(PJw[m].mean()), joint_clamp=float(PJc[m].mean()), single_wrap=float(PSw[m].mean()), single_clamp=float(PSc[m].mean()),
             fold_greedy=int(big_A[m].sum()), fold_selfcal=int(big_S[m].sum())); EF[name] = d
    P('  %-15s n=%5d | observed greedy %.3f, greedy+sc %.3f | joint-wrap %.3f joint-clamp %.3f | single-wrap %.3f single-clamp %.3f | fold errors (>90 deg): greedy %d, +sc %d' % (name, d['n'], d['obs_greedy'], d['obs_selfcal'], d['joint_wrap'], d['joint_clamp'], d['single_wrap'], d['single_clamp'], d['fold_greedy'], d['fold_selfcal']))
P('  total components %d ; wrong components: greedy %d, +sc %d ; fold errors (>90 deg): greedy %d (%.1f%% of wrong), +sc %d (%.1f%% of wrong); fold errors on end-fire comps: greedy %d, +sc %d' % (
    obsA.size, wrong_A.sum(), wrong_S.sum(), big_A.sum(), 100 * big_A.sum() / wrong_A.sum(), big_S.sum(), 100 * big_S.sum() / wrong_S.sum(), (big_A & endf_c).sum(), (big_S & endf_c).sum()))
tot_loss_A = (1 - obsA).sum(); ef_loss_A = (1 - obsA)[endf_c].sum()
P('  share of all failed components that are end-fire components: greedy %.1f%%, +sc %.1f%%' % (100 * ef_loss_A / tot_loss_A, 100 * (1 - obsS)[endf_c].sum() / (1 - obsS).sum()))
P('  score if end-fire comps were as accurate as other comps (greedy): %.4f vs observed %.4f' % ((obsA[~endf_c].sum() + endf_c.sum() * obsA[~endf_c].mean()) / obsA.size, obsA.mean()))
out['endfire'] = EF; out['fold'] = dict(wrong_greedy=int(wrong_A.sum()), wrong_selfcal=int(wrong_S.sum()), fold_greedy=int(big_A.sum()), fold_selfcal=int(big_S.sum()), fold_on_endfire_greedy=int((big_A & endf_c).sum()), fold_on_endfire_selfcal=int((big_S & endf_c).sum()))
P('\nweakest-path rank: observed vs joint CRB-limit (wrap / clamp)')
WR = {}
for r_, rn in [(0, 'weakest'), (1, 'middle'), (2, 'strongest')]:
    m6 = np.concatenate([rk == r_, rk == r_], 1)
    WR[rn] = dict(obs_greedy=float(obsA[m6].mean()), obs_selfcal=float(obsS[m6].mean()), joint_wrap=float(PJw[m6].mean()), joint_clamp=float(PJc[m6].mean()), single_clamp=float(PSc[m6].mean()))
    P('  %-9s: observed greedy %.3f, +sc %.3f | joint-wrap %.3f joint-clamp %.3f | single-clamp %.3f' % (rn, obsA[m6].mean(), obsS[m6].mean(), PJw[m6].mean(), PJc[m6].mean(), PSc[m6].mean()))
out['weakest_rank'] = WR
bb = {k_: v[idx] for k_, v in b.items()}
for nm, pp in [('greedy', uv_to_angles(UA, VA)), ('greedy+selfcal', uv_to_angles(US, VS))]:
    S_ = scene_stats([(pp[0][i], pp[1][i]) for i in range(N)], bb); v_ = metric_by_snr(S_, bb['snr'])['mean']['pd_paper']
    P('cross-reference, paper-style Pd of the same 2000 scenes (mean over SNR), %s: %.4f' % (nm, v_)); out.setdefault('paper_style_pd_same_scenes', {})[nm] = v_
np.savez_compressed(f'{OUT}/e5_perscene.npz', idx=idx, snr=snr, okA=okA, okS=okS, errA=errA, errS=errS, UA=UA, VA=VA, US=US, VS=VS, sdu=sdu, sdv=sdv, PJw=PJw, PJc=PJc, amp2=amp2, sig2=sig2_d3)
json.dump(out, open(f'{OUT}/e5_results.json', 'w'), indent=1, default=float)
P('total %.0fs' % (time.time() - t00))
open(f'{OUT}/e5_output.txt', 'w', encoding='utf-8').write('\n'.join(OUTTXT))
