"""Learning-threshold checks for the SignalNet paper (Dreifuerst & Heath, IEEE TSP 2022) and the same test for our benchmark.

Part A re-computes the paper's "worst-case learning thresholds" (best input-independent estimator under each loss) from the
paper's own definitions (Sec. VI, VIII; Algorithm 1, N = 64, M = 5):
  A1 detection loss (11) with m ~ uniform{1..5}: continuous constant (paper: m_hat ~ 3.69, L ~ 1.67) and integer constant;
     also the MSE threshold Var(m).
  A2 frequency: Monte-Carlo of Algorithm 1 (with its do-while rejection) vs the paper's closed form (45).
  A3 amplitude Var(U(0.1, 1)), phase Var(U(0, 2 pi)).
Part B: blind (input-independent) Pd for our benchmark geometry (L = 3 paths, angles uniform in [0, pi]^2 with min separation
  pi/6 = the exact sampler DG.generate_points; metric = common.scene_stats / metric_from_stats, paper-style Pd, 1 deg).
  Constant 3-path guesses are selected on one half of the scenes and evaluated on the other half (no selection optimism).
Seeds: 81000101 (MC of Algorithm 1), 81000202 (scene angles) - validation range, unused elsewhere.
Memory-light: numpy only in chunks; 2 threads.
Usage: python -I signalnet_thresholds.py   (stdout -> signalnet_thresholds.log)
"""
import os, sys, time
os.environ['OMP_NUM_THREADS'] = '2'; os.environ['MKL_NUM_THREADS'] = '2'; os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
db = lambda x: 10 * np.log10(x)
t0 = time.time()

# ------------------------------------------------------------------ A1 detection threshold
M = 5; ms = np.arange(1, M + 1)


def Ldet(m, c):
    d = m - c
    return np.where(d >= 0, np.exp(d) - 1, 0.5 * d ** 2)


cs = np.linspace(1, 5, 400001)
Lc = np.array([Ldet(ms, c).mean() for c in cs[::100]])
c_star = cs[::100][Lc.argmin()]
fine = np.linspace(c_star - 0.01, c_star + 0.01, 20001)
Lf = np.array([Ldet(ms, c).mean() for c in fine]); c_star = fine[Lf.argmin()]; L_star = Lf.min()
print('A1 detection loss (11), m ~ U{1..5}')
print('   continuous constant: m_hat* = %.4f, threshold = %.4f   (paper: 3.69, 1.67)' % (c_star, L_star))
from scipy.special import lambertw
alpha = 2.0; arg = (np.exp(4 - alpha) + np.exp(5 - alpha)) / 3
print('   Lambert-W formula (38): W(%.4f) + alpha = %.4f' % (arg, lambertw(arg).real + alpha))
for c in ms:
    print('   integer constant m_hat = %d: E[L_det] = %.4f' % (c, Ldet(ms, c).mean()))
print('   -> an integer-valued input-independent detector can reach at best %.4f (always predict %d); mixing 3 and 4 is linear in the mixing weight, so it cannot reach %.2f'
      % (min(Ldet(ms, c).mean() for c in ms), ms[np.argmin([Ldet(ms, c).mean() for c in ms])], L_star))
print('   MSE threshold (blind, continuous) = Var(m) = %.4f; best integer constant (3) = %.4f' % (ms.var(), ((ms - 3) ** 2).mean()))

# ------------------------------------------------------------------ A2 frequency threshold (Algorithm 1, N = 64)
N = 64; rng = np.random.default_rng(81000101); NMC = 1_000_000
print('\nA2 frequency threshold, Algorithm 1 Monte Carlo (%d accepted draws per m, N = %d)' % (NMC, N))
var_w = 2.5 / N * (1 - 2 / np.pi)
for m in range(1, M + 1):
    acc = []; n_acc = 0; n_tot = 0
    while n_acc < NMC:
        B = 200_000
        w0 = rng.uniform(0, 0.25, B)
        f = np.empty((B, m)); f[:, 0] = w0
        for i in range(1, m):
            f[:, i] = w0 + i / N + np.abs(rng.normal(0, np.sqrt(2.5 / N), B))
        ok = f.max(1) <= 0.5
        n_tot += B; acc.append(f[ok]); n_acc += ok.sum()
    F = np.concatenate(acc)[:NMC]
    mc = F.var(0).mean()
    paper = 1 / 64 + (1 - 1 / m) * var_w
    noreject = 1 / 192 + (1 - 1 / m) * var_w
    print('   m=%d: MC threshold %.5f (%.2f dB) | paper eq.(45) %.5f (%.2f dB) | same formula with Var(w0)=1/192, no rejection %.5f (%.2f dB) | acceptance %.3f'
          % (m, mc, db(mc), paper, db(paper), noreject, db(noreject), n_acc / n_tot))
print('   note: 1/64 = 0.125^2 = (E[w0])^2, whereas Var(U(0, 0.25)) = 0.25^2/12 = 1/192 = %.5f (%.2f dB)' % (1 / 192, db(1 / 192)))

# ------------------------------------------------------------------ A3 amplitude / phase
va = 0.9 ** 2 / 12; vp = (2 * np.pi) ** 2 / 12
print('\nA3 amplitude: Var(U(0.1,1)) = %.4f (%.2f dB)   (paper: 0.0675; dashed line in the plotted figure digitised at -11.25 dB)' % (va, db(va)))
print('   phase: Var(U(0,2pi)) = %.4f = pi^2/3 (%.2f dB)' % (vp, db(vp)))
print('   time %.1f s' % (time.time() - t0))

# ------------------------------------------------------------------ B blind Pd threshold for our benchmark
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import common as C
import dldoa_dataset_generation as DG
NS = 4000
rng = np.random.default_rng(81000202)
psi = np.zeros((NS, 3)); phi = np.zeros((NS, 3))
for b in range(NS):
    pts = DG.generate_points(3, np.pi / 6, rng=rng)
    phi[b] = [p[0] for p in pts]; psi[b] = [p[1] for p in pts]
bank = dict(psi=psi, phi=phi, L=np.full(NS, 3))
print('\nB blind (input-independent) Pd on our geometry: %d scenes, L=3, angles U[0,pi]^2, min separation pi/6' % NS)
print('   marginal check: psi mean %.1f deg, std %.1f deg (uniform [0,180]: 90, 52.0)' % (np.degrees(psi).mean(), np.degrees(psi).std()))
sel, ev = np.arange(0, NS, 2), np.arange(1, NS, 2)


def pd_const(P, Q, idx):
    preds = [(np.array(P), np.array(Q))] * len(idx)
    S = C.scene_stats(preds, dict(psi=psi[idx], phi=phi[idx], L=bank['L'][idx]))
    return C.metric_from_stats(S, np.arange(len(idx)))['pd_paper']


cands = {'3 x centre (90,90)': ([np.pi / 2] * 3, [np.pi / 2] * 3)}
r2 = np.random.default_rng(81000203)
for k in range(40):
    cands['random set %02d' % k] = (list(r2.uniform(0, np.pi, 3)), list(r2.uniform(0, np.pi, 3)))
# a structured spread guess (vertices of an equilateral-ish triangle around the centre)
cands['spread triangle'] = ([np.pi / 2, np.pi / 4, 3 * np.pi / 4], [3 * np.pi / 4, np.pi / 4, np.pi / 4])
sc = {k: pd_const(*v, sel) for k, v in cands.items()}
best = max(sc, key=sc.get)
print('   selection half: centre %.4f | spread triangle %.4f | best of 40 random sets %.4f (%s) | median random %.4f'
      % (sc['3 x centre (90,90)'], sc['spread triangle'], sc[best], best, np.median([v for k, v in sc.items() if k.startswith('random')])))
print('   held-out half:  best-selected guess %s -> blind Pd = %.4f; centre -> %.4f' % (best, pd_const(*cands[best], ev), pd_const(*cands['3 x centre (90,90)'], ev)))
print('   analytic scale: P(|err| <= 1 deg) for one uniform component on [0,180] deg = 2/180 = %.4f' % (2 / 180))
print('   total time %.1f s' % (time.time() - t0))
