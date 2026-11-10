"""Derived quantities quoted in Section III (System Model). Run: python sec3_derived_checks.py > sec3_derived_checks.log
Sampler: DG.generate_points of D:/ai_ml_project/IABR_Net_TestSuite/dldoa_dataset_generation.py (as used by iabr2_sim.sample_paths)."""
import sys, math, numpy as np
sys.path.insert(0, r'D:/ai_ml_project/IABR_Net_TestSuite')
import dldoa_dataset_generation as DG
N = 16
rng = np.random.default_rng(12345)

print('== 1. ramp of iid U(-d,d) phases (LS slope over n-7.5), N=16 ==')
S = float(np.sum((np.arange(N) - 7.5) ** 2)); print('sum (n-7.5)^2 =', S)
for d in (1, 2, 5, 8, 15):
    sd = math.radians(d) / math.sqrt(3) / math.sqrt(S)
    bias = lambda psi_deg: math.degrees(sd / (math.pi * math.sin(math.radians(psi_deg))))
    # Monte Carlo check
    eps = rng.uniform(-1, 1, (200000, N)) * math.radians(d); n = np.arange(N) - 7.5
    sl = eps @ n / S
    print(f'd={d:2d} deg: slope sd formula {sd:.5f} rad/antenna, MC {sl.std():.5f}; AoA bias sd at psi=90: {bias(90):.3f} deg, psi=10: {bias(10):.3f} deg, psi=5: {bias(5):.3f} deg')

print('== 2. gain-error mean power factor per side E[g^2]=sinh(x)/x, x=gamma ln10/10; both sides squared ==')
for g in (1, 2, 3, 4):
    x = g * math.log(10) / 10; e = math.sinh(x) / x
    print(f'gamma_max={g} dB: per side {e:.4f}, both sides {e*e:.4f} = {10*math.log10(e*e):+.2f} dB')

print('== 3. realised per-scene signal power ||H||_F^2/256 (ideal hardware, L=3), 4000 scenes ==')
def scene(L):
    pts = DG.generate_points(L, math.pi / 6, rng=rng)
    a = np.sqrt(1 / L) * (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
    a = a[np.argsort(-np.abs(a))]
    phi = np.array([p[0] for p in pts]); psi = np.array([p[1] for p in pts])
    return phi, psi, a
P = []
for _ in range(4000):
    phi, psi, a = scene(3)
    n = np.arange(N)
    H = sum(al * np.exp(-1j * n * math.pi * math.cos(ps))[:, None] * np.exp(1j * n * math.pi * math.cos(ph))[None, :] for al, ps, ph in zip(a, psi, phi))
    P.append(np.sum(np.abs(H) ** 2) / 256)
P = np.array(P); dB = 10 * np.log10(P)
print(f'mean power {P.mean():.3f}; dB: mean {dB.mean():.2f}, sd {dB.std():.2f}, 5th {np.percentile(dB,5):.2f}, 95th {np.percentile(dB,95):.2f}')

print('== 4. sub-cell frequency separation, L=3, 40000 scenes; circular distance max(|du|,|dv|) in (u,v) ==')
def cd(a, b): d = abs(a - b) % (2 * math.pi); return min(d, 2 * math.pi - d)
cell = 2 * math.pi / N; c1 = c2 = tot = 0
for _ in range(40000):
    phi, psi, a = scene(3)
    u = math.pi * np.cos(psi); v = math.pi * np.cos(phi)
    m = min(max(cd(u[i], u[j]), cd(v[i], v[j])) for i in range(3) for j in range(i + 1, 3))
    c1 += m < cell; c2 += m < cell / 2
    tot += 1
print(f'share of scenes with a pair closer than one cell: {c1/tot:.4f}; closer than half a cell: {c2/tot:.4f}')

print('== 5. end-fire aliasing: circular distance in u between psi and pi-psi, and 1-degree tolerance in u ==')
for d in (1, 2, 3, 5):
    u1 = math.pi * math.cos(math.radians(d)); u2 = math.pi * math.cos(math.radians(180 - d))
    print(f'psi={d} deg: circular |u(psi)-u(pi-psi)| = {cd(u1,u2):.5f} rad')
print('1-degree tolerance in u at psi=0 (one-sided): pi(1-cos 1deg) =', math.pi * (1 - math.cos(math.radians(1))))
print('1-degree tolerance in u at broadside: pi*pi/180 =', math.pi * math.pi / 180, ' cell', cell, ' ratio', math.pi * math.pi / 180 / cell)
print('single-path sd of u at 15 dB (spec 6.6 formula) not recomputed here')

print('== 6. DFT codebooks with P=Q=2N=32 (N=16): tight frame, scaled left inverse, noise variance ==')
Pn = 32; n = np.arange(N)
W = np.exp(1j * 2 * np.pi * np.outer(n, np.arange(Pn)) / Pn) / math.sqrt(N)
F = np.exp(-1j * 2 * np.pi * np.outer(n, np.arange(Pn)) / Pn) / math.sqrt(N)
print('max|W W^H - (P/N) I| =', np.abs(W @ W.conj().T - (Pn / N) * np.eye(N)).max())
Hm = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
Y = W.conj().T @ Hm @ F
Hr = (N / Pn) ** 2 * (W @ Y @ F.conj().T)
print('max|scaled inverse - H| =', np.abs(Hr - Hm).max())
Z = (rng.standard_normal((20000, Pn, Pn)) + 1j * rng.standard_normal((20000, Pn, Pn))) / math.sqrt(2)
Zt = (N / Pn) ** 2 * np.einsum('nq,bqp,mp->bnm', W, Z, F.conj())
print('var of scaled noise entry', np.mean(np.abs(Zt) ** 2), ' expected sigma^2 N^2/(PQ) =', (N / Pn) ** 2, '(for sigma^2=1)')
print('off-diagonal correlation of scaled noise (max |E z_i conj z_j|, i!=j, sample):',
      np.abs(np.corrcoef(Zt.reshape(20000, -1)[:, :40].T)[np.triu_indices(40, 1)]).max())
