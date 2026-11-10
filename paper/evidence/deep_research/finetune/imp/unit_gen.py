"""Unit test of the vectorised training generator icom.synth against stress_sim.generate (the test-bank simulator):
same base scenes (seed 71000002, n=64, L=3, SNR 15) and the same impairment draws (stress_sim.imp_rng conventions) -> Y must be identical.
Also: per-sample severity mixing (a batch whose samples carry different families) equals the per-family results, imp_batch timing,
family / mask counts and the IABC-target ranges.  usage: python unit_gen.py  (stdout -> unit_gen.log)"""
from icom import *

b = SS.base_scenes(64, 3, 15.0, 71000002)
B = 64; z = np.zeros(B)
args = (b['psi'], b['phi'], b['alpha'], b['noise'], b['Ur'], b['Ut'], b['Gr'], b['Gt'])
res = {}
for kind, sev in [('clean', None), ('M1', 0.3), ('M1', 0.05), ('M2', 0.2), ('M2', 0.02), ('M3', (45.0, 0.0)), ('M3', (30.0, 4.0)), ('M3', (0.0, 6.0)),
                  ('M6', 30.0), ('M6', 5.0)]:
    Yref = SS.generate(b, kind, sev)
    kw = {}
    if kind == 'M1':
        g = SS.imp_rng(b['seed'], 'M1'); kw = dict(rho=np.full(B, sev), th=g.uniform(0, 2 * np.pi, (4, B)))
    if kind == 'M2':
        g = SS.imp_rng(b['seed'], 'M2'); kw = dict(sig2=np.full(B, sev), zr=g.standard_normal((B, 16)), zt=g.standard_normal((B, 16)))
    if kind == 'M3':
        kw = dict(pdm=np.full(B, sev[0]), gdm=np.full(B, sev[1]))
    if kind == 'M6':
        g = SS.imp_rng(b['seed'], 'M6'); kw = dict(sig6=np.full(B, sev), z6=g.standard_normal((B, 16, 16)))
    Y, _, _ = synth(*args, **kw)
    d = np.abs(Y - Yref).max(); res[(kind, str(sev))] = (Y, kw)
    print(f'{kind:5s} {str(sev):12s} max|synth - stress_sim.generate| = {d:.2e}  (max|Y| {np.abs(Yref).max():.1f})', flush=True)

# mixed batch: first 16 samples M1 0.3, next 16 M2 0.2, next 16 M3 (45,0), last 16 M6 30 -> must equal the per-family outputs
sel = [('M1', '0.3'), ('M2', '0.2'), ('M3', '(45.0, 0.0)'), ('M6', '30.0')]
kw = dict(rho=np.zeros(B), th=res[sel[0]][1]['th'], sig2=np.zeros(B), zr=res[sel[1]][1]['zr'], zt=res[sel[1]][1]['zt'], pdm=np.zeros(B), gdm=np.zeros(B),
          sig6=np.zeros(B), z6=res[sel[3]][1]['z6'])
kw['rho'][:16] = 0.3; kw['sig2'][16:32] = 0.2; kw['pdm'][32:48] = 45.0; kw['sig6'][48:] = 30.0
Ym, _, _ = synth(*args, **kw)
for k, (nm, sv) in enumerate(sel):
    print(f'mixed batch, block {k} ({nm} {sv}): max|diff| vs single-family = {np.abs(Ym[16 * k:16 * k + 16] - res[(nm, sv)][0][16 * k:16 * k + 16]).max():.2e}')

t = []
for s in range(5):
    t0 = time.time(); y, h, im, mk, fam = imp_batch(np.random.default_rng([71000002, s]), 128); t.append(time.time() - t0)
print(f'imp_batch(128): {np.mean(t[1:]):.3f} s; shapes y {tuple(y.shape)} heat {tuple(h.shape)} imp {tuple(im.shape)}')
print('family counts (0 orig, 1 M1, 2 M2, 3 M3, 6 M6):', {int(k): int((fam == k).sum()) for k in np.unique(fam)}, '| mask sum', int(mk.sum()))
imn = im.numpy(); ph = np.rad2deg(np.abs(imn[..., 0])).max(1); gg = (20 / np.log(10)) * np.abs(imn[..., 1]).max(1)
for k in [0, 3]:
    print(f'fam {k}: identifiable target max |phase| deg median {np.median(ph[fam == k]):.1f} max {ph[fam == k].max():.1f}; max |gain| dB median {np.median(gg[fam == k]):.2f} max {gg[fam == k].max():.2f}')
print('fam 1/2/6 targets (masked) all zero:', bool(np.abs(imn[np.isin(fam, [1, 2, 6])]).max() < 1e-6))
print('determinism: same seed twice identical:', bool(torch.equal(imp_batch(np.random.default_rng([71000002, 9]), 64)[0], imp_batch(np.random.default_rng([71000002, 9]), 64)[0])))
