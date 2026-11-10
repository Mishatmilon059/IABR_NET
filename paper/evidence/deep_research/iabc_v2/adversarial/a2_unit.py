"""Adversarial unit checks (no test data): (1) seed disjointness of IABC-v2 training / validation / test streams, including the
numpy SeedSequence aliasing default_rng(s) == default_rng([s, 0]); (2) identical identifiable projection for target (iabr2_sim) and
estimate (iabc2_model.to_ident, torch); (3) sign / conjugation of apply_corr on a noiseless impaired channel, Rx and Tx separately,
and the effect of a deliberately WRONG sign; (4) v2a/v2b Tx-side sign: error with the Tx phase output flipped (seed 9350, 300 scenes,
20 deg / 4 dB, 25 dB, L=3). stdout -> a2_unit.log"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '6'
import sys
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc_v2')
from v2common import *            # noqa
from iabc2_model import load_v2, load_ref, predict, to_ident

# (1) seeds
def first(sd):
    return np.random.default_rng(sd).integers(0, 2**62, 4)
print('(1) SeedSequence aliasing: default_rng(8900) == default_rng([8900,0]) ?', np.array_equal(first(8900), first([8900, 0])))
train = [62999999] + [[62000000, k] for k in range(7000 * 1 + 10)] + [[63000000, k] for k in range(235)] + [[63999999, k] for k in range(5)]
test = [8900 + k for k in range(8)] + [[8900 + k, j] for k in range(8) for j in (1, 3)] + [9300 + k for k in range(8)] + [[9300 + k, j] for k in range(8) for j in (1, 3)]
def key(sd):
    e = np.atleast_1d(np.array(sd, dtype=np.uint64)).tolist()
    while len(e) > 1 and e[-1] == 0: e = e[:-1]        # trailing zeros alias (pool padded with 0)
    return tuple(e)
tk = {key(s) for s in train}; sk = {key(s) for s in test}
print(f'    training/validation seed keys {len(tk)}, test seed keys (8900-8907 orig, 9300-9307 mine) {len(sk)}, intersection {len(tk & sk)}')
# empirical check of the first 4 draws (stream identity) for a sample
F = {tuple(first(s)) for s in train[:3000]}; G = {tuple(first(s)) for s in test}
print(f'    identical first-draw streams between the sets: {len(F & G)}')

# (2) projection
x = np.random.default_rng(9351).standard_normal((50, 32, 2))
a = to_ident(torch.from_numpy(x).float()).double().numpy()
b = ident_vec(x[..., 0], x[..., 1])
print(f'(2) to_ident (estimate) vs ident_vec / S.project_impairment (target) max|d| = {np.abs(a - b).max():.2e}; '
      f'idempotent on targets: {np.abs(ident_vec(b[..., 0], b[..., 1]) - b).max():.2e}')

# (3) sign / conjugation, noiseless
d = make_scenes(9352, 200, 3, 300.0, 30, 6); Ht = to_ant(d['Y'])
def resid(Hc):
    r = Hc / d['H']; lg = np.log(np.abs(r)); ph = np.angle(r / r[:, :1, :1])
    # separable? row part from column 0, col part from row 0
    pr = np.unwrap(ph[:, :, 0], axis=1); pc = np.unwrap(ph[:, 0, :], axis=1)
    return np.rad2deg(np.abs(proj_np(pr)).max()), np.rad2deg(np.abs(proj_np(pc)).max()), DB * np.abs(lg[:, :, 0] - lg[:, :, 0].mean(1, keepdims=True)).max()
print('(3) noiseless 30 deg / 6 dB, residual non-identifiable-free part after apply_corr (max over scenes): row phase deg, col phase deg, row gain dB')
print('    none           ', ['%.3g' % v for v in resid(Ht)])
print('    true target    ', ['%.3g' % v for v in resid(apply_corr(Ht, d['tgt']))])
t2 = d['tgt'].copy(); t2[:, 16:, 0] *= -1
print('    Tx phase flipped', ['%.3g' % v for v in resid(apply_corr(Ht, t2))])
t3 = d['tgt'].copy(); t3[:, :16, 0] *= -1
print('    Rx phase flipped', ['%.3g' % v for v in resid(apply_corr(Ht, t3))])

# (4) the trained models' sign convention
v2a = load_v2(f'{V2CACHE}/v2a_best.pt'); ref = load_ref(f'{V2CACHE}/v2b_best.pt')
d = make_scenes(9350, 300, 3, 25.0, 20, 4); Ht = to_ant(d['Y'])
eA = predict(v2a, Ht); tok, glob, _ = hybrid_inputs(Ht, 3, eA)
with torch.no_grad():
    eB = ref(torch.from_numpy(tok), torch.from_numpy(glob), torch.from_numpy(eA.astype(np.float32))).double().numpy()
for nm, e in [('v2a', eA), ('v2b', eB)]:
    f = e.copy(); f[:, 16:, 0] *= -1; g = e.copy(); g[:, :16, 0] *= -1
    r0 = err_rms(d['tgt'], np.zeros_like(e))[0].mean(); r1 = err_rms(d['tgt'], e)[0].mean()
    print(f'(4) {nm}: RMS phase err deg none {r0:.2f} | as used {r1:.2f} | Tx sign flipped {err_rms(d["tgt"], f)[0].mean():.2f} | Rx sign flipped {err_rms(d["tgt"], g)[0].mean():.2f}')
print('DONE')
