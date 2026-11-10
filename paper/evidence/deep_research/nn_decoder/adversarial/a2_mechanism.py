"""Adversarial check of WHAT the learned decoder computes (lurd.pt, StagedSC(RefinerV3, pre=6, rounds=3, inner=2)) and of training/test overlap.
1. Count MLP (refiner) calls and self-calibration calls per forward, L=3, SC on/off (forward hooks).
2. Are the 'probe second differences' a finite-difference Hessian diagonal? Compare (f(u+d)+f(u-d)-2f)/(d^2 f) with the analytic d2f/du2 / f
   (exp.quantities) at the heatmap seeds of official scenes (first 1000), d = 0.02 and 0.08 rad.
3. Does the MLP step reproduce the classical Newton step? At the seeds (pre-step 1) compare the MLP step with the clipped 2-D Newton step
   (exp.newton_step, clip pi/32 = notebook newton clip 2pi/64).
4. Ablation: same StagedSC structure (6 pre-steps, 3 cyclic rounds x 2 steps, 2 SC iterations, lam=1) with the MLP REPLACED by the
   clipped Newton step (no learned parameter at all): official bank, all 8000 scenes, paired bootstrap vs the learned decoder.
5. Training/test overlap: regenerate the exact exp5 training scene angles (rng default_rng(55), 1500 batches x 64, same call order as
   exp.gen_batch; the CNN does not consume this rng) and measure nearest-neighbour distances to the official-bank scenes; control = 8000
   fresh draws of the same sampler with seed 880001 (new seed, recorded).
usage: python -u a2_mechanism.py > a2_mechanism.log"""
import os, sys, json, time, math
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '6'
LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
sys.path.insert(0, LD)
import numpy as np, torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
import exp5 as E5
torch.set_num_threads(6)
import iabr2_sim as sim
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
CACHE = f'{ROOT}/outputs/cache'
ADC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder_adv'
g = E5.g
OUT = {}

model = E5.StagedSC(E5.RefinerV3(), pre=6, rounds=3, inner=2)
model.load_state_dict(torch.load(f'{LD}/lurd.pt', map_location='cpu')); model.eval()
fb = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
tpsi, tphi, snr = fb['feat'][:, 0, :].astype(np.float64), fb['feat'][:, 1, :].astype(np.float64), fb['meta'][:, 1].astype(int)
print('official eval_bank stored seed:', int(fb['seed']), flush=True)
z = np.load(f'{CACHE}/net__official.npz'); heat, Hc = z['heat'], z['Hc']

# ---- 1. call counts
calls = dict(ref=0, ref_rows=0, cal=0)
h1 = model.ref.register_forward_hook(lambda m, i, o: calls.__setitem__('ref', calls['ref'] + 1) or calls.__setitem__('ref_rows', calls['ref_rows'] + len(o)))
orig_cal = model.calibrate
def cal_count(*a):
    calls['cal'] += 1; return orig_cal(*a)
model.calibrate = cal_count
U0, V0 = g['heat_peaks'](heat[:8].astype(np.float64), 3)
for sc in [True, False]:
    for k in calls: calls[k] = 0
    model.use_sc = sc
    with torch.no_grad():
        model(torch.from_numpy(Hc[:8].astype(np.complex64)), torch.tensor(U0, dtype=torch.float32), torch.tensor(V0, dtype=torch.float32))
    OUT[f'calls_sc_{sc}'] = dict(calls)
    print(f'1. use_sc={sc}: sequential MLP calls per forward {calls["ref"]}, self-calibration (ALS) calls {calls["cal"]}', flush=True)
h1.remove(); model.calibrate = orig_cal

# ---- 2/3. features vs analytic curvature; MLP step vs Newton step (at the seeds, residual = full Hc as in the pre-steps)
n = 1000
U0, V0 = g['heat_peaks'](heat[:n].astype(np.float64), 3)
r = torch.from_numpy(Hc[:n].astype(np.complex64))[:, None].expand(n, 3, 16, 16).reshape(-1, 16, 16)
u = torch.tensor(U0.ravel(), dtype=torch.float32); v = torch.tensor(V0.ravel(), dtype=torch.float32)
with torch.no_grad():
    S, gu, gv, huu, hvv, huv = E5.quantities(r, u, v); f0 = S.abs() ** 2 + 1e-9
    res2 = {}
    for d in [0.02, 0.08]:
        pu = (E5.score_at(r, u + d, v) + E5.score_at(r, u - d, v) - 2 * f0) / f0 / d ** 2
        pv = (E5.score_at(r, u, v + d) + E5.score_at(r, u, v - d) - 2 * f0) / f0 / d ** 2
        a = torch.cat([pu, pv]).numpy(); b = torch.cat([huu / f0, hvv / f0]).numpy()
        res2[d] = dict(pearson=float(np.corrcoef(a, b)[0, 1]), median_rel_err=float(np.median(np.abs(a - b) / (np.abs(b) + 1e-12))))
        print(f'2. probe d={d}: (f(+d)+f(-d)-2f)/(d^2 f) vs analytic Hessian diagonal/f: Pearson {res2[d]["pearson"]:.5f}, median rel. error {res2[d]["median_rel_err"]:.4f}', flush=True)
    OUT['probe_vs_hessian_diag'] = res2
    step = model.ref(r, u, v).numpy()
    nu, nv, ok = E5.newton_step(gu, gv, huu, hvv, huv, math.pi / 32); nt = torch.stack([nu, nv], 1).numpy(); okm = ok.numpy() > 0
    gd = torch.stack([gu / f0, gv / f0], 1).numpy()
    res3 = dict(frac_newton_ok=float(okm.mean()))
    for j, c in enumerate(['du', 'dv']):
        res3[c] = dict(pearson_all=float(np.corrcoef(step[:, j], nt[:, j])[0, 1]), pearson_newton_ok=float(np.corrcoef(step[okm, j], nt[okm, j])[0, 1]),
                       sign_agree_newton_ok=float(np.mean(np.sign(step[okm, j]) == np.sign(nt[okm, j]))),
                       sign_agree_with_gradient=float(np.mean(np.sign(step[:, j]) == np.sign(gd[:, j]))),
                       median_abs_diff_over_clip=float(np.median(np.abs(step[okm, j] - nt[okm, j])) / (math.pi / 32)))
    OUT['mlp_step_vs_newton'] = res3
    print('3. MLP step vs clipped Newton step at the seeds:', json.dumps(res3), flush=True)


# ---- 4. ablation: Newton step in place of the MLP, identical unrolled structure, lam fixed to 1
class NewtonRef(nn.Module):
    def forward(self, r, u, v):
        S, gu, gv, huu, hvv, huv = E5.quantities(r, u, v)
        du, dv, _ = E5.newton_step(gu, gv, huu, hvv, huv, math.pi / 32); return torch.stack([du, dv], 1)


abl = E5.StagedSC(NewtonRef(), pre=6, rounds=3, inner=2); abl.eval(); abl.use_sc = True
# run in float64/complex128: in complex64 the 1e-6 ridge of ls_gains_t is below float32 resolution and the LS solve fails (singular)
# when two Newton-refined paths merge (attempt 1, see a2_mechanism_attempt1.err). The learned model itself is run in complex64 as in s2.
E5.NC = E5.NC.double()
with torch.no_grad():
    abl.lam.fill_(1.0)
abl = abl.double()
print('4. ablation decoder learnable params:', sum(p.numel() for p in abl.parameters()), '(lam only, fixed to 1.0)', flush=True)
p_abl = f'{ADC}/official__newton_in_learned_structure.npz'
if os.path.exists(p_abl):
    zz = np.load(p_abl); pa, pb = zz['psi'], zz['phi']
else:
    t0 = time.time(); oa, ob = [], []
    for s in range(0, 8000, 250):
        Ua, Va = g['heat_peaks'](heat[s:s + 250].astype(np.float64), 3)
        with torch.no_grad():
            U, V = abl(torch.from_numpy(Hc[s:s + 250].astype(np.complex128)), torch.tensor(Ua, dtype=torch.float64), torch.tensor(Va, dtype=torch.float64))
        a, b = g['uv_to_angles'](U.numpy().astype(np.float64), V.numpy().astype(np.float64)); oa.append(a); ob.append(b)
    pa, pb = np.concatenate(oa), np.concatenate(ob); np.savez_compressed(p_abl, psi=pa, phi=pb, n=np.full(8000, 3))
    print(f'   ablation computed in {time.time() - t0:.1f}s', flush=True)
an = np.load(f'{ADC}/official__an_sc.npz')
heat_re = np.load(f'{ADC}/official__heat.npz')


def stats(ps, ph):
    S = np.zeros((8000, 2))
    for i in range(8000):
        G = np.stack([tpsi[i], tphi[i]], 1); P = np.stack([ps[i, :3], ph[i, :3]], 1)
        rr, cc = linear_sum_assignment(np.linalg.norm(G[:, None] - P[None], axis=2))
        ok = np.abs(np.degrees(np.angle(np.exp(1j * (G[rr] - P[cc]))))) <= 1.0; S[i] = [ok.sum(), (~ok).sum()]
    return S


def pdm(S):
    return float(np.mean([S[snr == s, 0].sum() / S[snr == s].sum() for s in sorted(set(snr.tolist()))]))


Sd = {'newton_in_structure': stats(pa, pb), 'learned_AN_sc': stats(an['psi'], an['phi']), 'classical_heat': stats(heat_re['psi'], heat_re['phi'])}
rng = np.random.default_rng(0); acc = {k: np.zeros(2000) for k in Sd}; grps = sorted(set(snr.tolist()))
for s in grps:
    ix = np.where(snr == s)[0]; R = rng.integers(0, len(ix), (2000, len(ix)))
    for k, S in Sd.items():
        acc[k] += S[ix, 0][R].sum(1) / (S[ix, 0][R].sum(1) + S[ix, 1][R].sum(1))
acc = {k: v / len(grps) for k, v in acc.items()}
res4 = {k: pdm(S) for k, S in Sd.items()}
for a, b in [('learned_AN_sc', 'newton_in_structure'), ('newton_in_structure', 'classical_heat'), ('learned_AN_sc', 'classical_heat')]:
    d = acc[a] - acc[b]; res4[f'{a}-{b}'] = [res4[a] - res4[b], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
OUT['ablation_official'] = res4
print('4. official Pd (paper-style, mean over SNR):', json.dumps(res4), flush=True)

# ---- 5. training/test overlap
t0 = time.time(); rng = np.random.default_rng(55); TP, TF = [], []
for s in range(1500):
    B = 64
    snr_b = rng.integers(-15, 25, B).astype(float); clean = rng.uniform(size=B) < 0.2
    pdg = np.where(clean, 0.0, rng.uniform(0, 8, B)); gdb = np.where(clean, 0.0, rng.uniform(0, 3, B))
    psi, phi, alpha = sim.sample_paths(np.full(B, 3), rng, lmax=3)
    sim.simulate(psi, phi, alpha, snr_b, rng, phase_deg_max=pdg, gain_db_max=gdb)
    TP.append(psi); TF.append(phi)
TP, TF = np.concatenate(TP), np.concatenate(TF)
print(f'5. regenerated {len(TP)} exp5 training scenes in {time.time() - t0:.0f}s; train SNR range used by gen_batch: -15..24 dB', flush=True)


def key(psi, phi):     # order-free scene descriptor: paths sorted by psi, degrees
    o = np.argsort(psi, 1); return np.degrees(np.concatenate([np.take_along_axis(psi, o, 1), np.take_along_axis(phi, o, 1)], 1))


tree = cKDTree(key(TP, TF))
d_off, _ = tree.query(key(tpsi, tphi), p=np.inf)
crng = np.random.default_rng(880001); cp, cf, _ = sim.sample_paths(np.full(8000, 3), crng, lmax=3)
d_ctl, _ = tree.query(key(cp, cf), p=np.inf)
res5 = dict(n_train=int(len(TP)), official=dict(min=float(d_off.min()), median=float(np.median(d_off)), frac_lt_0p01deg=float(np.mean(d_off < 0.01)), frac_lt_0p1deg=float(np.mean(d_off < 0.1))),
            control_fresh_draws_seed880001=dict(min=float(d_ctl.min()), median=float(np.median(d_ctl)), frac_lt_0p01deg=float(np.mean(d_ctl < 0.01)), frac_lt_0p1deg=float(np.mean(d_ctl < 0.1))))
OUT['train_test_overlap_Linf_deg'] = res5
print('5. nearest exp5-training scene (max-abs angle difference over the 6 sorted angles, deg):', json.dumps(res5), flush=True)
json.dump(OUT, open(f'{HERE}/a2_mechanism.json', 'w'), indent=1, default=float)
print('done', flush=True)
