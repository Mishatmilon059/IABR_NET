"""Unit tests for ucore.py (stdout -> unit_tests.log):
 T1 classical-equivalent configuration (classical=True, hypothesis H1 only, no polish, no self-cal, true sigma) reproduces
    notebook NOMP (os 4, rc 3) on 200 official scenes (angles and Pd);
 T2 gradients reach every learned module (design A, small batch, mixed ADC / ideal);
 T3 identical inputs -> identical outputs (eval mode, two calls);
 T4 outputs sane (finite, Pd of untrained A on 64 ideal 15 dB scenes close to NOMP's)."""
import os
os.environ.setdefault('THREADS', '4')
from ucore import *

torch.manual_seed(0)
b = C.load_bank('official'); idx = np.concatenate([np.where(b['snr'].astype(int) == s)[0][:25] for s in sorted(set(b['snr'].astype(int).tolist()))])
z = np.load(f'{C.CACHE}/net__official.npz'); Ht = z['Ht'][idx].astype(np.complex128); Y = to_beam(Ht)
tb = sub(dict(psi=b['psi'], phi=b['phi'], L=b['L'], snr=b['snr'].astype(float)), idx)
t0 = time.time(); Un, Vn = C.nomp_greedy(Ht, 3, 4, 3); tn = time.time() - t0
m = UNOMP('A'); m.cfg.update(classical=True, hyps=['H1'], polish=False, sc=False)
t0 = time.time(); U, V, _ = decode_np(m, Y, np.zeros(len(Y), int), 3, snr=tb['snr']); tc = time.time() - t0
du = np.abs(np.angle(np.exp(1j * (U - Un)))).max(); dv = np.abs(np.angle(np.exp(1j * (V - Vn)))).max()
Sn = stats(C.uv_to_angles(Un, Vn), tb); Sc = stats(C.uv_to_angles(U, V), tb)
print(f'T1 classical-equivalent core vs notebook NOMP os4 rc3 on 200 official scenes: max |du| {du:.2e} |dv| {dv:.2e} rad; '
      f'Pd NOMP {pd(Sn, tb["snr"]):.4f} core {pd(Sc, tb["snr"]):.4f}; time numpy {tn:.1f}s torch {tc:.1f}s', flush=True)
assert du < 1e-6 and dv < 1e-6

# T1b: classical core with polish (Gaussian LM, true sigma) and H1+H2 (heat seeds from adc_ft CNN) -> report only
m.cfg.update(polish=True, hyps=['H1', 'H2'])
U2, V2, pk = decode_np(m, Y, np.zeros(len(Y), int), 3, snr=tb['snr'])
print(f'T1b classical core + LM polish + heat hypothesis (lowest NLL): Pd {pd(stats(C.uv_to_angles(U2, V2), tb), tb["snr"]):.4f} (H2 picked {pk.mean():.2f})', flush=True)

# T2 gradients
rng = np.random.default_rng([79_999_999, 1])
d1 = make_mix(rng, 6, 79_999_999, comp='adc'); d2 = make_mix(rng, 6, 79_999_998, comp='ideal')
Yb = np.concatenate([d1['Y'], d2['Y']]); bits = np.concatenate([d1['bits'], d2['bits']]); snr = np.concatenate([d1['snr'], d2['snr']])
mA = UNOMP('A'); mA.train(); mA.cfg.update(dtype=torch.float32, train_stop=True)
with torch.no_grad():                       # make every learned gate non-zero so that gradients can reach each module
    mA.hyper.mlp[-1].bias.add_(0.3); mA.step.net[-1].weight.normal_(0, 1e-3); mA.fusion.c3.weight.normal_(0, 1e-3)
bt = prep_batch(Yb, bits, snr)
o = mA(bt, 3)
Ut = torch.from_numpy(np.pi * np.cos(np.concatenate([d1['psi'], d2['psi']]))); Vt = torch.from_numpy(np.pi * np.cos(np.concatenate([d1['phi'], d2['phi']])))
loss = 0
for h in o['hyps']:
    loss = loss + ((wrap(o['res'][h]['U'] - Ut.to(o['res'][h]['U'].dtype)) ** 2).sum() + (wrap(o['res'][h]['V'] - Vt.to(o['res'][h]['V'].dtype)) ** 2).sum())
loss = loss + o['sel_logits'].sum() + sum(x.sum() for x in o['stop_logits']) + sum(f[0].sum() for f in o['dets']) + o['front']['logsig'].sum() + o['front']['heat'].mean()
loss.backward()
for nm, mod in [('front.cnn', mA.front.cnn), ('front.film', mA.front.film), ('front.scene', mA.front.scene), ('hyper', mA.hyper), ('fusion', mA.fusion),
                ('step', mA.step), ('stop', mA.stop), ('selector', mA.selector)]:
    gn = sum(float(p.grad.norm()) for p in mod.parameters() if p.grad is not None); nz = all(p.grad is not None for p in mod.parameters())
    print(f'T2 grad {nm:12s} norm {gn:.3e} all-params-have-grad {nz}', flush=True)
    assert gn > 0 and math.isfinite(gn)
# T3 determinism
mA.eval(); mA.cfg.update(train_stop=False, dtype=torch.float64)
with torch.no_grad():
    o1 = mA(prep_batch(Yb, bits, snr), 3); o2 = mA(prep_batch(Yb, bits, snr), 3)
print('T3 identical outputs:', bool(torch.equal(o1['U'], o2['U']) and torch.equal(o1['V'], o2['V'])), flush=True)
# T4 sanity
rng = np.random.default_rng([79_999_999, 2]); d = make_mix(rng, 64, 79_999_997, comp='ideal'); d['snr'][:] = 15
bb = gen_scenes(np.random.default_rng([79_999_999, 3]), 64, 3, np.full(64, 15.0)); Yc = SS.generate(bb, 'clean')
m0 = UNOMP('A'); U, V, pk = decode_np(m0, Yc, np.zeros(64, int), 3)
Un, Vn = C.nomp_greedy(to_ant_np := SS.to_ant(Yc), 3, 4, 3)
print(f'T4 untrained A (sigma_hat from untrained scene head) Pd {pd(stats(C.uv_to_angles(U, V), bb)):.4f} vs NOMP {pd(stats(C.uv_to_angles(Un, Vn), bb)):.4f}; finite {np.isfinite(U).all()}', flush=True)
print('params A', sum(p.numel() for p in mA.parameters()), 'B', sum(p.numel() for p in UNOMP('B').parameters()))
print('unit tests done')
