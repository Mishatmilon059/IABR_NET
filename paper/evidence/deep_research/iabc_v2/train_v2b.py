"""Train the IABC-v2b refiner (iabc2_model.Refiner) on cache/v2b_data (gen_v2b.py). Loss: MSE (rad, neper) of the final identifiable
estimate vs impairment_targets. AdamW 1e-3 (warm-up 200, cosine to 1e-5), wd 0.01, clip 1, batch 256, STEPS steps, shuffling seed
64000000. Model selection on the val chunks (seed 63999999). Resumable (cache/v2b_ckpt.pt). Best -> cache/v2b_best.pt."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '6'
import glob as G, json, math, time
import numpy as np, torch
torch.set_num_threads(6)
from iabc2_model import Refiner, n_params
HERE = os.path.dirname(os.path.abspath(__file__)); CA = f'{HERE}/cache'; D = os.environ.get('V2B_DATA', f'{CA}/v2b_data')   # data moved to the C: scratchpad when D: ran out of space (gen_v2b_part2.err)
STEPS, BS, LR, WARM = int(os.environ.get('STEPS', 6000)), 256, 1e-3, 200
CK, BEST = f'{CA}/v2b_ckpt.pt', f'{CA}/v2b_best.pt'


def load(prefix):
    fs = sorted(G.glob(f'{D}/{prefix}_*.npz')); z = [np.load(f) for f in fs]
    return {k: np.concatenate([x[k] for x in z]) for k in z[0].files}, len(fs)


tr, ntr = load('tr'); va, nva = load('val')
print(f'train chunks {ntr} ({len(tr["tgt"])} scenes), val chunks {nva} ({len(va["tgt"])} scenes)', flush=True)
T = {k: torch.from_numpy(v.astype(np.float32)) for k, v in tr.items() if k in ('tok', 'glob', 'eA', 'tgt')}
Vt = {k: torch.from_numpy(v.astype(np.float32)) for k, v in va.items() if k in ('tok', 'glob', 'eA', 'tgt')}


def lr_at(s):
    if s < WARM:
        return LR * (s + 1) / WARM
    t = (s - WARM) / max(1, STEPS - WARM)
    return 1e-5 + 0.5 * (LR - 1e-5) * (1 + math.cos(math.pi * min(1.0, t)))


def val(m):
    m.eval()
    with torch.no_grad():
        p = torch.cat([m(Vt['tok'][i:i + 500], Vt['glob'][i:i + 500], Vt['eA'][i:i + 500]) for i in range(0, len(Vt['tgt']), 500)])
    e = ((p - Vt['tgt']) ** 2).numpy(); b0 = (Vt['tgt'] ** 2).numpy(); eA = ((Vt['eA'] - Vt['tgt']) ** 2).numpy()
    eALS = ((torch.from_numpy(va['tok'][..., 2:4]) - Vt['tgt']) ** 2).numpy()
    hi = (va['snr'] >= 5) & ((va['pd'] >= 8) | (va['gd'] >= 3)); r = dict(val=float(e.mean()))
    for nm, q in [('all', np.ones(len(e), bool)), ('snr>=5_imp', hi), ('snr<0', va['snr'] < 0)]:
        for j, c in enumerate(['ph', 'g']):
            den = b0[q, :, j].sum()
            r[f'rem_{c}_{nm}'] = float(100 * (1 - e[q, :, j].sum() / den)); r[f'remA_{c}_{nm}'] = float(100 * (1 - eA[q, :, j].sum() / den))
            r[f'remALS_{c}_{nm}'] = float(100 * (1 - eALS[q, :, j].sum() / den))
    clean = (va['pd'] == 0) & (va['gd'] == 0)
    r['clean_rms_ph_deg'] = float(np.rad2deg(np.sqrt(e[clean, :, 0].mean()))); r['clean_rms_g_db'] = float(20 / np.log(10) * np.sqrt(e[clean, :, 1].mean()))
    return r


m = Refiner(); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01); start, hist, best = 0, [], 1e9
if os.path.exists(CK):
    s = torch.load(CK); m.load_state_dict(s['model']); opt.load_state_dict(s['opt']); start, hist, best = s['step'], s['hist'], s['best']
print(f'Refiner {m.cfg}: {n_params(m):,} params; steps {STEPS}', flush=True)
N = len(T['tgt']); acc, na, t0 = 0.0, 0, time.time()
for step in range(start, STEPS):
    idx = torch.from_numpy(np.random.default_rng([64000000, step]).integers(0, N, BS))
    for g in opt.param_groups:
        g['lr'] = lr_at(step)
    m.train(); loss = ((m(T['tok'][idx], T['glob'][idx], T['eA'][idx]) - T['tgt'][idx]) ** 2).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    acc += loss.item(); na += 1
    if (step + 1) % 250 == 0 or step + 1 == STEPS:
        r = dict(step=step + 1, train=acc / na, s_per_step=(time.time() - t0) / na, **val(m)); acc, na, t0 = 0.0, 0, time.time()
        if r['val'] < best:
            best = r['val']; torch.save(dict(model=m.state_dict(), cfg=m.cfg, step=step + 1, val=best), BEST)
        hist.append(r); print(json.dumps(r), flush=True)
        torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step + 1, hist=hist, best=best), CK)
        json.dump(hist, open(f'{HERE}/train_v2b_history.json', 'w'), indent=0)
print('DONE', flush=True)
