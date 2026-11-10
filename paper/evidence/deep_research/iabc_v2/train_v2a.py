"""Train IABC-v2a (blind antenna-token transformer, iabc2_model.IABCv2, d=128, 4 layers) ALONE with MSE on impairment_targets.
CNN trunk is not involved (frozen / unused). Fresh synthetic batches: stream seed 62000000 (batch k <- default_rng([62000000, k])),
held-out validation set: default_rng(62999999), 2048 scenes, same distribution (model selection only; test seeds are 8900+).
Loss = mean over (B,32,2) of squared error in (rad, neper) on the identifiable subspace.
AdamW lr 5e-4 (warm-up 300 steps, cosine to 1e-5), wd 0.01, grad clip 1.0, batch 256, STEPS steps.
Resumable: cache/v2a_ckpt.pt every 250 steps (model, opt, step, history); best-val weights -> cache/v2a_best.pt.
usage (detached): python -u train_v2a.py > train_v2a.log"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '6'
import time, json, math
import numpy as np, torch
torch.set_num_threads(6)
from iabc2_model import *

HERE = os.path.dirname(os.path.abspath(__file__)); CA = f'{HERE}/cache'; os.makedirs(CA, exist_ok=True)
STEPS, BS, LR, LRMIN, WARM = int(os.environ.get('STEPS', 8000)), 256, 5e-4, 1e-5, 300
ARCH = dict(d=128, layers=4, heads=4, ff=256)
SEED, VSEED = 62000000, 62999999
CK, BEST, HIST = f'{CA}/v2a_ckpt.pt', f'{CA}/v2a_best.pt', f'{HERE}/train_v2a_history.json'


def lr_at(s):
    if s < WARM:
        return LR * (s + 1) / WARM
    t = (s - WARM) / max(1, STEPS - WARM)
    return LRMIN + 0.5 * (LR - LRMIN) * (1 + math.cos(math.pi * min(1.0, t)))


def main():
    torch.manual_seed(0)
    vb = make_train_batch(np.random.default_rng(VSEED), 2048)
    vH = torch.complex(torch.from_numpy(vb['Hr']), torch.from_numpy(vb['Hi'])); vt = torch.from_numpy(vb['tgt'])
    vsnr, vpd, vgd = vb['snr'], vb['pd'], vb['gd']
    m = IABCv2(**ARCH); opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    start, hist, best = 0, [], 1e9
    if os.path.exists(CK):
        s = torch.load(CK, map_location='cpu'); m.load_state_dict(s['model']); opt.load_state_dict(s['opt'])
        start, hist, best = s['step'], s['hist'], s['best']
        print(f'resumed at step {start}', flush=True)
    print(f'IABC-v2a {ARCH}: {n_params(m):,} params; steps {STEPS}, batch {BS}', flush=True)
    dl = torch.utils.data.DataLoader(Stream(SEED, start, BS), batch_size=None, num_workers=2, persistent_workers=True, prefetch_factor=3)
    it = iter(dl); acc, na, t0 = 0.0, 0, time.time()
    for step in range(start, STEPS):
        b = next(it)
        H = torch.complex(b['Hr'], b['Hi']); tg = b['tgt']
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        m.train(); loss = ((m(H) - tg) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        acc += loss.item(); na += 1
        if (step + 1) % 50 == 0:
            rec = dict(step=step + 1, train=acc / na, lr=lr_at(step), s_per_step=(time.time() - t0) / na); acc, na, t0 = 0.0, 0, time.time()
            if (step + 1) % 500 == 0 or step + 1 == STEPS:
                m.eval()
                with torch.no_grad():
                    vp = torch.cat([m(vH[i:i + 256]) for i in range(0, len(vH), 256)])
                e = ((vp - vt) ** 2).numpy(); b0 = (vt ** 2).numpy()
                rec['val'] = float(e.mean())
                hi = (vsnr >= 5) & ((vpd >= 8) | (vgd >= 3))
                for nm, q in [('all', np.ones(len(e), bool)), ('snr>=5_imp', hi), ('snr<0', vsnr < 0), ('clean', (vpd == 0) & (vgd == 0))]:
                    for j, comp in enumerate(['ph', 'g']):
                        den = b0[q, :, j].sum()
                        rec[f'rem_{comp}_{nm}'] = float(100 * (1 - e[q, :, j].sum() / den)) if den > 0 else None
                    rec[f'rms_ph_deg_{nm}'] = float(np.rad2deg(np.sqrt(e[q, :, 0].mean()))); rec[f'rms_g_db_{nm}'] = float(20 / np.log(10) * np.sqrt(e[q, :, 1].mean()))
                if rec['val'] < best:
                    best = rec['val']; torch.save(dict(model=m.state_dict(), cfg=ARCH, step=step + 1, val=best), BEST)
                print(json.dumps(rec), flush=True)
            else:
                print(f"step {rec['step']} train {rec['train']:.5f} lr {rec['lr']:.2e} {rec['s_per_step']:.2f}s/step", flush=True)
            hist.append(rec)
        if (step + 1) % 250 == 0 or step + 1 == STEPS:
            torch.save(dict(model=m.state_dict(), opt=opt.state_dict(), step=step + 1, hist=hist, best=best), CK)
            json.dump(hist, open(HIST, 'w'), indent=0)
    torch.save(dict(model=m.state_dict(), cfg=ARCH, step=STEPS), f'{CA}/v2a_final.pt')
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
