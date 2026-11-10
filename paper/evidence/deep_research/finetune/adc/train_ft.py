"""Step 2: fine-tune IABR-v2 (from outputs/checkpoints/IABRv2/final.pt, never overwritten) on a 50/50 mixture:
  half of every batch = the original training distribution (iabr2_sim.make_batch with the notebook CFG: L 1..9, SNR -15..24 dB,
  20% clean, phase <= 8 deg, gain <= 3 dB),
  other half = the same distribution with Y passed through stress_sim.adc(Y, b, 'peak') before to_ri, b ~ U{1,2,3,4} per sample.
Loss = the original loss: focal heatmap loss (notebook lines 686-691) + lam_imp (50) * MSE(IABC estimate, impairment_targets).
Adam, linear warm-up WARM steps, cosine LR_MAX -> LR_MIN, grad-norm clip 5, batch BS. torch threads 6.
Training data seeds: batch at step s is drawn from default_rng([31337, s]) (disjoint from every test/tuning seed: those are plain
integer seeds 5000/5015, 8300/8315, 9000/9015, 9900/9915, 9600/9615, and the frozen official bank). Deterministic resume: the
checkpoint cache/adc_ft_ckpt.pt (model, opt, step, hist) is written every CK steps; final weights -> cache/adc_ft.pt.
A fixed monitoring batch (default_rng([31338, 0]), 256 samples, half ADC) is evaluated every LOG steps (loss only; it is NOT used to
select a checkpoint: the final weights are used).
usage: python train_ft.py bench  |  python train_ft.py STEPS      (stdout -> train.log)"""
from acommon import *
import math
import iabr2_sim as SIM

torch.set_num_threads(int(os.environ.get("THREADS", 6)))
CFG = dict(train_L=(1, 9), train_snr=(-15, 24), clean_frac=0.2, train_phase_max=8.0, train_gain_max=3.0, grid=32, heat_sigma=1.0)
BS, LR_MAX, LR_MIN, WARM, CLIP, LAM_IMP = int(os.environ.get("BS", 128)), 3e-4, 1e-5, int(os.environ.get("WARM", 100)), 5.0, 50.0
CK, LOG = 250, 50
CKPT = f'{CACHE}/adc_ft_ckpt.pt'


def batch(rng, B):
    d = SIM.make_batch(rng, B, CFG)
    Y = d['y'][:, 0].astype(np.float64) + 1j * d['y'][:, 1].astype(np.float64)
    adc_mask = np.zeros(B, bool); adc_mask[rng.permutation(B)[:B // 2]] = True
    bits = rng.integers(1, 5, B)
    for bb in [1, 2, 3, 4]:
        s = adc_mask & (bits == bb)
        if s.any():
            Y[s] = SS.adc(Y[s], int(bb), 'peak')
    y = np.ascontiguousarray(SIM.to_ri(Y).transpose(0, 3, 1, 2))
    return torch.from_numpy(y), torch.from_numpy(d['heat']), torch.from_numpy(d['imp']), adc_mask, np.where(adc_mask, bits, 0)


def lr_at(step, total):
    if step < WARM:
        return LR_MAX * (step + 1) / WARM
    t = (step - WARM) / max(1, total - WARM)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * min(1.0, t)))


def losses(model, y, heat, imp):
    o = model(y)
    lh = focal_loss(o['heat'], heat); li = TF.mse_loss(o['imp'], imp)
    return lh + LAM_IMP * li, lh, li


if __name__ == '__main__':
    mode = sys.argv[1]
    model = load_cnn(ORIG_CKPT); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    if mode == 'bench':
        t = []
        for s in range(12):
            t0 = time.time(); y, h, im, _, _ = batch(np.random.default_rng([31337, 10 ** 6 + s]), BS); t1 = time.time()
            loss, lh, li = losses(model, y, h, im); opt.zero_grad(); loss.backward(); opt.step(); t2 = time.time()
            t.append((t1 - t0, t2 - t1))
            print(f'bench step {s}: data {t1 - t0:.3f}s  fwd/bwd/step {t2 - t1:.3f}s  loss {loss.item():.4f}', flush=True)
        t = np.array(t[2:]); print(f'mean data {t[:, 0].mean():.3f}s, compute {t[:, 1].mean():.3f}s, total {t.sum(1).mean():.3f}s/step (batch {BS})')
        sys.exit()
    STEPS = int(mode)
    start, hist, sec = 0, [], 0.0
    if os.path.exists(CKPT):
        s = torch.load(CKPT, map_location='cpu', weights_only=False)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); start, hist, sec = s['step'], s['hist'], s['sec']
        assert s['steps'] == STEPS
        print(f'resumed at step {start}', flush=True)
    mrng = np.random.default_rng([31338, 0]); my, mh, mi, mmask, mbits = batch(mrng, 256)

    def monitor():
        model.eval()
        with torch.no_grad():
            o = model(my)
            out = {}
            for nm, sel in [('orig', ~mmask), ('adc', mmask)]:
                sl = torch.from_numpy(sel)
                out[nm] = (focal_loss(o['heat'][sl], mh[sl]).item(), TF.mse_loss(o['imp'][sl], mi[sl]).item())
        model.train()
        return out
    if start == 0:
        m0 = monitor(); print(f'step 0 monitor: orig heat {m0["orig"][0]:.4f} imp {m0["orig"][1]:.2e} | adc heat {m0["adc"][0]:.4f} imp {m0["adc"][1]:.2e}', flush=True)
        hist.append(dict(step=0, mon=m0))
    acc = np.zeros(3); n = 0; t_run = time.time(); t_win = time.time()
    for step in range(start, STEPS):
        y, h, im, _, _ = batch(np.random.default_rng([31337, step]), BS)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, STEPS)
        loss, lh, li = losses(model, y, h, im)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
        acc += [loss.item(), lh.item(), li.item()]; n += 1
        if (step + 1) % LOG == 0 or step + 1 == STEPS:
            a = acc / n; acc[:] = 0; n = 0; dt = (time.time() - t_win) / LOG; t_win = time.time()
            m = monitor()
            hist.append(dict(step=step + 1, loss=a[0], heat=a[1], imp=a[2], lr=lr_at(step, STEPS), s_per_step=dt, mon=m))
            print(f'step {step + 1}/{STEPS} loss {a[0]:.4f} (heat {a[1]:.4f} imp {a[2]:.2e}) lr {lr_at(step, STEPS):.2e} | monitor orig heat '
                  f'{m["orig"][0]:.4f} imp {m["orig"][1]:.2e}, adc heat {m["adc"][0]:.4f} imp {m["adc"][1]:.2e} | {dt:.2f} s/step, '
                  f'ETA {dt * (STEPS - step - 1) / 60:.0f} min', flush=True)
        if (step + 1) % CK == 0 or step + 1 == STEPS:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, hist=hist, sec=sec + time.time() - t_run, steps=STEPS), CKPT)
    sec += time.time() - t_run
    model.eval(); torch.save(model.state_dict(), FT_CKPT)
    save_json(dict(steps=STEPS, batch=BS, lr_max=LR_MAX, lr_min=LR_MIN, warmup=WARM, clip=CLIP, lam_imp=LAM_IMP, train_min=sec / 60,
                   seeds='default_rng([31337, step]); monitor default_rng([31338, 0])', hist=hist), f'{HERE}/train_hist.json')
    print(f'done: {STEPS} steps, {sec / 60:.1f} min this session -> {FT_CKPT}', flush=True)
