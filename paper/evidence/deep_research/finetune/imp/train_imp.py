"""Step 1: fine-tune IABR-v2 (from outputs/checkpoints/IABRv2/final.pt, never overwritten) on 40% original training distribution +
60% extended impairments, one family per sample (icom.imp_batch: M3 phase U(0,45) deg with gain U(0,6) dB; M2 sigma U(0,0.2); M6 sigma
U(0,30) deg; M1 rho U(0,0.3); L 1..9, SNR -15..24 dB; unit-tested against stress_sim.generate in unit_gen.log).
Loss = the original loss: focal heatmap loss (notebook lines 686-691) + lam_imp (50) * MSE(IABC estimate, identifiable target), with the
IABC term MASKED (averaged over the samples that have a target): the identifiable diagonal target exists for the original distribution
and M3 (D_r, D_t are per-antenna diagonal factors); for M1 (C_r H C_t^T mixes antennas: not diagonal), M2 (position error = a
PATH-dependent phase exp(-j pi cos(psi_l) delta_n), diagonal only for L=1) and M6 (per-beam-measurement phase noise, not per antenna)
no per-antenna diagonal truth exists; a zero target would wrongly teach the IABC to switch off on distorted arrays, so there the IABC is
shaped only by the heatmap loss (it is trained jointly through the trunk, as in the original training).
Adam, linear warm-up WARM steps, cosine LR_MAX 3e-4 -> LR_MIN 1e-5, global grad-norm clip 5, batch BS 128, torch threads 6.
Seeds: batch at step s = default_rng([71000000, s]); fixed monitoring batch default_rng([71000001, 0]) (512 samples; loss only, NOT used
to select a checkpoint: the final weights are used). Resumable: cache/imp_ft_ckpt.pt (model, opt, step, hist) every CK steps; final
weights -> cache/imp_ft.pt; history -> train_hist.json.   usage: python train_imp.py bench | python train_imp.py STEPS"""
from icom import *
import math

BS, LR_MAX, LR_MIN, WARM, CLIP, LAM_IMP = int(os.environ.get('BS', 128)), 3e-4, 1e-5, int(os.environ.get('WARM', 200)), 5.0, 50.0
CK, LOG = 200, 50
CKPT = f'{CACHE}/imp_ft_ckpt.pt'
FOCAL = A.focal_loss


def lr_at(step, total):
    if step < WARM:
        return LR_MAX * (step + 1) / WARM
    t = (step - WARM) / max(1, total - WARM)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * min(1.0, t)))


def masked_mse(p, t, m):
    e = ((p - t) ** 2).mean((1, 2))
    return (e * m).sum() / m.sum().clamp(min=1.0)


def losses(model, y, heat, imp, mask):
    o = model(y)
    lh = FOCAL(o['heat'], heat); li = masked_mse(o['imp'], imp, mask)
    return lh + LAM_IMP * li, lh, li


if __name__ == '__main__':
    mode = sys.argv[1]
    model = load_cnn(ORIG_CKPT); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR_MAX)
    if mode == 'bench':
        t = []
        for s in range(12):
            t0 = time.time(); y, h, im, mk, _ = imp_batch(np.random.default_rng([71000002, 10 ** 6 + s]), BS); t1 = time.time()
            loss, lh, li = losses(model, y, h, im, mk); opt.zero_grad(); loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)); opt.step(); t2 = time.time()
            t.append((t1 - t0, t2 - t1))
            print(f'bench step {s}: data {t1 - t0:.3f}s fwd/bwd/step {t2 - t1:.3f}s loss {loss.item():.4f} (heat {lh.item():.4f} imp {li.item():.2e}) gradnorm {gn:.2f}', flush=True)
        t = np.array(t[2:]); print(f'mean data {t[:, 0].mean():.3f}s, compute {t[:, 1].mean():.3f}s, total {t.sum(1).mean():.3f}s/step (batch {BS})')
        sys.exit()
    STEPS = int(mode)
    start, hist, sec = 0, [], 0.0
    if os.path.exists(CKPT):
        s = torch.load(CKPT, map_location='cpu', weights_only=False)
        assert s['steps'] == STEPS
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); start, hist, sec = s['step'], s['hist'], s['sec']
        print(f'resumed at step {start}', flush=True)
    my, mh, mi, mm, mf = imp_batch(np.random.default_rng([71000001, 0]), 512)

    def monitor():
        model.eval(); out = {}
        with torch.no_grad():
            o = model(my)
            for nm, code in FAM_CODE.items():
                sl = torch.from_numpy(mf == code)
                out[nm] = (FOCAL(o['heat'][sl], mh[sl]).item(), masked_mse(o['imp'][sl], mi[sl], mm[sl]).item())
        model.train()
        return out

    def fmt(m):
        return ' '.join(f'{k} {v[0]:.4f}' + (f'/{v[1]:.1e}' if k in ('orig', 'M3') else '') for k, v in m.items())
    if start == 0:
        m0 = monitor(); print(f'step 0 monitor heat[/imp]: {fmt(m0)}', flush=True); hist.append(dict(step=0, mon=m0))
    acc = np.zeros(4); n = 0; t_run = time.time(); t_win = time.time()
    for step in range(start, STEPS):
        y, h, im, mk, _ = imp_batch(np.random.default_rng([71000000, step]), BS)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, STEPS)
        loss, lh, li = losses(model, y, h, im, mk)
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)); opt.step()
        acc += [loss.item(), lh.item(), li.item(), gn]; n += 1
        if (step + 1) % LOG == 0 or step + 1 == STEPS:
            a = acc / n; acc[:] = 0; n = 0; dt = (time.time() - t_win) / LOG; t_win = time.time()
            m = monitor()
            hist.append(dict(step=step + 1, loss=a[0], heat=a[1], imp=a[2], gradnorm=a[3], lr=lr_at(step, STEPS), s_per_step=dt, mon=m))
            print(f'step {step + 1}/{STEPS} loss {a[0]:.4f} (heat {a[1]:.4f} imp {a[2]:.2e}) |g| {a[3]:.2f} lr {lr_at(step, STEPS):.2e} | '
                  f'monitor {fmt(m)} | {dt:.2f} s/step, ETA {dt * (STEPS - step - 1) / 60:.0f} min', flush=True)
        if (step + 1) % CK == 0 or step + 1 == STEPS:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, hist=hist, sec=sec + time.time() - t_run, steps=STEPS), CKPT + '.tmp')
            os.replace(CKPT + '.tmp', CKPT)
    sec += time.time() - t_run
    model.eval(); torch.save(model.state_dict(), FT_CKPT)
    save_json(dict(steps=STEPS, batch=BS, lr_max=LR_MAX, lr_min=LR_MIN, warmup=WARM, clip=CLIP, lam_imp=LAM_IMP, train_min_last_session=sec / 60,
                   mix='40% original make_batch + 60% extended (M3/M2/M6/M1 uniform)',
                   seeds='default_rng([71000000, step]); monitor default_rng([71000001, 0])', hist=hist), f'{HERE}/train_hist.json')
    print(f'done: {STEPS} steps, {sec / 60:.1f} min total -> {FT_CKPT}', flush=True)
