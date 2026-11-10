"""Training of design B (HU-NOMP) and design A (CA-UNOMP), both unfolded from NOMP (Mamandipoor et al. 2016). Resumable, detached.
Stages (DESIGN.md section 2):
  S1  : front end only (CNN from adc_ft.pt + FiLM + scene head): focal heat loss + 50 x IABC MSE (exact targets only) + 0.5 x log-sigma MSE
        + 0.2 x L CE + 0.2 x impairment-type CE. Batch = 4 sub-batches of 32 (one mix component each).
  S2B : design B through the unrolled core (K = 2 hypotheses), front LR x 0.1; S2A: design A (K = 6, step MLP + selector) initialised
        from the S2B checkpoint; S3: A, all LR 1e-5, 50% ideal benchmark batches.
Loss (S2/S3): per hypothesis, Hungarian-matched smoothed hit loss 1 - sigmoid((1 deg - |err|)/tau) (tau 0.5 -> 0.15) + 0.05 Huber(err, 1 deg),
  on the final output (w 1.0) and the pre-polish output (w 0.3); detection CE of the fused map vs undetected true paths; stop BCE
  (candidate k accepted iff k <= L); selector listwise CE to softmax(hits / 0.25) (A); auxiliary front losses x 0.5.
Seeds: batch at step s of stage X = default_rng([70_200_000, STAGE_ID, s]); validation mix = 71_200_000 (+offsets), fixed, only for
  checkpoint selection (mean Pd over its groups). Adam, warm-up 100, cosine LR_MAX -> 1e-5, grad-norm clip 5, 6 threads.
usage: python train.py STAGE STEPS [bench]   (stdout -> train_STAGE.log)"""
import os
os.environ.setdefault('THREADS', '6')
from ucore import *

STAGE = sys.argv[1]; STEPS = int(sys.argv[2]); BENCH = len(sys.argv) > 3 and sys.argv[3] == 'bench'
SID = dict(S1=1, S2B=2, S2A=3, S3=4)[STAGE]
DESIGN = 'B' if STAGE in ('S1', 'S2B') else 'A'
BS = int(os.environ.get('BS', dict(S1=32, S2B=24, S2A=12, S3=12)[STAGE]))
LR_MAX = float(os.environ.get('LR', dict(S1=3e-4, S2B=3e-4, S2A=2e-4, S3=1e-5)[STAGE])); LR_MIN = 1e-5 if STAGE != 'S3' else 3e-6
WARM, CLIP, CK = 100, 5.0, 100
VAL_EVERY = int(os.environ.get('VAL_EVERY', 250))
CKPT = f'{BIG}/ckpt_{STAGE}.pt'
BEST = f'{HERE}/best_{STAGE}.pt'
PERMS = {L: torch.tensor(list(itertools.permutations(range(L)))) for L in range(1, 7)}


def lr_at(step):
    if step < WARM:
        return LR_MAX * (step + 1) / WARM
    t = (step - WARM) / max(1, STEPS - WARM)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * min(1.0, t)))


# ---------------------------------------------------------------- losses
def front_targets(d):
    Y = d['Y']; n = len(Y)
    rms = np.sqrt((np.abs(Y) ** 2).mean((1, 2)) / 2)
    Lv = np.full(n, d['L']) if np.isscalar(d['L']) else d['L']
    return dict(heat=torch.from_numpy(SIM.heat_targets(d['psi'], d['phi'], 32, 1.0)), imp=torch.from_numpy(d['imp']),
                imask=torch.from_numpy(d['imp_mask']), lsig=torch.from_numpy(np.log(sigma_of(d['snr']) / rms)).float(),
                typ=torch.from_numpy(d['imptype']).long(), L=torch.from_numpy(np.clip(Lv, 0, 8)).long())


def front_loss(fo, tg):
    lh = A.focal_loss(fo['heat'], tg['heat'])
    im = tg['imask'].float()
    li = (((fo['imp'] - tg['imp']) ** 2).mean((1, 2)) * im).sum() / im.sum().clamp_min(1)
    ls = TF.mse_loss(fo['logsig'], tg['lsig']); lL = TF.cross_entropy(fo['Lh'], tg['L']); lt = TF.cross_entropy(fo['typ'], tg['typ'])
    return lh + 50 * li + 0.5 * ls + 0.2 * lL + 0.2 * lt, dict(heat=lh.item(), imp=li.item(), lsig=ls.item(), L=lL.item(), typ=lt.item())


def matched_err(U, V, pt, ft):
    """U,V (R,L) estimates; pt, ft (R,L) true angles (deg) -> matched |err| psi, phi (R,L) (min total |err| over permutations)"""
    pe, fe = uv_to_deg(U, V)
    L = U.shape[1]; P = PERMS[L]
    ep = (pe[:, P] - pt[:, None, :]).abs(); ef = (fe[:, P] - ft[:, None, :]).abs()    # (R, nP, L)
    best = (ep + ef).sum(2).detach().argmin(1)
    ar = torch.arange(len(U))
    return ep[ar, best], ef[ar, best]


def hit_loss(e, tau):
    s = 1 - torch.sigmoid((1.0 - e) / tau)
    ec = torch.clamp(e, max=5.0)                       # Huber capped at 5 deg (missed paths give no pull)
    hub = torch.where(ec < 1, 0.5 * ec ** 2, ec - 0.5)
    return (s + 0.05 * hub).mean(1)


def core_loss(model, o, d, L, tau):
    B = len(d['Y']); dt = model.cfg['dtype']
    pt = torch.from_numpy(np.rad2deg(d['psi'][:, :L])).to(dt); ft = torch.from_numpy(np.rad2deg(d['phi'][:, :L])).to(dt)
    Ut = torch.from_numpy(np.pi * np.cos(d['psi'][:, :L])).to(dt); Vt = torch.from_numpy(np.pi * np.cos(d['phi'][:, :L])).to(dt)
    lh, hits = 0, []
    for h in o['hyps']:
        r = o['res'][h]
        ep, ef = matched_err(r['U'], r['V'], pt, ft); e = torch.cat([ep, ef], 1)
        ep0, ef0 = matched_err(r['Upre'], r['Vpre'], pt, ft); e0 = torch.cat([ep0, ef0], 1)
        lh = lh + (hit_loss(e, tau) + 0.3 * hit_loss(e0, tau)).mean()
        hits.append((e.detach() <= 1.0).sum(1).float())
    lh = lh / len(o['hyps'])
    hits = torch.stack(hits, 1)
    logs = dict(hit=lh.item(), hits_final=float(hits[torch.arange(B), o['pick']].mean() / (2 * L)))
    loss = lh
    if 'sel_logits' in o:
        tgt = torch.softmax(hits / 0.25, 1)
        ls = -(tgt * torch.log_softmax(o['sel_logits'], 1)).sum(1).mean()
        loss = loss + 0.5 * ls; logs['sel'] = ls.item()
    # stop head (H1 rows = first B rows of the greedy batch)
    if o.get('stop_logits'):
        lg = torch.stack([s[:B] for s in o['stop_logits']], 1)               # (B, n_det)
        tg = torch.zeros_like(lg); tg[:, :L] = 1.0
        lst = TF.binary_cross_entropy_with_logits(lg, tg)
        loss = loss + 0.3 * lst; logs['stop'] = lst.item()
    # detection map CE (H1 rows)
    ldet, nd = 0, 0
    for fused, Ud, Vd in o['dets']:
        fz = fused[:B]; Ud = Ud[:B]; Vd = Vd[:B]
        if Ud.shape[1]:
            du = wrap(Ud[:, :, None] - Ut[:, None, :]).abs(); dv = wrap(Vd[:, :, None] - Vt[:, None, :]).abs()
            det = ((du + dv) < 4 * math.pi / 64).any(1)
        else:
            det = torch.zeros(B, L, dtype=torch.bool)
        und = (~det).to(dt)
        tmap = blob64(Ut, Vt, und, 1.0)
        msk = tmap.reshape(B, -1).sum(1) > 1e-6
        if msk.any():
            tq = tmap.reshape(B, -1)[msk]; tq = tq / tq.sum(1, keepdim=True)
            lp = torch.log_softmax(model.det_scale * fz.reshape(B, -1)[msk], 1)
            ldet = ldet + (-(tq.float() * lp).sum(1)).mean(); nd += 1
    if nd:
        ldet = ldet / nd; loss = loss + 0.1 * ldet; logs['det'] = float(ldet)
    return loss, logs


# ---------------------------------------------------------------- batches
def s1_batch(step):
    parts = [make_mix(np.random.default_rng([70_200_000, SID, step, j]), BS, 70_200_000 + (SID * 10 ** 6 + step) * 4 + j) for j in range(4)]
    Y = np.concatenate([p['Y'] for p in parts]); bits = np.concatenate([p['bits'] for p in parts])
    tg = [front_targets(p) for p in parts]
    tgt = {k: torch.cat([t[k] for t in tg]) for k in tg[0]}
    return Y, bits, tgt


def s2_batch(step):
    rng = np.random.default_rng([70_200_000, SID, step])
    comp = None
    if STAGE == 'S3' and rng.uniform() < 0.5:
        comp = 'ideal'
    d = make_mix(rng, BS, 70_200_000 + SID * 10 ** 6 + step, comp=comp)
    return d


# ---------------------------------------------------------------- validation mix (fixed; checkpoint selection only)
def val_set():
    f = f'{BIG}/val_mix.npz'
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); return z['groups'].tolist()
    groups = []
    rng = np.random.default_rng(71_200_000)
    b = gen_scenes(rng, 160, 3, np.repeat(np.arange(-10, 26, 5), 20).astype(float), seed_int=71_200_000)
    groups.append(dict(name='ideal', Y=SS.generate(b, 'clean'), bits=np.zeros(160, int), L=3, psi=b['psi'], phi=b['phi'], snr=b['snr']))
    for j, (bb, snr) in enumerate([(1, 15), (2, 15), (3, 15), (4, 15), (1, 0), (2, 0), (3, 0), (4, 0)]):
        b = gen_scenes(np.random.default_rng([71_200_000, 1, j]), 60, 3, np.full(60, float(snr)), seed_int=71_200_010 + j)
        groups.append(dict(name=f'adc_b{bb}_{snr}', Y=SS.generate(b, 'M5', bb), bits=np.full(60, bb), L=3, psi=b['psi'], phi=b['phi'], snr=b['snr']))
    for j, (kind, sev, snr) in enumerate([('M2', 0.2, 15), ('M6', 30.0, 15), ('M6', 30.0, 0), ('M3', (45.0, 0.0), 0), ('M3', (30.0, 4.0), 15), ('M1', 0.3, 15)]):
        b = gen_scenes(np.random.default_rng([71_200_000, 2, j]), 60, 3, np.full(60, float(snr)), seed_int=71_200_020 + j)
        groups.append(dict(name=f'{kind}_{snr}', Y=SS.generate(b, kind, sev), bits=np.zeros(60, int), L=3, psi=b['psi'], phi=b['phi'], snr=b['snr']))
    for j, L in enumerate([1, 2, 4, 5, 6]):
        b = gen_scenes(np.random.default_rng([71_200_000, 3, j]), 40, L, np.repeat([0.0, 15.0], 20), sep=np.deg2rad(5.0), seed_int=71_200_030 + j)
        groups.append(dict(name=f'geom_L{L}', Y=SS.generate(b, 'clean'), bits=np.zeros(40, int), L=L, psi=b['psi'], phi=b['phi'], snr=b['snr']))
    np.savez(f, groups=np.array(groups, dtype=object))
    return groups


def validate(model, groups):
    model.eval(); res = {}
    cfg_old = dict(model.cfg); model.cfg.update(train_stop=False, dtype=torch.float64)
    for g in groups:
        U, V, _ = decode_np(model, g['Y'], g['bits'], g['L'], bs=120)
        res[g['name']] = pd(stats(C.uv_to_angles(U, V), dict(psi=g['psi'], phi=g['phi'], L=np.full(len(U), g['L']))))
    model.cfg.update(cfg_old); model.train()
    res['mean'] = float(np.mean([v for v in res.values()]))
    return res


# ---------------------------------------------------------------- main
if __name__ == '__main__':
    torch.manual_seed(SID)
    model = UNOMP(DESIGN)
    if STAGE == 'S2B' and os.path.exists(f'{HERE}/best_S1.pt'):
        sd = torch.load(f'{HERE}/best_S1.pt', map_location='cpu'); model.front.load_state_dict(sd)
    if STAGE == 'S2A' and os.path.exists(f'{HERE}/best_S2B.pt'):
        sd = torch.load(f'{HERE}/best_S2B.pt', map_location='cpu'); print('init from S2B:', model.load_state_dict(sd, strict=False), flush=True)
    if STAGE == 'S3':
        sd = torch.load(f'{HERE}/best_S2A.pt', map_location='cpu'); model.load_state_dict(sd)
    model.train()
    if STAGE == 'S1':
        params = [dict(params=model.front.parameters(), lr_mult=1.0)]
    else:
        model.cfg.update(dtype=torch.float32, train_stop=True)
        # attempt 2 (deviation, decided on validation, see deviations.txt): front end FROZEN in S2/S3 (its drift worsened sigma_hat),
        # hypernet output layer at 10x LR, gradient clipping per module (fusion gradients dominated the global norm)
        for p_ in model.front.parameters():
            p_.requires_grad_(False)
        if STAGE in ('S2A', 'S3'):     # deviation 10: A = frozen best B + trained step MLP (own gate), selector and stop head
            for mod in [model.hyper, model.fusion]:
                for p_ in mod.parameters():
                    p_.requires_grad_(False)
            model.det_scale.requires_grad_(False)
        hl = list(model.hyper.mlp[-1].parameters()); hid = {id(p) for p in hl} | {id(p) for p in model.front.parameters()}
        params = [dict(params=hl, lr_mult=10.0), dict(params=[p for p in model.parameters() if id(p) not in hid], lr_mult=1.0)]
        CLIP_MODS = [m_ for m_ in [model.hyper, model.fusion, model.stop, getattr(model, 'step', None), getattr(model, 'selector', None)] if m_ is not None]
        params = [dict(params=[p_ for p_ in g_['params'] if p_.requires_grad], lr_mult=g_['lr_mult']) for g_ in params]
        params = [g_ for g_ in params if g_['params']]
    opt = torch.optim.Adam(params, lr=LR_MAX)
    groups = val_set()
    start, hist, best = 0, [], -1.0
    if os.path.exists(CKPT) and not BENCH:
        s = torch.load(CKPT, map_location='cpu', weights_only=False)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); start, hist, best = s['step'], s['hist'], s['best']
        assert s['steps'] == STEPS
        print(f'resumed at step {start} (best val {best:.4f})', flush=True)
    if start == 0 and not BENCH:
        v = validate(model, groups); hist.append(dict(step=0, val=v)); best = v['mean']
        torch.save(model.state_dict() if STAGE != 'S1' else model.front.state_dict(), BEST)
        print(f'step 0 val mean {v["mean"]:.4f} ' + ' '.join(f'{k}:{x:.3f}' for k, x in v.items() if k != 'mean'), flush=True)
    acc = {}; n = 0; t_win = time.time(); t_all = time.time()
    rng_L = None
    for step in range(start, STEPS if not BENCH else start + 12):
        t0 = time.time()
        for g in opt.param_groups:
            g['lr'] = lr_at(step) * g['lr_mult']
        if STAGE == 'S1':
            Y, bits, tgt = s1_batch(step); bt = prep_batch(Y, bits)
            fo = model.front(bt['Y'].to(torch.complex64), bt['bits_oh'])
            loss, logs = front_loss(fo, tgt)
        else:
            d = s2_batch(step); L = d['L']; bt = prep_batch(d['Y'], d['bits'], d['snr'])
            t1 = time.time()
            o = model(bt, L)
            tau = 0.5 + (0.15 - 0.5) * min(1.0, step / max(1, 0.8 * STEPS))
            loss, logs = core_loss(model, o, d, L, tau)
            logs['L'] = L
        opt.zero_grad(set_to_none=True); loss.backward()
        if STAGE == 'S1':
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        else:
            gns = [torch.nn.utils.clip_grad_norm_([p_ for p_ in m_.parameters() if p_.grad is not None], CLIP) for m_ in CLIP_MODS
                   if any(p_.grad is not None for p_ in m_.parameters())]
            gn = torch.stack([g_ for g_ in gns]).max()
            if getattr(model, 'step_gate', None) is not None and model.step_gate.grad is not None:
                model.step_gate.grad.clamp_(-1, 1)
            if model.det_scale.grad is not None:
                model.det_scale.grad.clamp_(-1, 1)
        if torch.isfinite(gn):
            opt.step()
        else:
            logs['skipped_nonfinite'] = 1
        logs['loss'] = loss.item(); logs['gn'] = float(gn)
        for k_, v_ in logs.items():
            acc[k_] = acc.get(k_, 0.0) + float(v_)
        n += 1
        if BENCH:
            print(f'bench step {step}: {time.time() - t0:.2f}s ' + ' '.join(f'{k}:{v:.4f}' for k, v in logs.items()), flush=True)
            continue
        if (step + 1) % 25 == 0:
            a = {k: v / n for k, v in acc.items()}; acc = {}; n = 0; dtw = (time.time() - t_win) / 25; t_win = time.time()
            rec = dict(step=step + 1, lr=lr_at(step), s_per_step=dtw, **a); hist.append(rec)
            print(f'step {step + 1}/{STEPS} ' + ' '.join(f'{k}:{v:.4f}' for k, v in a.items()) + f' | {dtw:.2f} s/step ETA {dtw * (STEPS - step - 1) / 60:.0f} min', flush=True)
        if (step + 1) % VAL_EVERY == 0 or step + 1 == STEPS:
            v = validate(model, groups); hist.append(dict(step=step + 1, val=v))
            tag = ''
            if v['mean'] > best:
                best = v['mean']; torch.save(model.state_dict() if STAGE != 'S1' else model.front.state_dict(), BEST); tag = ' (best)'
            print(f'VAL step {step + 1} mean {v["mean"]:.4f}{tag} ' + ' '.join(f'{k}:{x:.3f}' for k, x in v.items() if k != 'mean'), flush=True)
        if (step + 1) % CK == 0 or step + 1 == STEPS:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1, hist=hist, best=best, steps=STEPS), CKPT)
    if not BENCH:
        save_json(dict(stage=STAGE, steps=STEPS, batch=BS, lr_max=LR_MAX, best_val=best, hist=hist), f'{HERE}/train_hist_{STAGE}.json')
        print(f'done {STAGE}: {STEPS} steps, best val mean {best:.4f} -> {BEST}', flush=True)
