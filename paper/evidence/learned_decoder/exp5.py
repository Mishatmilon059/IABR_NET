import os, sys, json, math, time
from exp3 import *      # RefinerV3, Staged, score_at, plus everything from exp

STEPS5 = int(os.environ.get('STEPS5', 1500)); BS5 = int(os.environ.get('BS5', 64))
GEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/generated_banks'
NC = (torch.arange(16, dtype=torch.float32) - 7.5)


def unwrap_t(d):                         # d: (B,16) complex -> unwrapped phase (B,16)
    dphi = torch.angle(d[:, 1:] * d[:, :-1].conj())
    return torch.cat([torch.zeros_like(dphi[:, :1]), torch.cumsum(dphi, 1)], 1)


def clean_t(d, lam):
    nc = NC.to(d.device)
    lg = torch.log(d.abs() + 1e-12); ph = unwrap_t(d)
    lg = lg - lg.mean(1, keepdim=True); ph = ph - ph.mean(1, keepdim=True)
    ph = ph - (ph @ nc)[:, None] / (nc @ nc) * nc
    return lam * lg, lam * ph


class StagedSC(Staged):
    """Learned refiner (shared MLP) + differentiable self-calibration stage with a learnable strength lam."""
    def __init__(self, refiner, **kw):
        super().__init__(refiner, **kw)
        self.lam = nn.Parameter(torch.tensor(1.0)); self.tau, self.sc_iters, self.sc_rounds, self.use_sc = 5.0, 2, 2, True

    def cyc(self, H, U, V, rounds):
        B, L = U.shape
        for _ in range(rounds):
            for l in range(L):
                x, A = ls_gains_t(H, U, V)
                own = A[:, :, l] * x[:, l:l + 1]; total = (A * x[:, None, :]).sum(-1)
                r = H - (total - own).reshape_as(H)
                for _ in range(self.inner):
                    d = self.ref(r, U[:, l], V[:, l])
                    U = torch.cat([U[:, :l], (U[:, l] + d[:, 0])[:, None], U[:, l + 1:]], 1)
                    V = torch.cat([V[:, :l], (V[:, l] + d[:, 1])[:, None], V[:, l + 1:]], 1)
        return U, V

    def calibrate(self, y, yc, U, V):
        B = y.shape[0]
        x, A = ls_gains_t(yc, U, V); Hm = (A * x[:, None, :]).sum(-1).reshape(B, 16, 16)
        snr_est = (Hm.abs() ** 2).sum((1, 2)) / ((yc - Hm).abs() ** 2).sum((1, 2)).clamp_min(1e-12)
        dt = torch.ones(B, 16, dtype=torch.complex64, device=y.device)
        for _ in range(3):
            M = Hm * dt[:, None, :]; dr = (y * M.conj()).sum(2) / (M.abs() ** 2).sum(2).clamp_min(1e-12)
            M = Hm * dr[:, :, None]; dt = (y * M.conj()).sum(1) / (M.abs() ** 2).sum(1).clamp_min(1e-12)
        lr, pr = clean_t(dr, self.lam); lt, pt = clean_t(dt, self.lam)
        gate = ((snr_est - self.tau) / 1.0).sigmoid() if self.training else (snr_est > self.tau).float()
        gate = gate[:, None]
        cr = gate * torch.exp(torch.complex(-lr, -pr)) + (1 - gate); ct = gate * torch.exp(torch.complex(-lt, -pt)) + (1 - gate)
        return y * cr[:, :, None] * ct[:, None, :]

    def forward_all(self, H, U0, V0):
        outs = super().forward_all(H, U0, V0); U, V = outs[-1]
        if self.use_sc:
            yc = H
            for _ in range(self.sc_iters):
                yc = self.calibrate(H, yc, U, V)
                U, V = self.cyc(yc, U, V, self.sc_rounds)
            outs.append((U, V))
        return outs


def load_any(name, n_per=250):
    if name in ('official', 'T2_d5'):
        T = load_test(name); return T
    b = np.load(f'{GEN}/{name}.npz'); n = np.load(f'{ROOT}/outputs/cache/net__{name}.npz')
    assert (b['L'] == 3).all(), name
    sel = np.arange(min(len(b['L']), 500))
    return dict(heat=n['heat'][sel], Hc=n['Hc'][sel], psi=b['psi'][sel][:, :3], phi=b['phi'][sel][:, :3], snr=b['snr'][sel])


if __name__ == '__main__':
    torch.manual_seed(5)
    model = StagedSC(RefinerV3(), pre=6, rounds=3, inner=2).to(DEV)
    npar = sum(p.numel() for p in model.parameters()); print('params', npar, 'steps', STEPS5, 'batch', BS5, flush=True)
    banks = ['official', 'T2_d5', 'gain_g4_snr15', 'oodphase_d15_snr15']
    tests = {n: load_any(n) for n in banks}

    def ev(use_sc):
        model.use_sc = use_sc; model.eval(); out = {}
        for n, T in tests.items():
            U, V = run_learned(model, T); out[n] = score(U, V, T)[1]
        return out

    rng = np.random.default_rng(55); opt = torch.optim.Adam(model.parameters(), 2e-3); t0 = time.time(); hist = []
    for s in range(STEPS5):
        model.train(); model.use_sc = True
        H, U0, V0, ug, vg = gen_batch(rng, BS5)
        for gp in opt.param_groups: gp['lr'] = 2e-3 * (0.5 * (1 + math.cos(math.pi * s / STEPS5)) * 0.95 + 0.05)
        outs = model.forward_all(H, U0, V0)
        loss = sum((1.0 if i == len(outs) - 1 else 0.4) * matched_loss(U, V, ug, vg) for i, (U, V) in enumerate(outs))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); hist.append(float(loss))
        if (s + 1) % max(1, STEPS5 // 10) == 0:
            print(f'  step {s+1}/{STEPS5} loss(avg last 50) {np.mean(hist[-50:]):.4f} lam {float(model.lam):.3f}  {time.time()-t0:.0f}s', flush=True)
    torch.save(model.state_dict(), 'lurd.pt')
    res = dict(learned_no_sc=ev(False), learned_with_sc=ev(True))
    for mode in ['newton+cyclic', 'newton+cyclic+selfcal']:
        res['classical ' + mode] = {n: score(*run_classical(T, mode), T)[1] for n, T in tests.items()}
    json.dump(dict(results=res, lam=float(model.lam), loss=hist[::20], params=npar), open('results5.json', 'w'))
    print('\n%-34s' % '', ' '.join('%20s' % b for b in banks))
    for k in ['classical newton+cyclic', 'classical newton+cyclic+selfcal', 'learned_no_sc', 'learned_with_sc']:
        print('%-34s' % k, ' '.join('%20.3f' % res[k][b] for b in banks))
    print('done', flush=True)
