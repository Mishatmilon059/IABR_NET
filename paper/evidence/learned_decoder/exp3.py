import os, sys, json, math, time
from exp import *          # cnn, quantities, ls_gains_t, matched_loss, gen_batch, load_test, score, run_classical, run_learned, grids, ...

STEPS3 = int(os.environ.get('STEPS3', 1500)); BS3 = int(os.environ.get('BS3', 64))


def score_at(r, u, v):
    a, b = grids(r.device)
    E = torch.exp(1j * a[None] * u[:, None, None] - 1j * b[None] * v[:, None, None])
    return (r * E).sum((1, 2)).abs() ** 2


class RefinerV3(nn.Module):
    """Learned optimiser: sees only the score f, its gradient and a few probes of the score surface.
    No Hessian and no Newton formula is given; the output step starts at zero and must be learned."""
    def __init__(self, hidden=64, probes=(0.02, 0.08), max_step=math.pi / 32):
        super().__init__(); self.probes, self.max_step = probes, max_step
        nf = 2 + 2 * len(probes) + 1
        self.net = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, u, v):
        S, gu, gv, *_ = quantities(r, u, v); f0 = S.abs() ** 2 + 1e-9
        feats = [gu / f0, gv / f0]
        for d in self.probes:
            feats.append((score_at(r, u + d, v) + score_at(r, u - d, v) - 2 * f0) / f0)
            feats.append((score_at(r, u, v + d) + score_at(r, u, v - d) - 2 * f0) / f0)
        feats.append(S.abs() / torch.sqrt(256 * (r.abs() ** 2).sum((1, 2)) + 1e-9))
        return self.max_step * torch.tanh(self.net(torch.stack(feats, 1)))


class Staged(UnrolledNOMP):
    def forward_all(self, H, U, V):
        B, L = U.shape; outs = []
        for _ in range(self.pre):
            r = H[:, None].expand(B, L, 16, 16).reshape(B * L, 16, 16)
            d = self.ref(r, U.reshape(-1), V.reshape(-1)); U = U + d[:, 0].reshape(B, L); V = V + d[:, 1].reshape(B, L)
        outs.append((U, V))
        for _ in range(self.rounds):
            for l in range(L):
                x, A = ls_gains_t(H, U, V)
                own = A[:, :, l] * x[:, l:l + 1]; total = (A * x[:, None, :]).sum(-1)
                r = H - (total - own).reshape_as(H)
                for _ in range(self.inner):
                    d = self.ref(r, U[:, l], V[:, l])
                    U = torch.cat([U[:, :l], (U[:, l] + d[:, 0])[:, None], U[:, l + 1:]], 1)
                    V = torch.cat([V[:, :l], (V[:, l] + d[:, 1])[:, None], V[:, l + 1:]], 1)
            outs.append((U, V))
        return outs

    def forward(self, H, U, V): return self.forward_all(H, U, V)[-1]


if __name__ == '__main__':
    torch.manual_seed(3)
    m3 = Staged(RefinerV3(), pre=6, rounds=3, inner=2).to(DEV)
    print('params V3', sum(p.numel() for p in m3.parameters()), 'steps', STEPS3, 'batch', BS3, flush=True)
    tests = {n: load_test(n) for n in ['official', 'T2_d5']}

    def evaluate(tag):
        out = {}
        for n, T in tests.items():
            U, V = run_learned(m3, T); per, mean = score(U, V, T); out[n] = dict(mean=mean, per_snr=per)
        print(f'{tag:34s} official {out["official"]["mean"]:.3f} | T2_d5 {out["T2_d5"]["mean"]:.3f}', flush=True); return out

    res = {'V3 from scratch, untrained': evaluate('V3 from scratch, untrained')}
    rng = np.random.default_rng(33); opt = torch.optim.Adam(m3.parameters(), 2e-3); t0 = time.time(); hist = []
    for s in range(STEPS3):
        H, U0, V0, ug, vg = gen_batch(rng, BS3)
        for gp in opt.param_groups: gp['lr'] = 2e-3 * (0.5 * (1 + math.cos(math.pi * s / STEPS3)) * 0.95 + 0.05)
        outs = m3.forward_all(H, U0, V0)
        loss = sum((1.0 if i == len(outs) - 1 else 0.4) * matched_loss(U, V, ug, vg) for i, (U, V) in enumerate(outs))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m3.parameters(), 1.0); opt.step(); hist.append(float(loss))
        if (s + 1) % max(1, STEPS3 // 10) == 0:
            print(f'  [V3] step {s+1}/{STEPS3} loss(avg last 50) {np.mean(hist[-50:]):.4f}  {time.time()-t0:.0f}s', flush=True)
    res['V3 from scratch, trained'] = evaluate('V3 from scratch, trained')
    json.dump(dict(results=res, loss=hist[::20]), open('results3.json', 'w')); print('done', flush=True)
