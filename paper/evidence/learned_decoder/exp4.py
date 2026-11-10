import os, sys, json, math, time
from exp3 import *      # Staged, score_at, ... (and everything from exp)

STEPS4 = int(os.environ.get('STEPS4', 1500)); BS4 = int(os.environ.get('BS4', 64))


def wrap(a): return torch.atan2(torch.sin(a), torch.cos(a))


class RefinerV4(nn.Module):
    """Bigger learned optimiser: 3 hidden layers of 128, more probes (3 axis scales + diagonals), and the (wrapped) position
    (u, v) itself so it can learn prior structure such as the end-fire boundary."""
    def __init__(self, hidden=128, axis=(0.01, 0.04, 0.12), diag=0.06, max_step=math.pi / 32):
        super().__init__(); self.axis, self.diag, self.max_step = axis, diag, max_step
        nf = 2 + 2 * len(axis) + 2 + 1 + 2
        self.net = nn.Sequential(nn.Linear(nf, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, u, v):
        S, gu, gv, *_ = quantities(r, u, v); f0 = S.abs() ** 2 + 1e-9
        feats = [gu / f0, gv / f0]
        for d in self.axis:
            feats.append((score_at(r, u + d, v) + score_at(r, u - d, v) - 2 * f0) / f0)
            feats.append((score_at(r, u, v + d) + score_at(r, u, v - d) - 2 * f0) / f0)
        d = self.diag
        feats.append((score_at(r, u + d, v + d) + score_at(r, u - d, v - d) - 2 * f0) / f0)
        feats.append((score_at(r, u + d, v - d) + score_at(r, u - d, v + d) - 2 * f0) / f0)
        feats.append(S.abs() / torch.sqrt(256 * (r.abs() ** 2).sum((1, 2)) + 1e-9))
        feats.append(wrap(u) / math.pi); feats.append(wrap(v) / math.pi)
        return self.max_step * torch.tanh(self.net(torch.stack(feats, 1)))


if __name__ == '__main__':
    torch.manual_seed(4)
    m4 = Staged(RefinerV4(), pre=6, rounds=3, inner=2).to(DEV)
    print('params V4', sum(p.numel() for p in m4.parameters()), 'steps', STEPS4, 'batch', BS4, flush=True)
    tests = {n: load_test(n) for n in ['official', 'T2_d5']}

    def evaluate(tag):
        out = {}
        for n, T in tests.items():
            U, V = run_learned(m4, T); per, mean = score(U, V, T); out[n] = dict(mean=mean, per_snr=per)
        print(f'{tag:34s} official {out["official"]["mean"]:.3f} | T2_d5 {out["T2_d5"]["mean"]:.3f}', flush=True); return out

    res = {}
    rng = np.random.default_rng(44); opt = torch.optim.Adam(m4.parameters(), 2e-3); t0 = time.time(); hist = []
    for s in range(STEPS4):
        H, U0, V0, ug, vg = gen_batch(rng, BS4)
        for gp in opt.param_groups: gp['lr'] = 2e-3 * (0.5 * (1 + math.cos(math.pi * s / STEPS4)) * 0.95 + 0.05)
        outs = m4.forward_all(H, U0, V0)
        loss = sum((1.0 if i == len(outs) - 1 else 0.4) * matched_loss(U, V, ug, vg) for i, (U, V) in enumerate(outs))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m4.parameters(), 1.0); opt.step(); hist.append(float(loss))
        if (s + 1) % max(1, STEPS4 // 10) == 0:
            print(f'  [V4] step {s+1}/{STEPS4} loss(avg last 50) {np.mean(hist[-50:]):.4f}  {time.time()-t0:.0f}s', flush=True)
    res['V4 bigger, trained'] = evaluate('V4 bigger, trained')
    json.dump(dict(results=res, loss=hist[::20]), open('results4.json', 'w')); print('done', flush=True)
