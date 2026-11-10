import os, sys, time, json, itertools, math
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import numpy as np, scipy.ndimage, torch, torch.nn as nn, torch.nn.functional as TF
import iabr2_sim as sim

STEPS = int(os.environ.get('STEPS', 1500)); BS = int(os.environ.get('BS', 128)); NTEST = int(os.environ.get('NTEST', 250))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device', DEV, 'steps', STEPS, 'batch', BS, flush=True)
torch.set_num_threads(6)

ROOT = 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'
src = open(f'{ROOT}/notebook_src.py', encoding='utf-8').read().split('\n')
g = dict(np=np, scipy=scipy, CFG=dict(nomp_os=4, nomp_rc=3)); exec('\n'.join(src[527:647]), g)
newton_np, cyclic_np, heat_peaks_np, selfcal_np, uv_to_angles_np = [g[k] for k in ['newton', 'cyclic', 'heat_peaks', 'selfcal', 'uv_to_angles']]
gm = dict(np=np, torch=torch, nn=nn, TF=TF, W_IDEAL=sim.W_IDEAL, F_IDEAL=sim.F_IDEAL, P=16, Q=16); exec('\n'.join(src[419:522]), gm)
cnn = gm['IABRv2'](32, 10, 32).to(DEV); cnn.load_state_dict(torch.load(f'{ROOT}/outputs/checkpoints/IABRv2/final.pt', map_location=DEV)); cnn.eval()

NN_T = None


def grids(dev):
    n = torch.arange(16, device=dev, dtype=torch.float32)
    return torch.meshgrid(n, n, indexing='ij')


def atoms_t(U, V):
    a, b = grids(U.device)
    ph = -1j * a[None, None] * U[:, :, None, None] + 1j * b[None, None] * V[:, :, None, None]
    return torch.exp(ph).reshape(U.shape[0], U.shape[1], 256).transpose(1, 2)


def ls_gains_t(H, U, V):
    A = atoms_t(U, V); AH = A.conj().transpose(1, 2); L = U.shape[1]
    reg = 1e-6 * torch.eye(L, device=H.device, dtype=torch.complex64)
    x = torch.linalg.solve(AH @ A + reg, AH @ H.reshape(len(H), -1, 1)).squeeze(-1)
    return x, A


def quantities(r, u, v):
    a, b = grids(r.device)
    E = torch.exp(1j * a[None] * u[:, None, None] - 1j * b[None] * v[:, None, None]); rE = r * E
    S = rE.sum((1, 2)); Su = (1j * a[None] * rE).sum((1, 2)); Sv = (-1j * b[None] * rE).sum((1, 2))
    Suu = (-(a[None] ** 2) * rE).sum((1, 2)); Svv = (-(b[None] ** 2) * rE).sum((1, 2)); Suv = ((a[None] * b[None]) * rE).sum((1, 2))
    gu = 2 * (S.conj() * Su).real; gv = 2 * (S.conj() * Sv).real
    huu = 2 * (Su.conj() * Su + S.conj() * Suu).real; hvv = 2 * (Sv.conj() * Sv + S.conj() * Svv).real; huv = 2 * (Su.conj() * Sv + S.conj() * Suv).real
    return S, gu, gv, huu, hvv, huv


def newton_step(gu, gv, huu, hvv, huv, clip):
    det = huu * hvv - huv ** 2; ok = (huu < 0) & (det > 0); sd = torch.where(ok, det, torch.ones_like(det))
    du = torch.where(ok, -(hvv * gu - huv * gv) / sd, torch.zeros_like(gu)); dv = torch.where(ok, -(-huv * gu + huu * gv) / sd, torch.zeros_like(gu))
    return du.clamp(-clip, clip), dv.clamp(-clip, clip), ok.float()


class RefinerV1(nn.Module):        # the pasted design: 8 raw features, tanh(.)*0.2 step
    def __init__(self, max_step=0.2):
        super().__init__(); self.max_step = max_step
        self.net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, r, u, v):
        S, gu, gv, huu, hvv, huv = quantities(r, u, v)
        f = torch.stack([S.real, S.imag, gu, gv, huu, hvv, huv, S.abs()], 1)
        return self.max_step * torch.tanh(self.net(f))


class RefinerV2(nn.Module):        # same MLP size, normalised inputs, output = classical Newton step + learned correction
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, r, u, v):
        S, gu, gv, huu, hvv, huv = quantities(r, u, v)
        nu, nv, ok = newton_step(gu, gv, huu, hvv, huv, math.pi / 32)
        f = S.abs() ** 2 + 1e-9; corr = S.abs() / torch.sqrt(256 * (r.abs() ** 2).sum((1, 2)) + 1e-9)
        feats = torch.stack([nu / 0.1, nv / 0.1, ok, gu / f, gv / f, huu / (50 * f), hvv / (50 * f), huv / (50 * f), corr], 1)
        return torch.stack([nu, nv], 1) + 0.05 * torch.tanh(self.net(feats))


class UnrolledNOMP(nn.Module):
    def __init__(self, refiner, L=3, pre=0, rounds=3, inner=1):
        super().__init__(); self.ref = refiner; self.L, self.pre, self.rounds, self.inner = L, pre, rounds, inner

    def forward(self, H, U, V):
        B, L = U.shape
        for _ in range(self.pre):
            r = H[:, None].expand(B, L, 16, 16).reshape(B * L, 16, 16)
            d = self.ref(r, U.reshape(-1), V.reshape(-1)); U = U + d[:, 0].reshape(B, L); V = V + d[:, 1].reshape(B, L)
        for _ in range(self.rounds):
            for l in range(L):
                x, A = ls_gains_t(H, U, V)
                own = A[:, :, l] * x[:, l:l + 1]; total = (A * x[:, None, :]).sum(-1)
                r = H - (total - own).reshape_as(H)
                for _ in range(self.inner):
                    d = self.ref(r, U[:, l], V[:, l])
                    U = torch.cat([U[:, :l], (U[:, l] + d[:, 0])[:, None], U[:, l + 1:]], 1)
                    V = torch.cat([V[:, :l], (V[:, l] + d[:, 1])[:, None], V[:, l + 1:]], 1)
        return U, V


PERMS = list(itertools.permutations(range(3)))


def wrapd(a): return torch.atan2(torch.sin(a), torch.cos(a))


def matched_loss(U, V, ug, vg):
    C = torch.stack([torch.sqrt(wrapd(U[:, list(p)] - ug) ** 2 + wrapd(V[:, list(p)] - vg) ** 2 + 1e-6).sum(1) for p in PERMS], 1)
    return C.min(1).values.mean()


def gen_batch(rng, B):
    snr = rng.integers(-15, 25, B).astype(float); clean = rng.uniform(size=B) < 0.2
    pd = np.where(clean, 0.0, rng.uniform(0, 8, B)); gd = np.where(clean, 0.0, rng.uniform(0, 3, B))
    psi, phi, alpha = sim.sample_paths(np.full(B, 3), rng, lmax=3)
    out = sim.simulate(psi, phi, alpha, snr, rng, phase_deg_max=pd, gain_db_max=gd)
    y = torch.from_numpy(sim.to_ri(out['Y']).transpose(0, 3, 1, 2).copy()).to(DEV)
    with torch.no_grad(): o = cnn(y)
    heat = torch.sigmoid(o['heat']).cpu().numpy(); U0, V0 = heat_peaks_np(heat, 3)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)
    return o['Hc'].detach(), t(U0), t(V0), t(np.pi * np.cos(psi)), t(np.pi * np.cos(phi))


def train(model, name, seed):
    rng = np.random.default_rng(seed); opt = torch.optim.Adam(model.parameters(), 1e-3); t0 = time.time(); hist = []
    for s in range(STEPS):
        H, U0, V0, ug, vg = gen_batch(rng, BS)
        for gp in opt.param_groups: gp['lr'] = 1e-3 * (0.5 * (1 + math.cos(math.pi * s / STEPS)) * 0.95 + 0.05)
        U, V = model(H, U0, V0); loss = matched_loss(U, V, ug, vg)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); hist.append(float(loss))
        if (s + 1) % max(1, STEPS // 10) == 0:
            print(f'  [{name}] step {s+1}/{STEPS} loss(avg last 50) {np.mean(hist[-50:]):.5f}  {time.time()-t0:.0f}s', flush=True)
    return hist


def load_test(name):
    if name == 'official':
        b = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz'); psi, phi, snr = b['feat'][:, 0], b['feat'][:, 1], b['meta'][:, 1]
    else:
        b = np.load(f'{ROOT}/data/banks/{name}.npz'); psi, phi, snr = b['psi'], b['phi'], b['snr']
    n = np.load(f'{ROOT}/outputs/cache/net__{name}.npz')
    idx = np.concatenate([np.where(snr == s)[0][:NTEST] for s in sorted(set(snr.tolist()))])
    return dict(heat=n['heat'][idx], Hc=n['Hc'][idx], psi=psi[idx][:, :3], phi=phi[idx][:, :3], snr=snr[idx])


def score(U, V, T):
    ps, ph = uv_to_angles_np(U, V); ps, ph = np.degrees(ps), np.degrees(ph); tp, tf = np.degrees(T['psi']), np.degrees(T['phi'])
    comp = []
    for k in range(len(U)):
        best = min(PERMS, key=lambda p: sum(abs(ps[k, q] - tp[k, j]) + abs(ph[k, q] - tf[k, j]) for j, q in enumerate(p)))
        comp.append([abs(ps[k, q] - tp[k, j]) <= 1 for j, q in enumerate(best)] + [abs(ph[k, q] - tf[k, j]) <= 1 for j, q in enumerate(best)])
    comp = np.array(comp, float).mean(1)      # per-sample fraction of the 6 components within 1 deg
    return {int(s): float(comp[T['snr'] == s].mean()) for s in sorted(set(T['snr'].tolist()))}, float(comp.mean())


def run_classical(T, mode):
    heat, Hc = T['heat'], T['Hc']; U0, V0 = heat_peaks_np(heat, 3)
    if mode == 'peaks': return U0, V0
    Un, Vn = newton_np(Hc[:, None].repeat(3, 1).reshape(-1, 16, 16), U0.ravel(), V0.ravel()); Un, Vn = Un.reshape(-1, 3), Vn.reshape(-1, 3)
    U, V = cyclic_np(Hc, Un, Vn, 3)
    if mode == 'newton+cyclic': return U, V
    return selfcal_np(Hc, U, V, 1.0, 5.0)


def run_learned(model, T):
    U0, V0 = heat_peaks_np(T['heat'], 3); Us, Vs = [], []
    with torch.no_grad():
        for i in range(0, len(U0), 200):
            H = torch.from_numpy(T['Hc'][i:i + 200]).to(DEV); u = torch.tensor(U0[i:i + 200], dtype=torch.float32, device=DEV); v = torch.tensor(V0[i:i + 200], dtype=torch.float32, device=DEV)
            U, V = model(H, u, v); Us.append(U.cpu().numpy()); Vs.append(V.cpu().numpy())
    return np.concatenate(Us).astype(np.float64), np.concatenate(Vs).astype(np.float64)


if __name__ == '__main__':
    torch.manual_seed(0)
    m1 = UnrolledNOMP(RefinerV1(), pre=0, rounds=3, inner=1).to(DEV)
    m2 = UnrolledNOMP(RefinerV2(), pre=4, rounds=3, inner=2).to(DEV)
    npar = lambda m: sum(p.numel() for p in m.parameters())
    print('params: V1', npar(m1), ' V2', npar(m2), flush=True)
    tests = {n: load_test(n) for n in ['official', 'T2_d5']}
    results = {}
    def evaluate(tag, fn):
        for n, T in tests.items():
            U, V = fn(T); per, mean = score(U, V, T); results.setdefault(tag, {})[n] = dict(mean=mean, per_snr=per)
        print(f'{tag:34s} official {results[tag]["official"]["mean"]:.3f} | T2_d5 {results[tag]["T2_d5"]["mean"]:.3f}', flush=True)
    print('\n--- classical baselines (heatmap init) ---', flush=True)
    for mode in ['peaks', 'newton+cyclic', 'newton+cyclic+selfcal']: evaluate('classical ' + mode, lambda T, mode=mode: run_classical(T, mode))
    print('\n--- untrained learned models ---', flush=True)
    evaluate('V1 pasted design, untrained', lambda T: run_learned(m1, T)); evaluate('V2 normalised, untrained', lambda T: run_learned(m2, T))
    print('\n--- training V1 ---', flush=True); h1 = train(m1, 'V1', 1)
    evaluate('V1 pasted design, trained', lambda T: run_learned(m1, T))
    print('\n--- training V2 ---', flush=True); h2 = train(m2, 'V2', 2)
    evaluate('V2 normalised, trained', lambda T: run_learned(m2, T))
    json.dump(dict(results=results, loss_v1=h1[::20], loss_v2=h2[::20], params=dict(v1=npar(m1), v2=npar(m2)), steps=STEPS, batch=BS), open('results.json', 'w'))
    print('done', flush=True)
