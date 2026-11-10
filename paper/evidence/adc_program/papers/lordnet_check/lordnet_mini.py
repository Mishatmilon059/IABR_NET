"""Small sanity re-implementation of LoRD-Net (Khobahi, Shlezinger, Soltanalian, Eldar, IEEE TSP 2021, arXiv 2102.02993v1).
Purpose: check the paper's MECHANISM (Sec. III, eqs. 6, 10-16, Alg. 1) in its own toy setting, not to reproduce its exact numbers.
Setting as in Sec. IV-A: real BPSK, m=128 receive antennas, n=16 users, one-bit ADCs with zero thresholds b=0, C=I known,
H ~ i.i.d. Gaussian (one realisation per run, the same H for the B pilots and the test symbols), L=30 layers, delta=0.01,
Stage 1 Adam lr 1e-3 / 400 epochs over H, Stage 2 Adam lr 1e-4 / 400 epochs over diagonal G_i = W_i W_i^T, minibatch 512.
NOT stated in the paper and chosen here (assumptions): x0 = 0; Stage-1 H initialised either i.i.d. N(0, SNR/n) independent of the
true H ('rand') or from the pilot correlation R_p^T X_p / B rescaled to RMS sqrt(SNR/n) ('corr'); Stage-1 minibatch 512 (stated only
for Stage 2);
SNR set by scaling H (entries N(0, SNR/n)) with sigma = 1 so that SNR = E||Hx||^2 / E||n||^2 (eq. 17) with C = I;
baseline 'relaxed ML, true H' = 700 plain gradient steps on the NLL with the step chosen by grid search on the training pilots
(the paper's nML of Choi et al. [30] is a different algorithm and is NOT re-implemented here).
Seeds: channel + pilots 80,000,000 + 100*SNR + k; test symbols 12,900 + 100*SNR + k (this toy BPSK problem shares no data with
the angle-estimation banks).
"""
import os, time, json, math
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)
m, n, L, delta = 128, 16, 30, 0.01
B, T = int(os.environ.get('B', 512)), 2048
SNRS = [float(s) for s in os.environ.get('SNRS', '4,8').split(',')]
K = int(os.environ.get('K', 6))
EPOCHS = int(os.environ.get('EPOCHS', 400))
INITS = os.environ.get('INITS', 'rand,corr').split(',')


def logQ(z):          # log Q(z) = log P(N(0,1) > z), numerically stable
    return torch.special.log_ndtr(-z)


def eta(z):           # eta(z) = Q'(z)/Q(z) = -phi(z)/Q(z)
    return -torch.exp(-0.5 * z * z - 0.5 * math.log(2 * math.pi) - logQ(z))


def nll(H, x, r):     # paper eq. (3) as printed (with its minus sign) = NEGATIVE log-likelihood; b = 0, C = I
    return -logQ(r * (0.0 - x @ H.T)).sum()


def grad6(H, x, r):   # paper eq. (6): H^T R~ eta(R~(b - Hx)), R~ = R C^{-1/2} = Diag(r); batched over rows of x
    return (r * eta(r * (0.0 - x @ H.T))) @ H


def unfold(H, g, r, Lyr):   # eqs. (10)-(12): x_{i+1} = x_i - G_i z_i with diagonal G_i (g: (Lyr, n)); x0 = 0
    x = torch.zeros(r.shape[0], H.shape[1])
    outs = []
    for i in range(Lyr):
        x = x - g[i] * grad6(H, x, r)
        outs.append(x)
    return x, outs


def ber(xhat, x):
    return float((torch.sign(xhat) != x).double().mean())


def data(H, N, rng):
    x = rng.choice([-1.0, 1.0], size=(N, n))
    y = x @ H.T + rng.standard_normal((N, m))
    return torch.tensor(x), torch.tensor(np.where(y >= 0, 1.0, -1.0))


def train(params, lossf, lr, epochs, Np, bs=512):
    opt = torch.optim.Adam(params, lr=lr)
    loss = None
    for ep in range(epochs):
        perm = torch.randperm(Np)
        for s in range(0, Np, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            loss = lossf(idx)
            loss.backward()
            opt.step()
    return float(loss)


t00 = time.time()
# --- 1. check eq. (6) against automatic differentiation of eq. (3) ---
g0 = torch.Generator().manual_seed(0)
Hc = torch.randn(m, n, generator=g0)
xc = torch.randn(1, n, generator=g0).requires_grad_(True)
rc = torch.sign(torch.randn(1, m, generator=g0))
nll(Hc, xc, rc).backward()
err = float((xc.grad - grad6(Hc, xc.detach(), rc)).abs().max() / xc.grad.abs().max())
print(f'eq.(6) vs autograd of eq.(3) (negative log-likelihood): max rel. difference {err:.2e}', flush=True)

res = {}
for snr_db in SNRS:
    snr = 10 ** (snr_db / 10)
    rows = []
    for k in range(K):
        rng = np.random.default_rng(80_000_000 + 100 * int(snr_db) + k)
        Htrue = rng.standard_normal((m, n)) * math.sqrt(snr / n)
        Xp, Rp = data(Htrue, B, rng)
        Xt, Rt = data(Htrue, T, np.random.default_rng(12_900 + 100 * int(snr_db) + k))
        Ht = torch.tensor(Htrue)
        torch.manual_seed(80_000_000 + k)
        r = {}
        # (a) relaxed ML with the TRUE H: 700 plain GD steps, step size grid-searched on the pilots
        best = None
        for st in [3e-4, 1e-3, 3e-3, 1e-2, 1e-1]:
            with torch.no_grad():
                xp, _ = unfold(Ht, torch.full((700, n), st), Rp, 700)
            b_ = ber(xp, Xp)
            if best is None or b_ < best[0]:
                best = (b_, st)
        with torch.no_grad():
            xt, _ = unfold(Ht, torch.full((700, n), best[1]), Rt, 700)
        r['relaxedML_trueH_700'] = ber(xt, Xt)
        r['relaxedML_step'] = best[1]
        # (b) true H, basic policy (delta I, 30 layers)
        with torch.no_grad():
            xt, _ = unfold(Ht, torch.full((L, n), delta), Rt, L)
        r['trueH_deltaI_30'] = ber(xt, Xt)
        for init in INITS:
            # (c) Stage 1: learn the surrogate H (competitive objective) with G_i = delta I fixed, eq. (15)
            if init == 'rand':
                H0 = torch.randn(m, n) * math.sqrt(snr / n)
            else:
                H0 = Rp.T @ Xp / B
                H0 = H0 / H0.pow(2).mean().sqrt() * math.sqrt(snr / n)
            with torch.no_grad():
                r[f'{init}_init_only'] = ber(unfold(H0, torch.full((L, n), delta), Rt, L)[0], Xt)
            Hs = H0.clone().requires_grad_(True)
            gfix = torch.full((L, n), delta)
            train([Hs], lambda idx: ((unfold(Hs, gfix, Rp[idx], L)[0] - Xp[idx]) ** 2).sum(1).mean(), 1e-3, EPOCHS, B)
            with torch.no_grad():
                xt, outs1 = unfold(Hs, gfix, Rt, L)
                r[f'{init}_stage1'] = ber(xt, Xt)
                r[f'{init}_stage1_train'] = ber(unfold(Hs, gfix, Rp, L)[0], Xp)
                r[f'{init}_cos_learnedH_trueH'] = float((Hs * Ht).sum() / (Hs.norm() * Ht.norm()))
                r[f'{init}_normratio_learnedH_trueH'] = float(Hs.norm() / Ht.norm())
            # (d) Stage 2: learn diagonal preconditioners G_i = W_i W_i^T, H fixed to the Stage-1 surrogate, eq. (16)
            Hs_f = Hs.detach()
            w = torch.full((L, n), math.sqrt(delta)).requires_grad_(True)
            train([w], lambda idx: ((unfold(Hs_f, w * w, Rp[idx], L)[0] - Xp[idx]) ** 2).sum(1).mean(), 1e-4, EPOCHS, B)
            with torch.no_grad():
                xt, outs2 = unfold(Hs_f, w * w, Rt, L)
                r[f'{init}_stage2'] = ber(xt, Xt)
                r[f'{init}_stage2_per_layer'] = [ber(o, Xt) for o in outs2]
                r[f'{init}_stage1_per_layer'] = [ber(o, Xt) for o in outs1]
        # (e) 'perfect CSI' LoRD-Net of the paper's Fig. 10: true H, only the diagonal G_i learned
        w2 = torch.full((L, n), math.sqrt(delta)).requires_grad_(True)
        train([w2], lambda idx: ((unfold(Ht, w2 * w2, Rp[idx], L)[0] - Xp[idx]) ** 2).sum(1).mean(), 1e-4, EPOCHS, B)
        with torch.no_grad():
            xt, _ = unfold(Ht, w2 * w2, Rt, L)
            r['trueH_learnedG'] = ber(xt, Xt)
        rows.append(r)
        print(f'SNR {snr_db:4.1f} dB ch {k}: relaxedML(trueH,700 it, step {best[1]:g}) {r["relaxedML_trueH_700"]:.5f} | '
              f'trueH deltaI 30 {r["trueH_deltaI_30"]:.5f} | trueH+learnedG {r["trueH_learnedG"]:.5f} | ' +
              ' | '.join(f'{i}: init {r[i + "_init_only"]:.4f} S1 {r[i + "_stage1"]:.5f} (train {r[i + "_stage1_train"]:.5f}) S2 {r[i + "_stage2"]:.5f} '
                         f'cos {r[i + "_cos_learnedH_trueH"]:.3f} norm {r[i + "_normratio_learnedH_trueH"]:.2f}' for i in INITS) +
              f'  [{time.time() - t00:.0f}s]', flush=True)
    keys = ['relaxedML_trueH_700', 'trueH_deltaI_30', 'trueH_learnedG'] + [f'{i}_{q}' for i in INITS for q in
            ['init_only', 'stage1', 'stage1_train', 'stage2', 'cos_learnedH_trueH', 'normratio_learnedH_trueH']]
    summ = {kk: float(np.mean([rr[kk] for rr in rows])) for kk in keys}
    for i in INITS:
        summ[f'{i}_stage1_per_layer_mean'] = np.mean([rr[f'{i}_stage1_per_layer'] for rr in rows], 0).round(5).tolist()
        summ[f'{i}_stage2_per_layer_mean'] = np.mean([rr[f'{i}_stage2_per_layer'] for rr in rows], 0).round(5).tolist()
        summ[f'{i}_channels_stage2_better_than_relaxedML'] = int(sum(rr[f'{i}_stage2'] < rr['relaxedML_trueH_700'] for rr in rows))
        summ[f'{i}_channels_stage2_better_than_trueH_learnedG'] = int(sum(rr[f'{i}_stage2'] < rr['trueH_learnedG'] for rr in rows))
    summ['channels'] = K
    summ['relaxedML_steps_chosen'] = [rr['relaxedML_step'] for rr in rows]
    res[f'{snr_db:g}dB'] = summ
    print(f'== SNR {snr_db:g} dB, mean BER over {K} channels x {T * n} test bits each:',
          json.dumps({k_: (round(v, 5) if isinstance(v, float) else v) for k_, v in summ.items() if 'per_layer' not in k_}), flush=True)
    for i in INITS:
        print(f'   {i}: per-layer BER Stage 1 (layers 1,4,...,28):', summ[f'{i}_stage1_per_layer_mean'][::3], flush=True)
        print(f'   {i}: per-layer BER Stage 2 (layers 1,4,...,28):', summ[f'{i}_stage2_per_layer_mean'][::3], flush=True)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lordnet_mini.json')
json.dump(dict(B=B, T=T, K=K, m=m, n=n, L=L, delta=delta, epochs=EPOCHS, results=res, eq6_check_rel_err=err), open(out, 'w'), indent=1)
print(f'done in {time.time() - t00:.0f}s')
