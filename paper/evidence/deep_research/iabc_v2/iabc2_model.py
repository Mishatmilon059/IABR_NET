"""IABC-v2 model + training-batch generator (importable by DataLoader workers; light imports only).

Antenna-domain model (exact, codebooks are unitary -- checked in t0_timing.log):
    Ht = W Y F^H = conj(D_r) H D_t + Z',  Z' ~ CN(0, sigma^2 I), sigma^2 = 10^(-SNR/10),
    H[n,m] = sum_l alpha_l exp(-j n u_l + j m v_l),  u = pi cos(psi), v = pi cos(phi).
Target (iabr2_sim.impairment_targets): identifiable [phase rad, natural-log gain] per antenna, rows 0-15 Rx, 16-31 Tx
(phase: mean + linear ramp removed; gain: mean removed). Output convention identical to the old IABC (drop-in for IABRv2.front).

Architecture (IABCv2): 32 antenna tokens + 1 global token -> pre-norm Transformer encoder -> per-token (phase, log-gain) head ->
projection onto the identifiable subspace.
  Rx token n : row n of Hn (16 complex), row n of R_r = Hn Hn^H / 16 (16 complex), row n of the coherence
               R_r[n,k] / sqrt(R_r[n,n] R_r[k,k]) (16 complex), old-IABC features (log row energy, strongest-beam detrended phase)
  Tx token m : the same computed on Hn^H (i.e. conj of column m), so Rx and Tx tokens have identical statistics; the Tx phase
               output is sign-flipped back (column distortion of Ht^H is conj(d_t)).
  global     : sorted log singular values of Hn (noise floor / rank cue -> implicit SNR and L estimate)
  Hn = Ht / RMS(Ht) (scale-free; absolute power is NOT given to the network)."""
import os, math
import numpy as np
import torch, torch.nn as nn
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
import sys
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2')
import iabr2_sim as S

W_T = torch.tensor(S.W_IDEAL, dtype=torch.complex64)
F_T = torch.tensor(S.F_IDEAL, dtype=torch.complex64)
_N16 = torch.arange(16, dtype=torch.float32) - 7.5
P = Q = 16
_COS_T = torch.tensor((1 / np.pi) * np.angle(np.exp(1j * (2 * np.pi / P) * np.arange(P))), dtype=torch.float32)
_COS_R = torch.tensor((1 / np.pi) * np.angle(np.exp(-1j * (2 * np.pi / Q) * np.arange(Q))), dtype=torch.float32)

TRAIN_CFG = dict(L=(1, 9), snr=(-15.0, 25.0), clean_frac=0.2, phase_max=45.0, gain_max=6.0, close_frac=0.1, close_sep=(np.pi / 90, np.pi / 12))


def t_project(x):
    n = _N16.to(x.device)
    x = x - x.mean(-1, keepdim=True)
    return x - (x @ n)[..., None] / (n @ n) * n


def to_ident(out):
    """(B,32,2) raw -> identifiable subspace (same projection as the old IABC.forward)."""
    ph = torch.cat([t_project(out[:, :16, 0]), t_project(out[:, 16:, 0])], 1)
    lg = torch.cat([out[:, :16, 1] - out[:, :16, 1].mean(1, keepdim=True), out[:, 16:, 1] - out[:, 16:, 1].mean(1, keepdim=True)], 1)
    return torch.stack([ph, lg], -1)


def _detrended_phase(z):
    d = torch.angle(z[:, 1:] * torch.conj(z[:, :-1]))
    ph = torch.cat([torch.zeros_like(d[:, :1]), torch.cumsum(d, 1)], 1)
    return t_project(ph)


def _side_tokens(Hn, u_strong):
    """Hn (B,16,16) complex, tokens for the ROWS of Hn. u_strong: (B,) beam frequency of the strongest beam for these rows."""
    B = Hn.shape[0]
    R = Hn @ Hn.conj().transpose(1, 2) / 16
    dg = R.diagonal(dim1=1, dim2=2).real.clamp(min=1e-9)
    Co = R / torch.sqrt(dg[:, :, None] * dg[:, None, :])
    E = dg
    mag = 0.5 * torch.log(E / (E.mean(1, keepdim=True)) + 1e-9)
    k = torch.arange(16, dtype=torch.float32, device=Hn.device)
    rot = torch.exp(1j * np.pi * k[None] * u_strong[:, None])
    dph = _detrended_phase((Hn * rot[:, :, None]).sum(2))
    f = torch.cat([Hn.real, Hn.imag, R.real, R.imag, Co.real, Co.imag, mag[..., None], dph[..., None]], -1)   # (B,16,98)
    return torch.nan_to_num(f)


def features(Ht):
    """Ht (B,16,16) complex64 -> tokens (B,32,98), global (B,16)."""
    Hn = Ht / (Ht.abs().pow(2).mean((1, 2), keepdim=True).sqrt() + 1e-9)
    Y = W_T.conj().T.to(Ht.device) @ Hn @ F_T.to(Ht.device)
    idx = Y.abs().reshape(-1, 256).argmax(1)
    u_r = _COS_R.to(Ht.device)[idx // 16]; u_t = _COS_T.to(Ht.device)[idx % 16]
    fr = _side_tokens(Hn, u_r)
    fc = _side_tokens(Hn.conj().transpose(1, 2), u_t)       # rows of Hn^H = conj columns (conj of the old exp(-j pi m u_t) de-rotation)
    sv = torch.linalg.svdvals(Hn)
    g = torch.log(sv ** 2 + 1e-6) - torch.log(sv[:, :1] ** 2 + 1e-6)
    return torch.cat([fr, fc], 1), g


class IABCv2(nn.Module):
    def __init__(self, d=128, layers=6, heads=4, ff=384):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(98, d), nn.GELU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(16, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(0.02 * torch.randn(16, d))         # antenna index (shared by Rx and Tx)
        self.typ = nn.Parameter(0.02 * torch.randn(3, d))          # Rx / Tx / global
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff)

    def forward(self, Ht):
        f, g = features(Ht)
        t = self.tok(f) + torch.cat([self.pos + self.typ[0], self.pos + self.typ[1]], 0)[None]
        gt = self.glob(g)[:, None] + self.typ[2]
        h = self.enc(torch.cat([t, gt], 1))[:, :32]
        out = self.head(h)
        out = torch.stack([torch.cat([out[:, :16, 0], -out[:, 16:, 0]], 1), out[..., 1]], -1)   # Tx phase sign back
        return to_ident(out)


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def make_train_batch(rng, B, cfg=TRAIN_CFG):
    """Fresh synthetic scenes: L U{1..9}, SNR U(-15,25) dB, 20% clean, phase max U(0,45) deg, gain max U(0,6) dB,
    10% closely-spaced scenes (min separation U(2,15) deg, L>=2), else the original sampler (sep pi/6)."""
    L = rng.integers(cfg['L'][0], cfg['L'][1] + 1, B)
    snr = rng.uniform(*cfg['snr'], B)
    clean = rng.uniform(size=B) < cfg['clean_frac']
    pd = np.where(clean, 0.0, rng.uniform(0, cfg['phase_max'], B))
    gd = np.where(clean, 0.0, rng.uniform(0, cfg['gain_max'], B))
    close = rng.uniform(size=B) < cfg['close_frac']
    Lm = cfg['L'][1]
    psi = np.full((B, Lm), np.nan); phi = np.full((B, Lm), np.nan); alpha = np.zeros((B, Lm), complex)
    nc = np.where(~close)[0]
    if len(nc):
        a, b, c = S.sample_paths(L[nc], rng, lmax=Lm); psi[nc], phi[nc], alpha[nc] = a, b, c
    for i in np.where(close)[0]:
        L[i] = max(L[i], 2)
        a, b, c = S.sample_paths(L[i:i + 1], rng, sep=rng.uniform(*cfg['close_sep']), lmax=Lm)
        psi[i], phi[i], alpha[i] = a[0], b[0], c[0]
    sim = S.simulate(psi, phi, alpha, snr, rng, phase_deg_max=pd, gain_db_max=gd)
    Ht = np.matmul(np.matmul(S.W_IDEAL[None], sim['Y']), S.F_IDEAL.conj().T[None])
    return dict(Hr=np.ascontiguousarray(Ht.real, np.float32), Hi=np.ascontiguousarray(Ht.imag, np.float32),
                tgt=S.impairment_targets(sim['D_r'], sim['D_t']), snr=snr.astype(np.float32), L=L,
                pd=pd.astype(np.float32), gd=gd.astype(np.float32))


class Stream(torch.utils.data.IterableDataset):
    """Deterministic stream: batch k uses default_rng([seed, k]); starts at batch `start` (resume)."""
    def __init__(self, seed, start, B):
        super().__init__(); self.seed, self.start, self.B = seed, start, B

    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); nw = wi.num_workers if wi else 1; w = wi.id if wi else 0
        k = self.start + w
        while True:
            b = make_train_batch(np.random.default_rng([self.seed, k]), self.B)
            b['k'] = k
            yield {kk: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v) for kk, v in b.items()}
            k += nw


def load_v2(path, map_location='cpu'):
    s = torch.load(path, map_location=map_location)
    m = IABCv2(**s['cfg']); m.load_state_dict(s['model']); return m.eval()


@torch.no_grad()
def predict(model, Ht, bs=250):
    out = []
    for s in range(0, len(Ht), bs):
        out.append(model(torch.from_numpy(Ht[s:s + bs].astype(np.complex64))).double().numpy())
    return np.concatenate(out)


class Refiner(nn.Module):
    """IABC-v2b stage-2 (decoder in the loop). Per-antenna tokens (32): [blind estimate e_A (2), un-shrunk ALS estimate from NOMP angles
    on the e_A-corrected channel e_ALS (2), log per-antenna residual energy (1), log per-antenna model energy (1)] + global token
    [log10 SNR estimate from the model fit, L/9, log10 total residual / total energy]. Output Delta, final = to_ident(e_A + Delta).
    Learns how much of the ALS correction to trust per antenna / SNR (learned shrinkage + gating)."""
    def __init__(self, d=96, layers=3, heads=4, ff=192):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(6, d), nn.GELU(), nn.Linear(d, d))
        self.glob = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(0.02 * torch.randn(32, d)); self.gt = nn.Parameter(0.02 * torch.randn(1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff)

    def forward(self, tok, glob, eA):
        t = self.tok(tok) + self.pos[None]
        g = self.glob(glob)[:, None] + self.gt
        h = self.enc(torch.cat([t, g], 1))[:, :32]
        return to_ident(eA + self.head(h))


def load_ref(path):
    s = torch.load(path, map_location='cpu')
    m = Refiner(**s['cfg']); m.load_state_dict(s['model']); return m.eval()
