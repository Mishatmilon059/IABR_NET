"""Design A 'CA-UNOMP' and design B 'HU-NOMP' (unfolded_v2/design/DESIGN.md), both UNFOLDED FROM NOMP (Mamandipoor et al. 2016).
Fixed classical skeleton (torch port of notebook_src.py newton / cyclic / ls_gains / nomp_greedy, verified equal in unit_tests.py):
  greedy detection with residual subtraction on an oversampled FFT grid -> 2D Newton on |a^H r|^2 -> cyclic re-refinement (LS gains,
  other paths subtracted).  Quantised data (GNOMP / EM style, as finetune/adversarial/mlq.py): every cyclic round starts with an
  E-step z_bar = E[z | ADC cell, mu, s] (truncated-normal mean), and the NOMP machinery runs on the de-quantised antenna channel;
  the final 'polish' layers are Levenberg-Marquardt steps on the exact interval-censored (or Gaussian, b = inf) likelihood with
  heteroscedastic std s_i = sqrt(kappa^2 sigma_hat^2 + eta |mu_i|^2).
Learned parts (all optional; with every flag off and true sigma the core IS NOMP, see unit_tests.py):
  Front: seed CNN IABRv2(32,10,32) from adc_ft.pt + bit/energy FiLM + scene head (log sigma_hat, impairment type, L_hat);
  HyperNet: per-layer kappa, damping, step clip, step-MLP gate, self-cal strength, heat weight, eta, stop bias;
  Fusion: small conv over [FFT power of residual, CNN heat, detected mask, log sigma_hat, bits] -> detection map;
  StepMLP (A only): zero-initialised correction of the Newton step;  StopHead: accept/stop for unknown L;
  Selector (A only): transformer over the hypotheses (no pairwise-separation or gain-prior feature).
"""
from bcommon import *

LOG2PI = 0.5 * math.log(2 * math.pi)
ST0 = 2 * math.pi / 64                     # NOMP Newton step clip (half a 32-grid cell)


class Consts:
    def __init__(self, dtype=torch.float64):
        self.rt = dtype; self.ct = torch.complex128 if dtype == torch.float64 else torch.complex64
        n = torch.arange(16, dtype=dtype)
        self.NN, self.MM = torch.meshgrid(n, n, indexing='ij')
        self.nf = self.NN.reshape(-1); self.mf = self.MM.reshape(-1); self.n = n
        self.W = torch.tensor(SS.W_IDEAL, dtype=self.ct); self.F = torch.tensor(SS.F_IDEAL, dtype=self.ct)
        self.WH = self.W.conj().T.contiguous(); self.FT = self.F.T.contiguous(); self.FH = self.F.conj().T.contiguous()


_CONSTS = {}


def K_(dtype):
    if dtype not in _CONSTS:
        _CONSTS[dtype] = Consts(dtype)
    return _CONSTS[dtype]


def to_ant_t(Yb, k):
    return k.W @ Yb @ k.FH


def to_beam_t(H, k):
    return k.WH @ H @ k.F


def ri(Z):
    R = Z.shape[0]
    return torch.cat([Z.real.reshape(R, -1), Z.imag.reshape(R, -1)], 1)


def cplx(z):
    R = z.shape[0]
    return torch.complex(z[:, :256], z[:, 256:]).reshape(R, 16, 16)


def atoms_t(U, V, k):
    ph = -k.nf[None, :, None] * U[:, None, :] + k.mf[None, :, None] * V[:, None, :]
    return torch.complex(torch.cos(ph), torch.sin(ph))


def ls_gains_t(y, U, V, k):
    A = atoms_t(U, V, k); AH = A.conj().transpose(1, 2)
    G = AH @ A + 1e-9 * torch.eye(U.shape[1], dtype=k.ct)[None]
    x = torch.linalg.solve(G, AH @ y.reshape(len(y), -1, 1))[..., 0]
    return x, A


def model_ant(x, A):
    return (A * x[:, None, :]).sum(-1).reshape(len(x), 16, 16)


def newton_terms(r, u, v, k):
    ph = k.NN[None] * u[:, None, None] - k.MM[None] * v[:, None, None]
    E = torch.complex(torch.cos(ph), torch.sin(ph)); rE = r * E
    S = rE.sum((1, 2)); Su = (1j * k.NN * rE).sum((1, 2)); Sv = (-1j * k.MM * rE).sum((1, 2))
    Suu = (-k.NN ** 2 * rE).sum((1, 2)); Svv = (-k.MM ** 2 * rE).sum((1, 2)); Suv = (k.NN * k.MM * rE).sum((1, 2))
    gu = 2 * (S.conj() * Su).real; gv = 2 * (S.conj() * Sv).real
    huu = 2 * (Su.conj() * Su + S.conj() * Suu).real; hvv = 2 * (Sv.conj() * Sv + S.conj() * Svv).real
    huv = 2 * (Su.conj() * Sv + S.conj() * Suv).real
    return S, gu, gv, huu, hvv, huv


def nll_terms_t(mu, lo, hi, s, isq):
    """per-component NLL, d/dmu, d2/dmu2 (Fisher-type, >= 0). isq (R,) bool: quantised rows use the interval-censored Gaussian
    (lo, hi), the others a Gaussian around lo (= the observation). Safe for both branches (no NaN gradients)."""
    q = isq[:, None]
    lo_s = torch.where(q, lo, lo - 1.0); hi_s = torch.where(q, hi, lo + 1.0)
    a = (lo_s - mu) / s; b = (hi_s - mu) / s
    flip = (a + b) > 0
    a2 = torch.where(flip, -b, a); b2 = torch.where(flip, -a, b)
    la = torch.special.log_ndtr(a2); lb = torch.special.log_ndtr(b2)
    logZ = lb + torch.log1p(-torch.exp(torch.clamp(la - lb, max=-1e-12)))
    ra = torch.exp(-0.5 * a * a - LOG2PI - logZ); rb = torch.exp(-0.5 * b * b - LOG2PI - logZ)
    gq = (rb - ra) / s
    hq = torch.clamp((b * rb - a * ra) / s ** 2 + gq * gq, min=1e-9) / 1.0
    hq = torch.maximum(hq, 1e-9 / s ** 2)
    e = (mu - lo) / s
    fg = 0.5 * e * e + torch.log(s); gg = e / s; hg = 1.0 / s ** 2 + 0 * e
    return torch.where(q, -logZ, fg), torch.where(q, gq, gg), torch.where(q, hq, hg)


def parts_t(U, V, k):
    r = torch.exp(torch.complex(torch.zeros_like(U)[..., None], -k.n * U[..., None]))   # (R,K,16)
    t = torch.exp(torch.complex(torch.zeros_like(V)[..., None], k.n * V[..., None]))
    rho = torch.einsum('qn,bkn->bkq', k.WH, r); tau = torch.einsum('pm,bkm->bkp', k.FT, t)
    drho = torch.einsum('qn,bkn->bkq', k.WH, -1j * k.n * r); dtau = torch.einsum('pm,bkm->bkp', k.FT, 1j * k.n * t)
    return rho, tau, drho, dtau


def jac_t(U, V, X, k):
    """beamspace model mu (R,512 real) and Jacobian (R,512,4K) wrt [u_1..K, v_1..K, Re x_1..K, Im x_1..K] (torch port of mlq.Model.jac)"""
    rho, tau, drho, dtau = parts_t(U, V, k)
    At = torch.einsum('bkq,bkp->bkqp', rho, tau)
    mu = torch.einsum('bk,bkqp->bqp', X, At)
    Ju = X[:, :, None, None] * torch.einsum('bkq,bkp->bkqp', drho, tau)
    Jv = X[:, :, None, None] * torch.einsum('bkq,bkp->bkqp', rho, dtau)
    Jc = torch.cat([Ju, Jv, At, 1j * At], 1)
    R = len(U); Jc = Jc.reshape(R, Jc.shape[1], 256)
    J = torch.cat([Jc.real, Jc.imag], 2).transpose(1, 2)
    m = mu.reshape(R, 256)
    return torch.cat([m.real, m.imag], 1), J


def mu_t(U, V, X, k):
    rho, tau, _, _ = parts_t(U, V, k)
    m = torch.einsum('bk,bkq,bkp->bqp', X, rho, tau).reshape(len(U), 256)
    return torch.cat([m.real, m.imag], 1)


def wrap(u):
    return torch.atan2(torch.sin(u), torch.cos(u))


def uv_to_deg(U, V):
    cu = torch.clamp(wrap(U) / math.pi, -1 + 1e-4, 1 - 1e-4); cv = torch.clamp(wrap(V) / math.pi, -1 + 1e-4, 1 - 1e-4)
    return torch.rad2deg(torch.arccos(cu)), torch.rad2deg(torch.arccos(cv))


def local_max_rank(P, ranks):
    """P (R,G,G); ranks (R,) long: index of the (rank+1)-th largest local maximum (3x3 wrap) per row -> flat index (R,)"""
    Pp = TF.pad(P[:, None], (1, 1, 1, 1), mode='circular')
    mx = TF.max_pool2d(Pp, 3, 1)[:, 0]
    big = P.detach().reshape(len(P), -1).abs().max(1)[0][:, None, None] * 10 + 10
    score = torch.where(P >= mx, P + big, P).reshape(len(P), -1)
    top = torch.topk(score, 3, 1)[1]
    return top.gather(1, ranks[:, None])[:, 0]


def up64(h32):
    """(R,32,32) -> (R,64,64) circular linear interpolation on the 64 FFT grid (cell 2q = heat cell q)"""
    R = h32.shape[0]
    a = torch.zeros(R, 64, 32, dtype=h32.dtype); a[:, 0::2] = h32; a[:, 1::2] = 0.5 * (h32 + torch.roll(h32, -1, 1))
    b = torch.zeros(R, 64, 64, dtype=h32.dtype); b[:, :, 0::2] = a; b[:, :, 1::2] = 0.5 * (a + torch.roll(a, -1, 2))
    return b


def blob64(U, V, valid, width=1.5):
    """(R,K) estimates -> Gaussian blobs on the 64 grid (detected mask / detection targets); same index map as NOMP"""
    R = U.shape[0]; g = torch.arange(64, dtype=U.dtype)
    if U.shape[1] == 0:
        return torch.zeros(R, 64, 64, dtype=U.dtype)
    q = torch.remainder(-U * 64 / (2 * math.pi), 64); p = torch.remainder(V * 64 / (2 * math.pi), 64)
    dq = (g[None, None] - q[..., None]).abs(); dq = torch.minimum(dq, 64 - dq)
    dp = (g[None, None] - p[..., None]).abs(); dp = torch.minimum(dp, 64 - dp)
    e = torch.exp(-dq[..., :, None] ** 2 / (2 * width ** 2) - dp[..., None, :] ** 2 / (2 * width ** 2)) * valid[..., None, None]
    return e.max(1)[0]


# ================================================================ learned modules
class Front(nn.Module):
    """seed CNN (IABRv2 32/10/32, initialised from adc_ft.pt) + per-block FiLM from bits / energy + scene head"""
    def __init__(self, ckpt=A.FT_CKPT):
        super().__init__()
        self.cnn = A.load_cnn(ckpt)
        self.film = nn.Linear(6, 2 * 32 * 10); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.scene = nn.Sequential(nn.Linear(45, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                   nn.Linear(256, 15))

    @staticmethod
    def hand_stats(Y, bits_oh):
        with torch.no_grad():
            R = Y.shape[0]
            e = (Y.abs() ** 2).reshape(R, -1)
            le = torch.log(e.mean(1) + 1e-12)
            kurt = torch.log((e ** 2).mean(1) / (e.mean(1) ** 2 + 1e-12) + 1e-12)
            top = torch.maximum(Y.real.abs(), Y.imag.abs()).reshape(R, -1).max(1)[0]
            outer = ((Y.real.abs() >= 0.999 * top[:, None, None]) | (Y.imag.abs() >= 0.999 * top[:, None, None])).float().reshape(R, -1).mean(1)
            outer = outer * (1 - bits_oh[:, 4])
            P = torch.fft.fft2(Y, s=(32, 32)).abs() ** 2          # beamspace oversampled power (peaks ~ paths)
            Pp = TF.pad(P[:, None], (1, 1, 1, 1), mode='circular'); mx = TF.max_pool2d(Pp, 3, 1)[:, 0]
            sc = torch.where(P >= mx, P, torch.zeros_like(P)).reshape(R, -1)
            t5 = torch.topk(sc, 5, 1)[0] / (P.reshape(R, -1).sum(1, keepdim=True) + 1e-12)
            return torch.cat([outer[:, None], le[:, None] / 10, kurt[:, None], torch.log(t5 + 1e-9) / 5], 1)   # 8

    def forward(self, Y, bits_oh):
        """Y (R,16,16) complex64 beamspace (quantised or not); bits_oh (R,5). Returns heat logits (R,32,32), imp, trunk feat, scene out."""
        m = self.cnn
        y = torch.stack([Y.real, Y.imag], 1).float()
        Hc, Ht, imp, Yc = m.front(y)
        Yn = Yc / (Yc.abs().pow(2).mean((1, 2), keepdim=True).sqrt() + 1e-8)
        x = torch.stack([Yn.real, Yn.imag, torch.log(Yn.abs() ** 2 + 1e-3)], 1)
        x = torch.relu(m.stem(TF.pad(x, (1, 1, 1, 1), mode='circular')))
        le = torch.log((Y.abs() ** 2).mean((1, 2)) + 1e-12).float()[:, None] / 10
        fm = self.film(torch.cat([bits_oh.float(), le], 1)).reshape(-1, 10, 2, 32)
        for i, b in enumerate(m.blocks):
            x = b(x)
            x = x * (1 + fm[:, i, 0, :, None, None]) + fm[:, i, 1, :, None, None]
        heat = TF.pixel_shuffle(m.head2(torch.relu(m.head1(x))), m.r)[:, 0]
        feat = torch.cat([x.mean((2, 3)), self.hand_stats(Y, bits_oh).float(), bits_oh.float()], 1)
        so = self.scene(feat)
        return dict(heat=heat, imp=imp, logsig=so[:, 0], typ=so[:, 1:6], Lh=so[:, 6:15])


NOUT = 8   # kappa, damp, clip, gate, sc_lam, beta, eta, stop_bias


class HyperNet(nn.Module):
    def __init__(self, T=24):
        super().__init__()
        self.emb = nn.Embedding(T, 16); self.T = T
        self.mlp = nn.Sequential(nn.Linear(36, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NOUT))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, bits_oh, lsig_rel, typ_p, L_p):
        """-> (R, T, NOUT) raw outputs (zero at init = classical settings)"""
        R = bits_oh.shape[0]
        ctx = torch.cat([bits_oh, lsig_rel[:, None], typ_p, L_p], 1).float()                 # 5 + 1 + 5 + 9 = 20
        x = torch.cat([ctx[:, None].expand(R, self.T, 20), self.emb.weight[None].expand(R, self.T, 16)], 2)
        return self.mlp(x)


def hp_decode(o):
    """raw hypernet outputs (R, NOUT) -> physical per-layer settings (classical values at o = 0 up to ~1e-3)"""
    return dict(kappa=1 + F_softplus(o[:, 0] - 4), damp=F_softplus(o[:, 1] - 4), clip=torch.exp(0.7 * torch.tanh(o[:, 2])),
                gate=o[:, 3], lam=torch.sigmoid(o[:, 4]), beta=o[:, 5], eta=F_softplus(o[:, 6] - 5) * 0.1, stop=o[:, 7],
                lmlam=10 ** (o[:, 1] - 2.0))


def F_softplus(x):
    return TF.softplus(x)


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(5, 8, 1); self.c2 = nn.Conv2d(8, 8, 3); self.c3 = nn.Conv2d(8, 1, 3)
        nn.init.zeros_(self.c3.weight); nn.init.zeros_(self.c3.bias)

    def forward(self, x):
        h = torch.relu(self.c1(x))
        h = torch.relu(self.c2(TF.pad(h, (1, 1, 1, 1), mode='circular')))
        return self.c3(TF.pad(h, (1, 1, 1, 1), mode='circular'))[:, 0]


class StepMLP(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, f):
        return torch.tanh(self.net(f))


class StopHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(19, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, f):
        return self.net(f)[:, 0]


class Selector(nn.Module):
    def __init__(self, d_in=16):
        super().__init__()
        self.inp = nn.Linear(d_in, 128)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = nn.Linear(128, 1)

    def forward(self, f):          # (B, K, d) -> (B, K) scores
        return self.out(self.enc(self.inp(f)))[..., 0]


# ================================================================ the unfolded decoder
class UNOMP(nn.Module):
    """design='A' (step MLP + 6 hypotheses + learned selector) or 'B' (2 hypotheses, lowest-NLL pick).
    Inference/training entry: forward(batch) with batch from prep_batch()."""
    def __init__(self, design='B', rc=3, inner=2, n_polish=int(os.environ.get('NPOL', 12)), sc_iters=2, sc_rc=2, front_ckpt=A.FT_CKPT):
        super().__init__()
        self.design = design; self.rc, self.inner, self.n_polish, self.sc_iters, self.sc_rc = rc, inner, n_polish, sc_iters, sc_rc
        self.front = Front(front_ckpt); self.hyper = HyperNet(); self.fusion = Fusion(); self.stop = StopHead()
        self.det_scale = nn.Parameter(torch.tensor(20.0))
        if design == 'A':
            self.step = StepMLP(); self.selector = Selector()
            # own per-layer gate (init 0.1): gate x zero-init MLP would otherwise be a zero-gradient saddle (deviation 10)
            self.step_gate = nn.Parameter(torch.full((24,), 0.1))
        # switches for ablations / classical-equivalent configuration
        self.cfg = dict(classical=False, true_sigma=False, kappa1=False, eta0=False, step_off=False, sel_nll=False, beta0=False,
                        polish=True, sc=True, hyps=None, os_=4, dtype=torch.float64, train_stop=False)

    # ------------------------------------------------------------ helpers
    def hp(self, H, t):
        o = H[:, min(t, H.shape[1] - 1)]
        d = hp_decode(o)
        c = self.cfg
        if c['classical']:
            one = torch.ones_like(d['kappa'])
            d = dict(kappa=one, damp=0 * one, clip=one, gate=0 * one, lam=0 * one, beta=0 * one, eta=0 * one, stop=0 * one, lmlam=1e-2 * one)
        if c['kappa1']:
            d['kappa'] = torch.ones_like(d['kappa'])
        if c['eta0']:
            d['eta'] = torch.zeros_like(d['eta'])
        if c['beta0']:
            d['beta'] = torch.zeros_like(d['beta'])
        if c['step_off']:
            d['gate'] = torch.zeros_like(d['gate'])
        return d

    def estep(self, ob, U, V, Hd, p):
        """E-step: de-quantised antenna channel given the current paths (LS gains on the current channel Hd). Unquantised rows: y."""
        k = ob['k']
        if not bool(ob['isq'].any()):
            return ob['yant']
        if U.shape[1] == 0:
            mu = torch.zeros_like(ob['lo'])
        else:
            x, Am = ls_gains_t(Hd, U, V, k); mu = ri(to_beam_t(model_ant(x, Am), k))
        s = self.noise_std(ob, mu, p)
        _, g, _ = nll_terms_t(mu, ob['lo'], ob['hi'], s, ob['isq'])
        zb = mu - s ** 2 * g
        Hq = to_ant_t(cplx(zb), k)
        return torch.where(ob['isq'][:, None, None], Hq, ob['yant'])

    def noise_std(self, ob, mu, p):
        m2 = (mu[:, :256] ** 2 + mu[:, 256:] ** 2).repeat(1, 2)
        sg = ob['sig'][:, None].to(mu.dtype)
        return torch.sqrt((p['kappa'][:, None].to(mu.dtype) * sg) ** 2 + p['eta'][:, None].to(mu.dtype) * m2)

    def newton(self, ob, r, u, v, iters, p, t):
        k = ob['k']
        for _ in range(iters):
            S, gu, gv, huu, hvv, huv = newton_terms(r, u, v, k)
            det = huu * hvv - huv ** 2; ok = (huu < 0) & (det > 0)
            dmp = p['damp'].to(u.dtype)
            if self.cfg['classical'] or float(dmp.detach().abs().max()) == 0.0:
                a, b, sd = huu, hvv, torch.where(ok, det, torch.ones_like(det))
            else:
                scl = 0.5 * (huu.abs() + hvv.abs()); a = huu - dmp * scl; b = hvv - dmp * scl
                sd = torch.where(ok, a * b - huv ** 2, torch.ones_like(det))
            du = torch.where(ok, -(b * gu - huv * gv) / sd, torch.zeros_like(gu))
            dv = torch.where(ok, -(-huv * gu + a * gv) / sd, torch.zeros_like(gv))
            st = ST0 * p['clip'].to(u.dtype)
            if self.design == 'A' and not self.cfg['classical'] and not self.cfg['step_off']:
                s2 = (S.abs() ** 2).detach() + 1e-12
                f = torch.stack([gu / s2, gv / s2, huu / (100 * s2), hvv / (100 * s2), huv / (100 * s2), ok.to(u.dtype),
                                 torch.log(s2 / (65536 * 2 * ob['sig'].to(u.dtype) ** 2) + 1e-6) / 10,
                                 p['kappa'].to(u.dtype) - 1, torch.full_like(u, t / 24), torch.log(ob['sig'].to(u.dtype)) / 5,
                                 10 * p['eta'].to(u.dtype)], 1)
                f = torch.cat([f, ob['bits_oh'].to(u.dtype)], 1).float()
                corr = self.step(torch.clamp(torch.nan_to_num(f), -50, 50)).to(u.dtype)
                g = self.step_gate[min(t, 23)].to(u.dtype)
                du = du + g * st * corr[:, 0]; dv = dv + g * st * corr[:, 1]
            u = u + torch.maximum(torch.minimum(du, st), -st)
            v = v + torch.maximum(torch.minimum(dv, st), -st)
        return u, v

    def cyclic(self, ob, Hd, U, V, rounds, p, t):
        k = ob['k']
        for _ in range(rounds):
            Hd = self.estep(ob, U, V, Hd, p)
            for j in range(U.shape[1]):
                x, Am = ls_gains_t(Hd, U, V, k)
                tot = (Am * x[:, None, :]).sum(-1); own = Am[:, :, j] * x[:, j:j + 1]
                r = Hd - (tot - own).reshape(Hd.shape)
                u, v = self.newton(ob, r, U[:, j], V[:, j], self.inner, p, t)
                U = torch.cat([U[:, :j], u[:, None], U[:, j + 1:]], 1); V = torch.cat([V[:, :j], v[:, None], V[:, j + 1:]], 1)
        return U, V, Hd

    # ------------------------------------------------------------ hypotheses
    def greedy(self, ob, Hh, L, ranks, n_det=None):
        """greedy detection (NOMP structure) with fused detection map; ranks (R, Lmax) long = which local max to take at each step.
        Returns U, V, Hd, per-step stop features and the detection-loss terms."""
        k = ob['k']; R = ob['yant'].shape[0]; G = 16 * self.cfg['os_']
        n_det = n_det or L
        U = torch.zeros(R, 0, dtype=k.rt); V = torch.zeros(R, 0, dtype=k.rt)
        Hd = ob['yant']; feats, dets, states = [], [], []
        for kk in range(n_det):
            p = self.hp(Hh, kk)
            Hd = self.estep(ob, U, V, Hd, p)
            if U.shape[1]:
                x, Am = ls_gains_t(Hd, U, V, k); res = Hd - model_ant(x, Am)
            else:
                res = Hd
            P = torch.fft.fft2(res, s=(G, G)).abs()
            if self.cfg['classical']:
                fused = P
            else:
                Pn = P / (P.reshape(R, -1).max(1)[0][:, None, None] + 1e-12)
                h64 = up64(ob['heat32']) if G == 64 else None
                mask = blob64(U.detach(), V.detach(), torch.ones_like(U))
                xin = torch.stack([Pn, h64, mask, torch.ones_like(Pn) * torch.log(ob['sig_rel'])[:, None, None],
                                   torch.ones_like(Pn) * ob['bits_oh'][:, 4][:, None, None].to(Pn.dtype)], 1).float()
                fused = Pn.float() + p['beta'][:, None, None].float() * torch.sigmoid(h64.float()) + self.fusion(xin)
                dets.append((fused, U.detach(), V.detach()))
            idx = local_max_rank(fused.detach(), ranks[:, kk]) if ranks is not None else fused.detach().reshape(R, -1).argmax(1)
            u0 = (-2 * math.pi * (idx // G) / G).to(k.rt); v0 = (2 * math.pi * (idx % G) / G).to(k.rt)
            e_before = (res.abs() ** 2).sum((1, 2))
            u, v = self.newton(ob, res, u0, v0, 4, p, kk)
            U = torch.cat([U, u[:, None]], 1); V = torch.cat([V, v[:, None]], 1)
            U, V, Hd = self.cyclic(ob, Hd, U, V, self.rc, p, kk)
            if self.cfg['train_stop'] or ob.get('unknown_L', False):
                x, Am = ls_gains_t(Hd, U, V, k); e_after = ((Hd - model_ant(x, Am)).abs() ** 2).sum((1, 2))
                s2 = 2 * ob['sig'].to(k.rt) ** 2
                q = torch.remainder(-u * 32 / (2 * math.pi), 32).round().long() % 32; pp = torch.remainder(v * 32 / (2 * math.pi), 32).round().long() % 32
                hv = torch.sigmoid(ob['heat32'][torch.arange(R), q, pp])
                f = torch.cat([torch.log1p(torch.clamp((e_before - e_after) / s2, min=0))[:, None] / 5,
                               torch.log(x[:, -1].abs() ** 2 / s2 + 1e-9)[:, None] / 10, hv[:, None].to(k.rt),
                               torch.full((R, 1), (kk + 1) / 8, dtype=k.rt), ob['bits_oh'].to(k.rt), torch.log(ob['sig_rel'])[:, None].to(k.rt) / 5,
                               ob['Lp'].to(k.rt)], 1).float()
                feats.append(self.stop(f) + p['stop'].float())
                states.append((U, V, Hd))
        return U, V, Hd, feats, dets, states

    def heat_seeded(self, ob, Hh, L):
        k = ob['k']; R = ob['yant'].shape[0]
        h = torch.sigmoid(ob['heat32']).detach()
        Pp = TF.pad(h[:, None], (1, 1, 1, 1), mode='circular'); mx = TF.max_pool2d(Pp, 3, 1)[:, 0]
        score = torch.where(h >= mx, h + 1.0, h).reshape(R, -1)
        idx = torch.topk(score, L, 1)[1]
        U0 = (-2 * math.pi * (idx // 32) / 32).to(k.rt); V0 = (2 * math.pi * (idx % 32) / 32).to(k.rt)
        p = self.hp(Hh, 0)
        Hd = self.estep(ob, U0[:, :0], V0[:, :0], ob['yant'], p)
        rep = Hd[:, None].expand(R, L, 16, 16).reshape(R * L, 16, 16)
        obr = dict(ob); obr['sig'] = ob['sig'].repeat_interleave(L); obr['bits_oh'] = ob['bits_oh'].repeat_interleave(L, 0)
        pr = {kk: vv.repeat_interleave(L) for kk, vv in p.items()}
        u, v = self.newton(obr, rep, U0.reshape(-1), V0.reshape(-1), 4, pr, 0)
        U, V = u.reshape(R, L), v.reshape(R, L)
        p2 = self.hp(Hh, L - 1)
        U, V, Hd = self.cyclic(ob, Hd, U, V, self.rc, p2, L - 1)
        return U, V, Hd

    def selfcal(self, ob, Hd, U, V, p):
        """differentiable self-calibration (port of notebook selfcal / exp5 StagedSC.calibrate), strength lam (hypernet), unquantised
        rows only, tuned gate tau = 2; returns corrected antenna observation and refined paths."""
        k = ob['k']; lam = p['lam'].to(k.rt) * (~ob['isq']).to(k.rt)
        if float(lam.detach().abs().max()) == 0.0:
            return ob['yant'], U, V
        y = ob['yant']; yc = y; nc = (torch.arange(16, dtype=k.rt) - 7.5)
        for _ in range(self.sc_iters):
            x, Am = ls_gains_t(yc, U, V, k); Hm = model_ant(x, Am)
            snr_est = (Hm.abs() ** 2).sum((1, 2)) / ((yc - Hm).abs() ** 2).sum((1, 2)).clamp_min(1e-12)
            dt = torch.ones(len(y), 16, dtype=k.ct)
            for _ in range(3):
                M = Hm * dt[:, None, :]; dr = (y * M.conj()).sum(2) / (M.abs() ** 2).sum(2).clamp_min(1e-12)
                M = Hm * dr[:, :, None]; dt = (y * M.conj()).sum(1) / (M.abs() ** 2).sum(1).clamp_min(1e-12)

            def clean(d):
                lg = torch.log(d.abs() + 1e-12)
                dphi = torch.angle(d[:, 1:] * d[:, :-1].conj()); ph = torch.cat([torch.zeros_like(dphi[:, :1]), torch.cumsum(dphi, 1)], 1)
                lg = lg - lg.mean(1, keepdim=True); ph = ph - ph.mean(1, keepdim=True); ph = ph - (ph @ nc)[:, None] / (nc @ nc) * nc
                return lam[:, None] * lg, lam[:, None] * ph
            lr_, pr_ = clean(dr); lt_, pt_ = clean(dt)
            gate = torch.sigmoid((snr_est - 2.0) * 4) if self.training else (snr_est > 2.0).to(k.rt)
            cr = gate[:, None] * torch.exp(torch.complex(-lr_, -pr_)) + (1 - gate[:, None]); ct = gate[:, None] * torch.exp(torch.complex(-lt_, -pt_)) + (1 - gate[:, None])
            yc = y * cr[:, :, None] * ct[:, None, :]
            obc = dict(ob); obc['yant'] = yc
            U, V, _ = self.cyclic(obc, yc, U, V, self.sc_rc, p, 8)
        return yc, U, V

    def polish(self, ob, yobs, Hd, U, V, Hh):
        """joint LM on the exact likelihood of the (self-calibrated) observation; accept only NLL decreases (classical safeguard)"""
        k = ob['k']
        x, _ = ls_gains_t(Hd, U, V, k)
        y_ri = ri(to_beam_t(yobs, k))
        lo = torch.where(ob['isq'][:, None], ob['lo'], y_ri); hi = torch.where(ob['isq'][:, None], ob['hi'], y_ri)
        Kp = U.shape[1]; eye = torch.eye(4 * Kp, dtype=k.rt)[None]
        X = x
        lam_st = None
        for i in range(self.n_polish):
            p = self.hp(Hh, 9 + min(i, 12))
            mu, J = jac_t(U, V, X, k); s = self.noise_std(ob, mu.detach(), p)
            f, g, h = nll_terms_t(mu, lo, hi, s, ob['isq']); F0 = f.sum(1)
            Am = torch.einsum('bik,bi,bil->bkl', J, h, J); gr = torch.einsum('bik,bi->bk', J, g)
            D = torch.diagonal(Am, dim1=1, dim2=2)
            # classical LM damping schedule (mlq.lm: /3 on accept, x4 on reject) times a learned per-layer factor (1 at init)
            lam = (1e-2 if lam_st is None else lam_st) * (p['lmlam'].to(k.rt) / 1e-2)
            d = -torch.linalg.solve(Am + (lam[:, None] * (D + 1e-9))[:, :, None] * eye, gr[..., None])[..., 0]
            du = torch.clamp(d[:, :Kp], -0.1, 0.1); dv = torch.clamp(d[:, Kp:2 * Kp], -0.1, 0.1)
            Un, Vn, Xn = U + du, V + dv, X + torch.complex(d[:, 2 * Kp:3 * Kp], d[:, 3 * Kp:])
            mun = mu_t(Un, Vn, Xn, k); fn, _, _ = nll_terms_t(mun, lo, hi, s, ob['isq'])
            okk = (fn.sum(1) < F0)[:, None]
            U = torch.where(okk, Un, U); V = torch.where(okk, Vn, V); X = torch.where(okk, Xn, X)
            lam_st = torch.clamp(torch.where(okk[:, 0], lam / 3, lam * 4), 1e-7, 1e7).detach()
        return U, V, X, lo, hi

    def nll_final(self, ob, U, V, X, lo, hi, p, kappa1=True):
        k = ob['k']
        mu = mu_t(U, V, X, k)
        if kappa1:
            p = dict(p); p['kappa'] = torch.ones_like(p['kappa']); p['eta'] = torch.zeros_like(p['eta'])
        s = self.noise_std(ob, mu, p)
        f, _, _ = nll_terms_t(mu, lo, hi, s, ob['isq'])
        return f.sum(1)

    def finish(self, ob, Hh, U, V, Hd):
        """self-cal (b = inf rows) -> polish -> final NLLs"""
        k = ob['k']
        Upre, Vpre = U, V
        yobs = ob['yant']
        if self.cfg['sc'] and not self.cfg['classical']:
            yobs, U, V = self.selfcal(ob, Hd, U, V, self.hp(Hh, 8))
            Hd = torch.where(ob['isq'][:, None, None], Hd, yobs)
        if self.cfg['polish']:
            U, V, X, lo, hi = self.polish(ob, yobs, Hd, U, V, Hh)
        else:
            X, _ = ls_gains_t(Hd, U, V, k); y_ri = ri(to_beam_t(yobs, k))
            lo = torch.where(ob['isq'][:, None], ob['lo'], y_ri); hi = torch.where(ob['isq'][:, None], ob['hi'], y_ri)
        pT = self.hp(Hh, 9 + min(max(self.n_polish - 1, 0), 12))
        return dict(U=U, V=V, X=X, Upre=Upre, Vpre=Vpre, nll1=self.nll_final(ob, U, V, X, lo, hi, pT, True),
                    nllT=self.nll_final(ob, U, V, X, lo, hi, pT, False), eta=pT['eta'], kappa=pT['kappa'])

    # ------------------------------------------------------------ full forward
    def scene_ctx(self, bt):
        """front end -> obs dict + hypernet outputs. bt: dict of torch tensors from prep_batch."""
        dt = self.cfg['dtype']; k = K_(dt)
        fo = self.front(bt['Y'].to(torch.complex64), bt['bits_oh'])
        Yd = bt['Y'].to(k.ct)
        rms = torch.sqrt((Yd.abs() ** 2).mean((1, 2)) / 2)
        sig_hat = rms * torch.exp(fo['logsig'].to(dt))
        if self.cfg['true_sigma'] or self.cfg['classical']:
            sig = bt['sig_true'].to(dt)
        else:
            sig = sig_hat
        typ_p = torch.softmax(fo['typ'], 1); L_p = torch.softmax(fo['Lh'], 1)
        ob = dict(k=k, yant=to_ant_t(Yd, k), lo=bt['lo'].to(dt), hi=bt['hi'].to(dt), isq=bt['isq'], sig=sig, sig_rel=(sig / rms).detach(),
                  bits_oh=bt['bits_oh'].to(dt), heat32=fo['heat'].detach().to(dt), Lp=L_p.detach().to(dt))
        Hh = self.hyper(bt['bits_oh'].float(), torch.log(sig / rms).float(), typ_p, L_p)
        return ob, Hh, fo

    def forward(self, bt, L, unknown_L=False):
        ob, Hh, fo = self.scene_ctx(bt)
        B = ob['yant'].shape[0]
        hyps = self.cfg['hyps'] or (['H1', 'H2'] if self.design == 'B' else ['H1', 'H2', 'H3', 'H4', 'H5', 'H6'])
        greedy_h = [h for h in hyps if h != 'H2']
        rank_map = dict(H1=(0, 0), H3=(1, 0), H4=(2, 0), H5=(0, 1), H6=(0, 2))
        out = dict(front=fo, ob=ob, hyps=hyps)
        n_det = L + 1 if self.cfg['train_stop'] else L
        res = {}
        if greedy_h:
            G = len(greedy_h)
            obg = rep_ob(ob, G); Hg = Hh.repeat(G, 1, 1)
            Lm = max(n_det, 2)
            ranks = torch.zeros(G * B, Lm, dtype=torch.long)
            for gi, h in enumerate(greedy_h):
                ranks[gi * B:(gi + 1) * B, 0] = rank_map[h][0]; ranks[gi * B:(gi + 1) * B, 1] = rank_map[h][1]
            U, V, Hd, feats, dets, states = self.greedy(obg, Hg, L, ranks, n_det)
            out['stop_logits'] = feats; out['dets'] = dets; out['states'] = states
            if n_det > L:          # stop-head training: evaluate the hit loss on the state after L detections
                U, V, Hd = states[L - 1]
            fin = self.finish(obg, Hg, U, V, Hd)
            for gi, h in enumerate(greedy_h):
                res[h] = {kk: vv[gi * B:(gi + 1) * B] for kk, vv in fin.items()}
        if 'H2' in hyps:
            U, V, Hd = self.heat_seeded(ob, Hh, L)
            res['H2'] = self.finish(ob, Hh, U, V, Hd)
        out['res'] = res
        # selection
        nll1 = torch.stack([res[h]['nll1'] for h in hyps], 1)
        if self.design == 'A' and not self.cfg['sel_nll'] and not self.cfg['classical']:
            sf = self.sel_feats(ob, res, hyps, nll1)
            out['sel_logits'] = self.selector(sf)
            pick = out['sel_logits'].argmax(1)
        else:
            pick = nll1.argmin(1)
        out['pick'] = pick
        Us = torch.stack([res[h]['U'] for h in hyps], 1); Vs = torch.stack([res[h]['V'] for h in hyps], 1)
        ar = torch.arange(B)
        out['U'] = Us[ar, pick]; out['V'] = Vs[ar, pick]
        return out

    def decode_unknown(self, bt, Lmax=8):
        """unknown number of paths: greedy hypothesis H1 with the learned accept/stop head (candidate k accepted while
        logit > 0), then self-cal + polish of the accepted paths. Returns list of (U, V) numpy per scene."""
        old = self.cfg['train_stop']; self.cfg['train_stop'] = False
        ob, Hh, fo = self.scene_ctx(bt); ob['unknown_L'] = True
        B = ob['yant'].shape[0]
        U, V, Hd, feats, dets, states = self.greedy(ob, Hh, Lmax, None, Lmax)
        acc = torch.stack(feats, 1) > 0
        kst = torch.cumprod(acc.long(), 1).sum(1)
        out = [None] * B
        for kv in sorted(set(kst.tolist())):
            ii = torch.where(kst == kv)[0]
            if kv == 0:
                for i in ii.tolist():
                    out[i] = (np.zeros(0), np.zeros(0))
                continue
            Uk, Vk, Hk = states[kv - 1]
            obs = sub_ob(ob, ii)
            fin = self.finish(obs, Hh[ii], Uk[ii], Vk[ii], Hk[ii])
            for j, i in enumerate(ii.tolist()):
                out[i] = (fin['U'][j].numpy(), fin['V'][j].numpy())
        self.cfg['train_stop'] = old
        return out

    def sel_feats(self, ob, res, hyps, nll1):
        k = ob['k']; B = nll1.shape[0]
        nllT = torch.stack([res[h]['nllT'] for h in hyps], 1)
        d1 = (nll1 - nll1.min(1, keepdim=True)[0]) / 10; dT = (nllT - nllT.min(1, keepdim=True)[0]) / 10
        rk = nll1.argsort(1).argsort(1).to(k.rt) / len(hyps)
        s2 = 2 * ob['sig'].to(k.rt) ** 2
        rows = []
        for h in hyps:
            X = res[h]['X']; lg = torch.log(X.abs() ** 2 / s2[:, None] + 1e-9) / 10
            q = torch.remainder(-res[h]['U'] * 32 / (2 * math.pi), 32).round().long() % 32; pp = torch.remainder(res[h]['V'] * 32 / (2 * math.pi), 32).round().long() % 32
            hv = torch.sigmoid(ob['heat32'][torch.arange(B)[:, None], q, pp])
            rows.append(torch.stack([lg.mean(1), lg.min(1)[0], lg.max(1)[0], hv.mean(1), hv.min(1)[0],
                                     torch.full_like(lg[:, 0], X.shape[1] / 8), 10 * res[h]['eta'].to(k.rt), res[h]['kappa'].to(k.rt) - 1], 1))
        R = torch.stack(rows, 1)                                       # (B,K,8)
        g = torch.cat([torch.log(ob['sig_rel'])[:, None] / 5, ob['bits_oh'].to(k.rt)], 1)[:, None].expand(B, len(hyps), 6)
        f = torch.cat([torch.clamp(d1, max=50)[..., None], torch.clamp(dT, max=50)[..., None], rk[..., None], R, g[..., :0]], 2)
        f = torch.cat([f, g[..., :5]], 2)                              # 3 + 8 + 5 = 16
        return torch.nan_to_num(f).float()


def sub_ob(ob, ii):
    return {kk: (vv[ii] if torch.is_tensor(vv) else vv) for kk, vv in ob.items()}


def rep_ob(ob, G):
    if G == 1:
        return ob
    o = {}
    for kk, vv in ob.items():
        if torch.is_tensor(vv):
            o[kk] = vv.repeat(G, *([1] * (vv.dim() - 1)))
        else:
            o[kk] = vv
    return o


# ================================================================ batch preparation
def prep_batch(Y, bits, snr=None):
    """numpy Y (B,16,16) complex beamspace (possibly quantised), bits (B,) int (0 = none) -> torch dict"""
    lo, hi = cells(Y, bits)
    oh = np.zeros((len(Y), 5)); oh[np.arange(len(Y)), [BITS_LIST.index(int(b)) for b in bits]] = 1
    bt = dict(Y=torch.from_numpy(Y.astype(np.complex128)), lo=torch.from_numpy(lo), hi=torch.from_numpy(hi), isq=torch.from_numpy(bits > 0),
              bits_oh=torch.from_numpy(oh))
    bt['sig_true'] = torch.from_numpy(sigma_of(snr)) if snr is not None else torch.ones(len(Y), dtype=torch.float64)
    return bt


def decode_np(model, Y, bits, L, snr=None, bs=200, unknown_L=False):
    """inference helper -> U, V numpy (B, L)"""
    model.eval(); Us, Vs, picks = [], [], []
    with torch.no_grad():
        for s in range(0, len(Y), bs):
            bt = prep_batch(Y[s:s + bs], bits[s:s + bs], None if snr is None else snr[s:s + bs])
            o = model(bt, L)
            Us.append(o['U'].numpy()); Vs.append(o['V'].numpy()); picks.append(o['pick'].numpy())
    return np.concatenate(Us), np.concatenate(Vs), np.concatenate(picks)
