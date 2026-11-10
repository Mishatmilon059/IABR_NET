"""Shared helpers for the severe-array-imperfection fine-tuning study (finetune/imp). Read-only use of existing code:
  decoder / metrics / CNN / classical ADC-stage helpers : finetune/adc/acommon.py (imported as module A; nothing there is modified)
     (notebook_src.py slices [527:647] decoder, [649:668] parabolic network-only read-out, [419:522] IABRv2, focal loss [685:691])
  simulator                                            : stress/stress_sim.py (base_scenes, generate) for every TEST bank
  IABC-v2b corrector + EB-ML classical self-cal        : iabc_v2/adversarial/a_lib.py (v2a_best.pt + v2b_best.pt, read only)
Threads: OMP/OPENBLAS/MKL 6 and torch 6 (set before numpy / torch import).
Seeds used by this stage (all checked unused elsewhere with grep): training batches default_rng([71000000, step]); monitor batch
default_rng([71000001, 0]); unit test of the generator 71000002; fresh replicate banks 8400 (SNR 0) / 8415 (SNR 15); stress banks
(re-used, identical Y) 5000 / 5015; official bank (frozen); bootstrap seed 0."""
import os, sys, json, time
for _k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_k] = os.environ.get('THREADS', '6')
os.environ.setdefault('IABR2_SUITE', 'D:/ai_ml_project')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/finetune/adc')
import acommon as A                         # noqa: E402  (np, torch, C, SS, load_cnn, cnn_forward, fft_heat, ...)
np, torch, nn, TF, C, SS = A.np, A.torch, A.nn, A.TF, A.C, A.SS
import iabr2_sim as SIM                     # noqa: E402

torch.set_num_threads(int(os.environ.get('THREADS', 6)))
CACHE = f'{HERE}/cache'; os.makedirs(CACHE, exist_ok=True)
PCACHE = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/finetune_imp'
os.makedirs(PCACHE, exist_ok=True)                      # D: is almost full -> large per-scene arrays on C: (regenerable)
ORIG_CKPT = A.ORIG_CKPT
FT_CKPT = f'{CACHE}/imp_ft.pt'
SC, SC_TUNED = A.SC, A.SC_TUNED
STRESS_SEEDS = {0: 5000, 15: 5015}
FRESH_SEEDS = {0: 8400, 15: 8415}
CFG = dict(train_L=(1, 9), train_snr=(-15, 24), clean_frac=0.2, train_phase_max=8.0, train_gain_max=3.0, grid=32, heat_sigma=1.0)
_n = np.arange(16.)
FAM_CODE = {'orig': 0, 'M1': 1, 'M2': 2, 'M3': 3, 'M6': 6}
EXT = dict(M3=(45.0, 6.0), M2=0.2, M6=30.0, M1=0.3)     # upper ends of the extended training ranges


# ---------------------------------------------------------------- forward model with explicit draws (follows stress_sim.generate)
def coupling(rho, th1, th2):
    """per-sample rho (B,) version of stress_sim.coupling_matrix: C[i,i]=1, C[i,i+-1]=rho e^{j th1}, C[i,i+-2]=(rho/2) e^{j th2}."""
    B = len(th1); Cm = np.zeros((B, 16, 16), complex); i = np.arange(16)
    Cm[:, i, i] = 1.0
    c1 = rho * np.exp(1j * th1); c2 = (rho / 2) * np.exp(1j * th2)
    for d, c in [(1, c1), (2, c2)]:
        Cm[:, i[:-d], i[d:]] = c[:, None]; Cm[:, i[d:], i[:-d]] = c[:, None]
    return Cm


def synth(psi, phi, alpha, noise, Ur, Ut, Gr, Gt, sig2=None, zr=None, zt=None, rho=None, th=None, pdm=None, gdm=None, sig6=None, z6=None):
    """Vectorised per-sample-severity version of stress_sim.generate (M1, M2, M3, M6; each severity array is (B,), 0 = off):
    sig2 M2 position sigma (half-wavelengths) with standard draws zr, zt (B,16); rho M1 coupling with phases th (4,B);
    pdm / gdm M3 phase (deg) / gain (dB) maxima with uniforms Ur, Ut, Gr, Gt (B,16); sig6 M6 phase-noise sigma (deg) with z6 (B,16,16).
    Order exactly as generate: steering -> H -> C_r H C_t^T -> diag(D) codebooks -> + noise -> phase noise. Returns Y, D_r, D_t."""
    B = alpha.shape[0]; z = np.zeros(B)
    sig2 = z if sig2 is None else sig2; rho = z if rho is None else rho; pdm = z if pdm is None else pdm
    gdm = z if gdm is None else gdm; sig6 = z if sig6 is None else sig6
    ps = np.nan_to_num(psi); ph = np.nan_to_num(phi)
    xr = _n[None] + (0 if zr is None else sig2[:, None] * zr); xt = _n[None] + (0 if zt is None else sig2[:, None] * zt)
    xr = np.broadcast_to(xr, (B, 16)); xt = np.broadcast_to(xt, (B, 16))
    a_r = np.exp(-1j * np.pi * np.cos(ps)[..., None] * xr[:, None, :]) / 4.0
    a_t = np.exp(-1j * np.pi * np.cos(ph)[..., None] * xt[:, None, :]) / 4.0
    H = 16.0 * np.einsum('bl,bln,blm->bnm', alpha, a_r, np.conj(a_t))
    if th is not None and np.any(rho > 0):
        Cr = coupling(rho, th[0], th[1]); Ct = coupling(rho, th[2], th[3])
        H = Cr @ H @ np.transpose(Ct, (0, 2, 1))
    eps_r = Ur * np.deg2rad(pdm)[:, None]; eps_t = Ut * np.deg2rad(pdm)[:, None]
    g_r = 10 ** (Gr * gdm[:, None] / 20); g_t = 10 ** (Gt * gdm[:, None] / 20)
    D_r = g_r * np.exp(1j * eps_r); D_t = g_t * np.exp(1j * eps_t)
    W = D_r[:, :, None] * SS.W_IDEAL[None]; F = D_t[:, :, None] * SS.F_IDEAL[None]
    Y = np.einsum('bnq,bnm,bmp->bqp', np.conj(W), H, F) + noise
    if z6 is not None and np.any(sig6 > 0):
        Y = Y * np.exp(1j * np.deg2rad(sig6)[:, None, None] * z6)
    return Y, D_r, D_t


def imp_batch(rng, B, frac_orig=0.4):
    """Training batch: round(frac_orig*B) samples from the ORIGINAL training distribution (iabr2_sim.make_batch, notebook CFG) and the
    rest from the extended impairments, ONE family per sample, uniformly among M3 / M2 / M6 / M1:
      M3 phase max U(0,45) deg AND gain max U(0,6) dB (per-antenna uniform draws in +-max, as simulate / generate);
      M2 position sigma U(0,0.2) half-wavelengths;  M6 phase-noise sigma U(0,30) deg;  M1 coupling rho U(0,0.3).
    L U{1..9}, SNR U{-15..24} dB (integer, as make_batch). Returns y (B,2,16,16) f32, heat (B,32,32), imp (B,32,2), mask (B,) [1 where an
    identifiable-diagonal IABC target exists: original distribution and M3], fam (B,) codes FAM_CODE."""
    B0 = int(round(frac_orig * B)); B1 = B - B0
    d = SIM.make_batch(rng, B0, CFG)
    L = rng.integers(1, 10, B1); snr = rng.integers(-15, 25, B1).astype(float)
    fam = np.array([3, 2, 6, 1])[rng.integers(0, 4, B1)]
    s1 = rng.uniform(size=B1); s2 = rng.uniform(size=B1)
    psi, phi, alpha = SIM.sample_paths(L, rng, lmax=9)
    Ur, Ut, Gr, Gt = rng.uniform(-1, 1, (4, B1, 16))
    zr, zt = rng.standard_normal((2, B1, 16)); th = rng.uniform(0, 2 * np.pi, (4, B1)); z6 = rng.standard_normal((B1, 16, 16))
    s = np.sqrt(10 ** (-snr / 10) / 2)[:, None, None]
    noise = s * (rng.standard_normal((B1, 16, 16)) + 1j * rng.standard_normal((B1, 16, 16)))
    Y, D_r, D_t = synth(psi, phi, alpha, noise, Ur, Ut, Gr, Gt,
                        sig2=np.where(fam == 2, EXT['M2'] * s1, 0.0), zr=zr, zt=zt,
                        rho=np.where(fam == 1, EXT['M1'] * s1, 0.0), th=th,
                        pdm=np.where(fam == 3, EXT['M3'][0] * s1, 0.0), gdm=np.where(fam == 3, EXT['M3'][1] * s2, 0.0),
                        sig6=np.where(fam == 6, EXT['M6'] * s1, 0.0), z6=z6)
    y1 = np.ascontiguousarray(SIM.to_ri(Y).transpose(0, 3, 1, 2))
    h1 = SIM.heat_targets(psi, phi, CFG['grid'], CFG['heat_sigma'])
    i1 = SIM.impairment_targets(D_r, D_t)
    y = np.concatenate([d['y'], y1]); heat = np.concatenate([d['heat'], h1]); imp = np.concatenate([d['imp'], i1])
    mask = np.concatenate([np.ones(B0, np.float32), (fam == 3).astype(np.float32)])
    famc = np.concatenate([np.zeros(B0, int), fam])
    return (torch.from_numpy(y), torch.from_numpy(heat), torch.from_numpy(imp.astype(np.float32)), torch.from_numpy(mask), famc)


# ---------------------------------------------------------------- networks
def load_cnn(path):
    return A.load_cnn(path)


def net_forward(cnn, Y, bs=250):
    """full CNN (its own IABC front) -> sigmoid heat (B,32,32), Hc (B,16,16), imp (B,32,2)."""
    X = np.ascontiguousarray(SIM.to_ri(Y).transpose(0, 3, 1, 2)); hs, hc, im = [], [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            o = cnn(torch.from_numpy(X[s:s + bs]))
            hs.append(torch.sigmoid(o['heat']).double().numpy()); hc.append(o['Hc'].numpy().astype(complex)); im.append(o['imp'].double().numpy())
    return np.concatenate(hs), np.concatenate(hc), np.concatenate(im)


def trunk_heat(cnn, Yc, bs=250):
    """trunk + heat head of `cnn` on a given (already corrected) beamspace Yc = W^H Hc F: identical to IABRv2.forward after front()
    (notebook lines 497-520; same as iabc_v2/v2common.trunk_heat but for any model). Returns sigmoid heat."""
    out = []
    with torch.no_grad():
        for s in range(0, len(Yc), bs):
            Y = torch.from_numpy(Yc[s:s + bs].astype(np.complex64))
            Yn = Y / (Y.abs().pow(2).mean((1, 2), keepdim=True).sqrt() + 1e-8)
            x = torch.stack([Yn.real, Yn.imag, torch.log(Yn.abs() ** 2 + 1e-3)], 1)
            x = torch.relu(cnn.stem(TF.pad(x, (1, 1, 1, 1), mode='circular')))
            for b in cnn.blocks:
                x = b(x)
            h = TF.pixel_shuffle(cnn.head2(torch.relu(cnn.head1(x))), cnn.r)[:, 0]
            out.append(torch.sigmoid(h).double().numpy())
    return np.concatenate(out)


def to_beam(H):
    return np.matmul(np.matmul(SS.W_IDEAL.conj().T[None], H), SS.F_IDEAL[None])


def apply_corr(Ht, est):
    """IABRv2.front convention: Hc = Ht * c_r * c_t, c = exp(-lg - j ph), est (B,32,2) [phase, log-gain]."""
    c = np.exp(-est[..., 1] - 1j * est[..., 0])
    return Ht * c[:, :16, None] * c[:, None, 16:]


_AL = None


def AL():
    """lazy import of iabc_v2/adversarial/a_lib (loads v2a_best.pt, v2b_best.pt and the original CNN; read only)."""
    global _AL
    if _AL is None:
        sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/iabc_v2/adversarial')
        import a_lib
        _AL = a_lib
    return _AL


def v2b_est(Ht, L):
    return AL().v2b_est(Ht, L)


def ebml_g5(Ht, L, UV0):
    """EB-ML-g5 + NOMP (iabc_v2/adversarial: joint ML + James-Stein shrinkage, applied only where the blind SNR estimate >= 5 dB)."""
    al = AL(); e, UV, _, snr_hat = al.eb_ml(Ht, L)
    _, (U, V) = al.gate(e, UV, UV0, snr_hat, 5.0)
    return U, V


# ---------------------------------------------------------------- metrics (paper-style Pd, |err| <= 1 deg; boot within SNR groups, seed 0)
stats, pack, as_list, save_json = A.stats, A.pack, A.as_list, A.save_json


def boot_table(S, snr, pairs, B=2000):
    Bt = C.boot_metrics(S, snr, B=B, seed=0)
    pdm = {m: C.metric_by_snr(S[m], snr)['mean']['pd_paper'] for m in S}
    out = {}
    for a_, b_ in pairs:
        if a_ in S and b_ in S and a_ != b_:
            d = Bt[a_][:, 0] - Bt[b_][:, 0]
            out[f'{a_} - {b_}'] = [pdm[a_] - pdm[b_], float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    return pdm, {m: list(C.ci(Bt[m][:, 0])[1:]) for m in S}, out
