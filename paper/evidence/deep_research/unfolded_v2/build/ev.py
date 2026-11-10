"""Evaluation framework for unfolded_v2/build: test banks (DESIGN.md section 4 / criteria.json; fresh seeds 9520-9561), all named
baselines, per-scene stats cached per (bank, method) under BIG/stats (resumable, chunks <= 400 scenes).
usage: python ev.py GROUP [GROUP ...]  where GROUP in {official, regime, adcextra, unknownL, prior, ood, ablate}  (stdout -> ev_GROUP.log)"""
import os
os.environ.setdefault('THREADS', '6')
from ucore import *
import mlt
LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
STATS = f'{BIG}/stats'; os.makedirs(STATS, exist_ok=True)
GEN = 'D:/ai_ml_project/IABR_Net_TestSuite/data/generated_banks'
CH = 400


# ================================================================ banks
def _cached_bank(name):
    b = C.load_bank(name); z = np.load(f'{C.CACHE}/net__{name}.npz')
    Lm = b['psi'].shape[1]
    return dict(Y=to_beam(z['Ht'].astype(np.complex128)), bits=np.zeros(len(b['snr']), int), L=b['L'], psi=b['psi'], phi=b['phi'],
                snr=b['snr'].astype(float), cnn_heat=z['heat'].astype(np.float64), cnn_Hc=z['Hc'].astype(np.complex128), kind='clean')


def _stress(n, L, snr, seed, kind, sev=None, bits=0):
    b = SS.base_scenes(n, L, float(snr), seed)
    Y = SS.generate(b, kind, sev) if kind != 'clean' else SS.generate(b, 'clean')
    return dict(Y=Y, bits=np.full(n, bits), L=b['L'], psi=b['psi'], phi=b['phi'], snr=b['snr'], kind=kind)


def _fresh(n_per, L, snrs, seed, sep=np.pi / 6):
    rng = np.random.default_rng(seed); snr = np.repeat(np.asarray(snrs, float), n_per)
    if np.isscalar(L):
        b = gen_scenes(rng, len(snr), L, snr, sep=sep, seed_int=seed); Y = SS.generate(b, 'clean')
        return dict(Y=Y, bits=np.zeros(len(snr), int), L=b['L'], psi=b['psi'], phi=b['phi'], snr=snr, kind='clean')
    Ls = rng.integers(L[0], L[1] + 1, len(snr)); psi = np.full((len(snr), L[1]), np.nan); phi = psi.copy(); Y = np.zeros((len(snr), 16, 16), complex)
    for l in range(L[0], L[1] + 1):
        ii = np.where(Ls == l)[0]
        b = gen_scenes(np.random.default_rng([seed, l]), len(ii), l, snr[ii], sep=sep, seed_int=seed)
        Y[ii] = SS.generate(b, 'clean'); psi[ii, :l] = b['psi']; phi[ii, :l] = b['phi']
    return dict(Y=Y, bits=np.zeros(len(snr), int), L=Ls, psi=psi, phi=phi, snr=snr, kind='clean')


ADC_CELLS = {(1, 15): 9521, (2, 15): 9522, (3, 15): 9523, (4, 15): 9524, (1, 0): 9525, (2, 0): 9526, (3, 0): 9527, (4, 0): 9528}
IMP_CELLS = {'M2_0.2_snr15': ('M2', 0.2, 15, 9541), 'M6_30_snr15': ('M6', 30.0, 15, 9542), 'M6_30_snr0': ('M6', 30.0, 0, 9543),
             'M3_ph45_g0_snr0': ('M3', (45.0, 0.0), 0, 9544), 'M3_ph30_g4_snr15': ('M3', (30.0, 4.0), 15, 9545), 'M1_0.3_snr15': ('M1', 0.3, 15, 9546)}
REGIME = [f'M5_b{b}_snr{s}' for (b, s) in ADC_CELLS] + list(IMP_CELLS)


def load(name):
    if name == 'official_sub':                        # ablation subset: first 250 scenes of every SNR group of the official bank
        d = load('official'); snr = d['snr'].astype(int)
        idx = np.concatenate([np.where(snr == s)[0][:250] for s in sorted(set(snr.tolist()))])
        return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(snr) else v) for k, v in d.items()}
    if name.startswith('sub500:'):                    # ablation subset of a regime cell: first 500 scenes
        d = with_sev(name[7:], load(name[7:]))
        return {k: (v[:500] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(d['snr']) else v) for k, v in d.items()}
    if name in ('official', 'T2_d5') or name.startswith('sep_') or (name.startswith('L') and name[1].isdigit()):
        return _cached_bank(name)
    if name == 'rep9520':
        return _fresh(500, 3, range(-10, 26, 5), 9520)
    if name.startswith('M5_b') and name.count('_') == 2:
        b, s = int(name[4]), int(name.split('snr')[1]); return _stress(1000, 3, s, ADC_CELLS[(b, s)], 'M5', b, bits=b)
    if name in IMP_CELLS:
        k, sev, s, seed = IMP_CELLS[name]; return _stress(1000, 3, s, seed, k, sev)
    if name.startswith('stressM5_'):                  # stress seeds 5000 / 5015 replicate (report only)
        b, s = int(name.split('_b')[1][0]), int(name.split('snr')[1]); return _stress(1000, 3, s, {0: 5000, 15: 5015}[s], 'M5', b, bits=b)
    if name == 'M5c_b2_snr15':
        d = _stress(1000, 3, 15, 9529, 'M5c', 2, bits=2); return d
    if name.startswith('sweep_'):                     # sweep_b1_snr-5 ... seeds 9530-9537
        b, s = int(name.split('_b')[1][0]), int(name.split('snr')[1])
        seed = 9530 + [(1, -5), (1, 5), (1, 10), (1, 25), (2, -5), (2, 5), (2, 10), (2, 25)].index((b, s))
        return _stress(500, 3, s, seed, 'M5', b, bits=b)
    if name == 'M8_0.5_snr15':
        return _stress(1000, 3, 15, 9548, 'M8', 0.5)
    if name == 'M4_2_snr15':
        return _stress(1000, 3, 15, 9547, 'M4', 2)
    if name == 'Lmix9550':
        return _fresh(250, (1, 6), range(-10, 26, 5), 9550)
    if name == 'nosep9560':
        return _fresh(250, 3, [-10, -5, 0, 5], 9560, sep=0.0)
    if name == 'sep10_9561':
        return _fresh(250, 3, [-10, -5, 0, 5], 9561, sep=np.deg2rad(10))
    if name == 'valLmix':                             # validation only (threshold tuning of the NOMP stopping baseline)
        return _fresh(100, (1, 6), range(-10, 26, 5), 71_200_200)
    raise KeyError(name)


# ================================================================ methods
_NETS = {}


def net(name):
    if name not in _NETS:
        if name == 'orig':
            _NETS[name] = A.load_cnn(A.ORIG_CKPT)
        elif name == 'ft':
            _NETS[name] = A.load_cnn(A.FT_CKPT)
        elif name == 'lurd':
            sys.path.insert(0, LD)
            from exp5 import StagedSC, RefinerV3
            torch.set_num_threads(int(TH))
            m = StagedSC(RefinerV3(), pre=6, rounds=3, inner=2); m.load_state_dict(torch.load(f'{LD}/lurd.pt', map_location='cpu')); m.eval()
            _NETS[name] = m
        elif name in ('A', 'B'):
            m = UNOMP(name); m.load_state_dict(torch.load(MODEL_CKPT[name], map_location='cpu')); m.eval(); _NETS[name] = m
    return _NETS[name]


MODEL_CKPT = dict(A=os.environ.get('CKPT_A', f'{HERE}/final_A.pt'), B=os.environ.get('CKPT_B', f'{HERE}/final_B.pt'))


def heat_peaks(h, L):
    return C.g['heat_peaks'](h, L)


def orig_cnn(d, sl):
    if 'cnn_heat' in d:
        return d['cnn_heat'][sl], d['cnn_Hc'][sl]
    return A.cnn_forward(net('orig'), d['Y'][sl])


def by_L(Lv, fn):
    """fn(idx, L) -> (U, V) (len(idx), L); returns list of (psi, phi)"""
    out = [None] * len(Lv)
    for L in sorted(set(np.asarray(Lv).tolist())):
        ii = np.where(np.asarray(Lv) == L)[0]
        U, V = fn(ii, int(L)); ps, ph = C.uv_to_angles(U, V)
        for j, i in enumerate(ii):
            out[i] = (ps[j], ph[j])
    return out


def nomp_stop(H, sig2, Lmax=8, pfa=0.01, mult=1.0, os_=4, rc=3):
    """NOMP with its own residual-energy stopping rule (Mamandipoor et al. 2016): stop when max_(u,v) |a^H r|^2 / ||a||^2 < tau,
    tau = sig2 (log N - log(-log(1 - P_fa))) x mult, N = 256 measurements, sig2 = complex noise variance per sample."""
    B = len(H); NF = 16 * os_; tau = sig2 * (np.log(256) - np.log(-np.log(1 - pfa))) * mult
    U = np.zeros((B, 0)); V = np.zeros((B, 0)); res = H.copy(); alive = np.ones(B, bool); n = np.zeros(B, int); states = []
    for k in range(Lmax):
        X = np.fft.fft2(res, s=(NF, NF)); P = np.abs(X).reshape(B, -1) ** 2 / 256
        alive &= P.max(1) > tau; n += alive
        if not alive.any():
            break
        idx = P.argmax(1)
        u, v = C.newton(res, -2 * np.pi * (idx // NF) / NF, 2 * np.pi * (idx % NF) / NF)
        U = np.c_[U, u]; V = np.c_[V, v]; U, V = C.cyclic(H, U, V, rc)
        x, Am = C.ls_gains(H, U, V); res = H - np.einsum('bik,bk->bi', Am, x).reshape(H.shape)
        states.append((U.copy(), V.copy()))
    out = []
    for i in range(B):
        if n[i] == 0:
            out.append((np.zeros(0), np.zeros(0)))
        else:
            ps, ph = C.uv_to_angles(states[n[i] - 1][0][i:i + 1], states[n[i] - 1][1][i:i + 1]); out.append((ps[0], ph[0]))
    return out


def sig2_est(H):
    """classical blind noise-variance estimate: median |Y_beam|^2 / ln 2 (sparse beamspace)"""
    Yb = to_beam(H)
    return np.median(np.abs(Yb.reshape(len(H), -1)) ** 2, 1) / np.log(2)


def model_decode(name, d, sl, unknown=False, cfg=None):
    m = net(name); old = dict(m.cfg)
    if cfg:
        m.cfg.update(cfg)
    Y = d['Y'][sl]; bits = d['bits'][sl]; Lv = np.asarray(d['L'])[sl] if not np.isscalar(d['L']) else np.full(len(Y), d['L'])
    snr = d['snr'][sl]
    if unknown:
        out = []
        with torch.no_grad():
            for s in range(0, len(Y), 200):
                bt = prep_batch(Y[s:s + 200], bits[s:s + 200], snr[s:s + 200])
                for U_, V_ in m.decode_unknown(bt, Lmax=8):
                    ps, ph = C.uv_to_angles(U_[None], V_[None]); out.append((ps[0], ph[0]))
        m.cfg.update(old); return out

    def fn(ii, L):
        U, V, _ = decode_np(m, Y[ii], bits[ii], L, snr=snr[ii], bs=200)
        return U, V
    r = by_L(Lv, fn); m.cfg.update(old); return r


def run_method(meth, d, sl):
    """-> list of (psi, phi) for scenes sl"""
    Y = d['Y'][sl]; H = SS.to_ant(Y); bits = d['bits'][sl]; snr = d['snr'][sl]
    Lv = np.asarray(d['L'])[sl] if not np.isscalar(d['L']) else np.full(len(Y), d['L'])
    b0 = int(bits[0]) if len(bits) else 0
    if meth in ('A', 'B'):
        return model_decode(meth, d, sl)
    if meth in ('A unknownL', 'B unknownL'):
        return model_decode(meth[0], d, sl, unknown=True)
    if meth.startswith('A abl:'):
        ab = meth.split(':')[1]
        cfg = {'kappa1': dict(kappa1=True), 'eta0': dict(eta0=True), 'truesig': dict(true_sigma=True), 'stepoff': dict(step_off=True),
               'selnll': dict(sel_nll=True), 'beta0': dict(beta0=True)}[ab]
        return model_decode('A', d, sl, cfg=cfg)
    if meth == 'NOMP':
        return by_L(Lv, lambda ii, L: C.nomp_greedy(H[ii], L, 4, 3))
    if meth == 'NOMP os8 rc6':
        return by_L(Lv, lambda ii, L: C.nomp_greedy(H[ii], L, 8, 6))
    if meth == 'NOMP+SC tuned':
        return by_L(Lv, lambda ii, L: C.selfcal(H[ii], *C.nomp_greedy(H[ii], L, 4, 3), 1.0, 2.0, rc=2, iters=4))
    if meth == 'NOMP os8 rc6+SC tuned':
        return by_L(Lv, lambda ii, L: C.selfcal(H[ii], *C.nomp_greedy(H[ii], L, 8, 6), 1.0, 2.0, rc=2, iters=4))
    if meth == 'classical skeleton':        # ucore classical configuration: H1 + H3..H6 (no network), true sigma, 12 LM layers, lowest NLL
        m = UNOMP('A'); m.cfg.update(classical=True, hyps=['H1', 'H3', 'H4', 'H5', 'H6'], sc=False)
        return by_L(Lv, lambda ii, L: decode_np(m, Y[ii], bits[ii], L, snr=snr[ii], bs=200)[:2])
    if meth == 'classical heat decoder':
        heat, Hc = orig_cnn(d, sl)
        return by_L(Lv, lambda ii, L: tuple(np.pi * np.cos(a) for a in C.decode(Hc[ii], L, heat[ii], 'heat', C.SC)))
    if meth == 'lurd all-neural':
        heat, Hc = orig_cnn(d, sl); ld = net('lurd'); ld.use_sc = True

        def fn(ii, L):
            U0, V0 = heat_peaks(heat[ii], L)
            with torch.no_grad():
                Uu, Vv = ld(torch.from_numpy(Hc[ii].astype(np.complex64)), torch.tensor(U0, dtype=torch.float32), torch.tensor(V0, dtype=torch.float32))
            return Uu.numpy().astype(np.float64), Vv.numpy().astype(np.float64)
        return by_L(Lv, fn)
    if meth == 'consistent NOMP':
        return by_L(Lv, lambda ii, L: A.consistent_nomp(Y[ii], b0, L, 8)[:2])
    if meth == 'FT network only':
        hf, _ = A.cnn_forward(net('ft'), Y)
        return A.as_list(A.parabolic_heat_decode(hf, int(Lv[0])))
    if meth.startswith('MLfam:'):
        raise RuntimeError('ML family is run jointly (run_mlfam)')
    if meth == 'EB-ML-g5':
        sys.path.insert(0, f'{DR}/finetune/imp'); import icom as I
        return by_L(Lv, lambda ii, L: I.ebml_g5(H[ii], L, C.nomp_greedy(H[ii], L, 4, 3)))
    if meth in ('PN-ML all starts', 'POS-MAP all starts'):
        return physics_ml(meth, d, sl)
    if meth.startswith('NOMP stop'):
        mult = STOP_MULT.get(meth, 1.0)
        s2 = 2 * sigma_of(snr) ** 2 if 'true' in meth else sig2_est(H)
        return nomp_stop(H, s2, mult=mult)
    raise KeyError(meth)


STOP_MULT = {}
_S8 = None


def s8():
    global _S8
    if _S8 is None:
        import ast
        path = f'{DR}/finetune/adversarial/s8_physics_ml.py'
        tree = ast.parse(open(path, encoding='utf-8').read())
        keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('pn_eval', 'lm_pn', 'pos_jac', 'pos_eval', 'lm_pos', 'pick3', 'heat_ref')]
        keep += [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') in ('_Bq', 'QR') for t in n.targets)]
        g = dict(np=np, A=A, C=C, MO=MO, WH=MO.WH, FT_=MO.FT, _n=np.arange(16.), L=3, RC=3, time=time)
        exec(compile(ast.Module(body=keep, type_ignores=[]), path, 'exec'), g); _S8 = g
    return _S8


def physics_ml(meth, d, sl):
    """PN-ML (M6) / POS-MAP (M2) 'all starts' exactly as finetune/adversarial/s8_physics_ml.py (known impairment level, true sigma)."""
    g = s8(); Y = d['Y'][sl]; H = SS.to_ant(Y); B = len(Y); y = Y.reshape(B, 256); L = 3
    sn2 = (sigma_of(d['snr'][sl]) ** 2)[:, None]
    st = [C.nomp_greedy(H, L, 4, 3), C.nomp_greedy(H, L, 8, 6), g['heat_ref'](H, A.fft_heat(H, 4)), A.multistart(H, L, 6)[:2]]
    sev = d['sev']
    if meth.startswith('PN'):
        res = [g['lm_pn'](U, V, y, sn2, np.deg2rad(sev)) for U, V in st]
    else:
        res = [g['lm_pos'](U, V, y, sn2, sev, noramp=True) for U, V in st]
    Ub, Vb, _ = g['pick3'](res)
    return A.as_list(C.uv_to_angles(Ub, Vb))


def run_mlfam(d, sl):
    """all quantised-ML variants (mlt.ml_family) + hybrid from FT-network seeds, one chunk -> dict name -> preds"""
    Y = d['Y'][sl]; b0 = int(d['bits'][sl][0]); L = 3
    hf, _ = A.cnn_forward(net('ft'), Y); ps = A.parabolic_heat_decode(hf, L)
    Un = np.array([np.pi * np.cos(p[0][:L]) for p in ps]); Vn = np.array([np.pi * np.cos(p[1][:L]) for p in ps])
    r = mlt.ml_family(Y, b0, d['snr'][sl], L, ft_seeds=(Un, Vn))
    return {k: A.as_list(C.uv_to_angles(u, v)) for k, (u, v) in r.items()}


MLFAM = ['ML', 'annealed ML', 'ML snr+5', 'ML snr-5', 'ML joint sigma', 'ML nominal 10dB', 'hybrid ML from FT seeds']


# ================================================================ unknown-L metric
def match_hits(pred, psi_t, phi_t):
    """pred (psi, phi) arrays (n_est,), truth (L,) -> number of paths with both |errors| <= 1 deg under the min-total-error matching"""
    from scipy.optimize import linear_sum_assignment
    ne = len(pred[0]); nt = len(psi_t)
    if ne == 0 or nt == 0:
        return 0
    cost = np.abs(np.rad2deg(pred[0])[:, None] - np.rad2deg(psi_t)[None]) + np.abs(np.rad2deg(pred[1])[:, None] - np.rad2deg(phi_t)[None])
    r, c = linear_sum_assignment(cost)
    return int(((np.abs(np.rad2deg(pred[0][r] - psi_t[c])) <= 1) & (np.abs(np.rad2deg(pred[1][r] - phi_t[c])) <= 1)).sum())


def f1_stats(preds, d):
    out = np.zeros((len(preds), 3))
    for i, p in enumerate(preds):
        L = int(np.asarray(d['L'])[i]) if not np.isscalar(d['L']) else d['L']
        out[i] = [match_hits(p, d['psi'][i, :L], d['phi'][i, :L]), L, len(p[0])]
    return out    # hits, n_true, n_est


# ================================================================ driver
def stats_path(bank, meth):
    bank = bank.replace(':', '-')
    return f'{STATS}/{bank}__{meth.replace(" ", "_").replace(":", "-").replace("/", "-")}.npz'


def ensure(bank, meths, d=None, unknown_metric=False):
    todo = [m for m in meths if not os.path.exists(stats_path(bank, m))]
    if not todo:
        return
    d = d or load(bank); N = len(d['snr']); t00 = time.time()
    for m in todo:
        if m in MLFAM:
            continue
        t0 = time.time(); P = []
        for c0 in range(0, N, CH):
            P += run_method(m, d, slice(c0, c0 + CH))
        S = stats(P, d); extra = {}
        if unknown_metric:
            extra['f1'] = f1_stats(P, d)
        n = np.array([len(p[0]) for p in P]); Lm = max(1, n.max())
        ps = np.full((N, Lm), np.nan); ph = ps.copy()
        for i, p in enumerate(P):
            ps[i, :n[i]] = p[0]; ph[i, :n[i]] = p[1]
        np.savez(stats_path(bank, m), S=S, psi=ps, phi=ph, n=n, sec=time.time() - t0, **extra)
        print(f'  {bank}: {m} Pd {pd(S, d["snr"]):.4f} ({time.time() - t0:.0f}s)', flush=True)
    fam = [m for m in todo if m in MLFAM]
    if fam:
        t0 = time.time(); P = {k: [] for k in MLFAM}
        for c0 in range(0, N, 200):
            cf = f'{STATS}/_mlfam_{bank}_{c0}.npz'
            if os.path.exists(cf):
                z = np.load(cf, allow_pickle=True); r = z['r'].item()
            else:
                r = run_mlfam(d, slice(c0, c0 + 200)); np.savez(cf, r=np.array(r, dtype=object))
                print(f'  {bank}: ML family chunk {c0} [{time.time() - t0:.0f}s]', flush=True)
            for k in MLFAM:
                P[k] += r[k]
        for m in MLFAM:
            S = stats(P[m], d); n = np.array([len(p[0]) for p in P[m]])
            ps = np.array([p[0] for p in P[m]]); ph = np.array([p[1] for p in P[m]])
            np.savez(stats_path(bank, m), S=S, psi=ps, phi=ph, n=n, sec=(time.time() - t0) / len(MLFAM))
            print(f'  {bank}: {m} Pd {pd(S, d["snr"]):.4f}', flush=True)
        for c0 in range(0, N, 200):
            try:
                os.remove(f'{STATS}/_mlfam_{bank}_{c0}.npz')
            except OSError:
                pass
    print(f'{bank}: done {todo} [{time.time() - t00:.0f}s]', flush=True)


def load_stats(bank, meth):
    z = np.load(stats_path(bank, meth)); return {k: z[k] for k in z.files}


def cached_pred_stats(bank, meth):
    """U-Net / NOMP / lurd cached predictions on the cached banks"""
    d = C.load_bank(bank)
    if meth == 'U-Net (Lloria et al.)':
        p = C.load_preds(f'{C.CACHE}/pred__U-Net_paper__{bank}.npz')
    elif meth == 'lurd all-neural (cached)':
        z = np.load(f'C:/Users/Milon/iabr_deep_research_cache/dr_cache/nn_decoder/{bank}__allneural_sc.npz')
        p = [(z['psi'][i, :z['n'][i]], z['phi'][i, :z['n'][i]]) for i in range(len(z['n']))]
    return C.scene_stats(p, d)


BASE_ALL = ['NOMP', 'NOMP os8 rc6', 'NOMP+SC tuned', 'classical skeleton']
GROUPS = {
    'official': [('official', BASE_ALL + ['classical heat decoder']), ('rep9520', BASE_ALL + ['classical heat decoder', 'lurd all-neural']),
                 ('T2_d5', ['NOMP', 'NOMP os8 rc6'])],
    'regime': [(f'M5_b{b}_snr{s}', BASE_ALL + ['consistent NOMP', 'FT network only', 'classical heat decoder', 'lurd all-neural'] + MLFAM) for (b, s) in ADC_CELLS] +
              [(c, BASE_ALL + ['NOMP os8 rc6+SC tuned', 'classical heat decoder', 'lurd all-neural'] + (['EB-ML-g5'] if c.startswith(('M3', 'M1')) else []) +
                (['PN-ML all starts'] if c.startswith('M6') else []) + (['POS-MAP all starts'] if c.startswith('M2') else [])) for c in IMP_CELLS],
    'unknownL': [('Lmix9550', ['NOMP stop true sigma', 'NOMP stop est sigma'])] +
                [(f'L{l}_snr{s}', ['NOMP stop true sigma', 'NOMP stop est sigma']) for l in [1, 2, 4, 5, 6] for s in [0, 15]],
    'prior': [(f'sep_{k}deg_snr15', BASE_ALL) for k in [1, 2, 3, 5, 10, 20]] + [('nosep9560', BASE_ALL), ('sep10_9561', BASE_ALL)] +
             [(f'L{l}_snr{s}', BASE_ALL) for l in [1, 2, 4, 5, 6] for s in [0, 15]],
    'ood': [('M8_0.5_snr15', BASE_ALL), ('M4_2_snr15', BASE_ALL), ('M5c_b2_snr15', BASE_ALL + ['consistent NOMP', 'FT network only'])] +
           [(f'sweep_b{b}_snr{s}', ['NOMP', 'consistent NOMP', 'FT network only', 'classical skeleton']) for b in [1, 2] for s in [-5, 5, 10, 25]],
    'stress': [(f'stressM5_b{b}_snr{s}', ['NOMP', 'NOMP os8 rc6', 'consistent NOMP', 'FT network only', 'classical skeleton']) for b in [1, 2, 3, 4] for s in [0, 15]],
}
GROUPS['ablate'] = [('official_sub', []), *[(f'sub500:{c}', []) for c in ['M5_b1_snr15', 'M5_b2_snr15', 'M5_b1_snr0', 'M5_b2_snr0', 'M2_0.2_snr15', 'M6_30_snr15']]]
MODELS = ['B', 'A']


def with_sev(bank, d):
    if bank in IMP_CELLS:
        d['sev'] = IMP_CELLS[bank][1]
    return d


if __name__ == '__main__':
    for grp in sys.argv[1:]:
        models = os.environ.get('MODELS', '')
        for bank, meths in GROUPS[grp.replace('+models', '')]:
            ms = list(meths)
            if grp.endswith('+models'):
                ms = [m for m in models.split(',') if m]
                if bank.startswith('L') and grp.startswith('unknownL'):
                    ms = [f'{m} unknownL' for m in ms]
            unk = grp.startswith('unknownL')
            todo = [m for m in ms if not os.path.exists(stats_path(bank, m))]
            if not todo:
                continue
            d = with_sev(bank, load(bank))
            if unk and any(m.startswith('NOMP stop') for m in todo):
                for m in todo:
                    if m.startswith('NOMP stop'):
                        STOP_MULT[m] = json.load(open(f'{HERE}/stop_tune.json'))[m]
            ensure(bank, todo, d, unknown_metric=unk)
