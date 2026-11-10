"""Ideal-hardware regression claims of the three fine-tuning stages:
  (i)  FRESH official-like bank: stress_sim.base_scenes(250, L=3, SNR, 8780+k) + generate('clean') for SNR = -10, -5, ..., 25 dB (k = 0..7),
       2000 scenes (same L, SNR grid and ideal hardware as the frozen official bank; new seeds);
  (ii) reproduction on the frozen official bank itself (8000 scenes) of the same pipelines.
Pipelines: NOMP; NOMP os8 rc6; NOMP+SC tuned; orig dual+SC (= IABR-v2 dual); ADC-FT dual+SC; ADC-FT network only;
  imp-FT dual+SC (own IABC); imp-FT seeds on Ht+SC (the 'learned seeds' heat-only path); imp-FT nocorr dual+SC; imp-FT v2b dual+SC;
  ADC-FT CNN + adapted learned decoder; ADC-FT CNN + original learned decoder (lurd.pt); orig CNN + original learned decoder.
Paired bootstrap 2000 within SNR groups, seed 0.  usage: python s5_official.py  (stdout -> s5_official.log; writes s5_official.json)"""
from advc import *
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/finetune/imp')
import icom as I                        # noqa: E402
import iabr2_sim as SIM                 # noqa: E402
LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
sys.path.insert(0, LD)
from exp5 import StagedSC, RefinerV3, heat_peaks_np, uv_to_angles_np      # noqa: E402
torch.set_num_threads(6)

L, RC, CH = 3, 3, 400
SMOKE = int(os.environ.get('SMOKE', 0))
SFX = '_smoke' if SMOKE else ''
NPER = SMOKE if SMOKE else 250
orig = A.load_cnn(A.ORIG_CKPT); adc = A.load_cnn(A.FT_CKPT); imp = I.load_cnn(I.FT_CKPT)


def mk_ld(p):
    m = StagedSC(RefinerV3(), pre=6, rounds=3, inner=2); m.load_state_dict(torch.load(p, map_location='cpu')); m.eval(); m.use_sc = True
    return m


ld_a = mk_ld('D:/ai_ml_project/paper/evidence/deep_research/nn_decoder_adapt/cache/ld_adc_final_run.pt'); ld_o = mk_ld(f'{LD}/lurd.pt')
METH = ['NOMP', 'NOMP os8 rc6', 'NOMP+SC tuned', 'orig dual+SC', 'ADC-FT dual+SC', 'ADC-FT network only', 'imp-FT dual+SC',
        'imp-FT seeds on Ht+SC', 'imp-FT nocorr dual+SC', 'imp-FT v2b dual+SC', 'ADC-FT + adapted LD', 'ADC-FT + original LD', 'orig + original LD']


def heat_ref(H, heat):
    Uh, Vh = C.g['heat_peaks'](heat, L)
    Uh, Vh = C.newton(H[:, None].repeat(L, 1).reshape(-1, 16, 16), Uh.ravel(), Vh.ravel())
    return C.cyclic(H, Uh.reshape(-1, L), Vh.reshape(-1, L), RC)


def dual(H, h, g):
    pick = C.residual_energy(H, *h) < C.residual_energy(H, *g)
    return np.where(pick[:, None], h[0], g[0]), np.where(pick[:, None], h[1], g[1])


def scd(H, UV):
    return C.uv_to_angles(*C.selfcal(H, UV[0], UV[1], *A.SC))


def run_ld(m, heat, Hc):
    with torch.no_grad():
        Uh, Vh = heat_peaks_np(heat, L)
        U, V = m(torch.from_numpy(Hc.astype(np.complex64)), torch.tensor(Uh, dtype=torch.float32), torch.tensor(Vh, dtype=torch.float32))
    return uv_to_angles_np(U.numpy().astype(np.float64), V.numpy().astype(np.float64))


def run(Y):
    H = SS.to_ant(Y); P = {}
    g = C.nomp_greedy(H, L, 4, RC)
    P['NOMP'] = C.uv_to_angles(*g); P['NOMP os8 rc6'] = C.uv_to_angles(*C.nomp_greedy(H, L, 8, 6)); P['NOMP+SC tuned'] = C.uv_to_angles(*A.selfcal_tuned(H, *g))
    ho, Hco = A.cnn_forward(orig, Y); P['orig dual+SC'] = C.decode(Hco, L, ho, 'dual', A.SC)
    ha, Hca = A.cnn_forward(adc, Y); P['ADC-FT dual+SC'] = C.decode(Hca, L, ha, 'dual', A.SC); P['ADC-FT network only'] = A.parabolic_heat_decode(ha, L)
    hi_, Hci, _ = I.net_forward(imp, Y)
    P['imp-FT dual+SC'] = scd(Hci, dual(Hci, heat_ref(Hci, hi_), C.nomp_greedy(Hci, L, 4, RC)))
    P['imp-FT seeds on Ht+SC'] = scd(H, heat_ref(H, hi_))
    hn = I.trunk_heat(imp, Y); P['imp-FT nocorr dual+SC'] = scd(H, dual(H, heat_ref(H, hn), g))
    al = I.AL(); Hv = al.apply_corr(H, I.v2b_est(H, L)); gv = C.nomp_greedy(Hv, L, 4, RC)
    hv = I.trunk_heat(imp, I.to_beam(Hv)); P['imp-FT v2b dual+SC'] = scd(Hv, dual(Hv, heat_ref(Hv, hv), gv))
    P['ADC-FT + adapted LD'] = run_ld(ld_a, ha, Hca); P['ADC-FT + original LD'] = run_ld(ld_o, ha, Hca); P['orig + original LD'] = run_ld(ld_o, ho, Hco)
    return P


PAIRS = [('ADC-FT dual+SC', 'orig dual+SC'), ('ADC-FT dual+SC', 'NOMP'), ('ADC-FT network only', 'NOMP'), ('imp-FT dual+SC', 'NOMP'),
         ('imp-FT dual+SC', 'orig dual+SC'), ('imp-FT seeds on Ht+SC', 'NOMP'), ('imp-FT nocorr dual+SC', 'NOMP'), ('imp-FT v2b dual+SC', 'NOMP'),
         ('ADC-FT + adapted LD', 'NOMP'), ('ADC-FT + adapted LD', 'ADC-FT + original LD'), ('ADC-FT + adapted LD', 'NOMP os8 rc6'),
         ('orig + original LD', 'NOMP'), ('orig dual+SC', 'NOMP'), ('NOMP os8 rc6', 'NOMP')]


def banks():
    psi, phi, Lr, snr, Ys, alpha = [], [], [], [], [], []
    for k, s in enumerate(range(-10, 30, 5)):
        b = SS.base_scenes(NPER, 3, float(s), OFF_SEED0 + k); Ys.append(SS.generate(b, 'clean'))
        psi.append(b['psi']); phi.append(b['phi']); Lr.append(b['L']); snr.append(b['snr'])
    yield 'fresh official-like (2000)', dict(psi=np.concatenate(psi), phi=np.concatenate(phi), L=np.concatenate(Lr), snr=np.concatenate(snr).astype(np.float32)), np.concatenate(Ys)
    z = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
    Y16 = SIM.downsample16(z['data']); Y = Y16[..., 0].astype(np.float64) + 1j * Y16[..., 1].astype(np.float64)
    b = dict(psi=z['feat'][:, 0, :].astype(np.float64), phi=z['feat'][:, 1, :].astype(np.float64), L=z['meta'][:, 0].astype(int), snr=z['meta'][:, 1].astype(np.float32))
    del z, Y16
    if SMOKE:
        idx = np.concatenate([np.where(b['snr'] == s)[0][:SMOKE] for s in range(-10, 30, 5)]); b = {k: v[idx] for k, v in b.items()}; Y = Y[idx]
    yield 'official (8000, reproduction)', b, Y


out = {}
for name, b, Y in banks():
    t0 = time.time(); P = {}
    for s0 in range(0, len(Y), CH):
        Pc = run(Y[s0:s0 + CH])
        for k, v in Pc.items():
            P.setdefault(k, []).extend(A.as_list(v))
    S = {m: A.stats(P[m], b) for m in METH}
    np.savez_compressed(f'{PCACHE}/s5_stats_{name.split()[0]}{SFX}.npz', snr=b['snr'], **S)
    pdm, ci_, D = boot(S, b['snr'], PAIRS)
    out[name] = dict(pd=pdm, ci=ci_, diffs=D, sec=time.time() - t0)
    print(f'=== {name} [{time.time() - t0:.0f}s]', flush=True)
    for m in METH:
        print(f'   {m:24s} Pd {pdm[m]:.4f} [{ci_[m][0]:.4f},{ci_[m][1]:.4f}]', flush=True)
    for k, d in D.items():
        print(f'   {k:50s} {fmt(d)}', flush=True)
    json.dump(out, open(f'{HERE}/s5_official{SFX}.json', 'w'), indent=1, default=float)
print('done', flush=True)
