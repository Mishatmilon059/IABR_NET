"""ADC bit-depth test (one bit depth per process). usage: python adc_bits.py <bits>
Fresh scenes seeds 14000 (0 dB) / 14010 (10 dB) / 14020 (20 dB), L=3, peak-AGC uniform ADC, ideal hardware, N scenes per cell
(same scenes for every method and every bit depth), chunks of <= 50, resumable per chunk. Threads: 2 (env THREADS=2)."""
import os, sys
os.environ['THREADS'] = '2'
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '2'
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/deep_research/unfolded_v2/build')
import mlt
from mlt import *   # np, torch, C, SS, A, mlq, sigma_of, lm_t, ml_multi, starts_classical, lm_joint_sigma, init_gains_t
torch.set_num_threads(2)
from threadpoolctl import threadpool_limits
OUT = 'D:/ai_ml_project/paper/evidence/adc_bits_test'
BITS = int(sys.argv[1]); N = int(os.environ.get('NSC', 150)); CH = 50; L = 3
SFX = '' if N == 150 else f'_n{N}'
CACHE = f'{OUT}/cells{SFX}'; os.makedirs(CACHE, exist_ok=True)
SEEDS = {0: 14000, 10: 14010, 20: 14020}
METHODS = ['NOMP', 'consistent NOMP', 'CNN read-out', 'ML known sigma', 'ML snr+5', 'ML snr-5', 'ML nominal 10dB', 'ML joint sigma',
           'hybrid known sigma', 'hybrid snr+5', 'hybrid snr-5', 'hybrid nominal 10dB', 'hybrid joint sigma']
ft = A.load_cnn(A.FT_CKPT)


def uv_from_ang(pl):
    U = np.zeros((len(pl), L)); V = np.zeros((len(pl), L))
    for i, p in enumerate(pl):
        a = np.pi * np.cos(np.asarray(p[0])[:L]); b = np.pi * np.cos(np.asarray(p[1])[:L])
        U[i, :len(a)] = a; V[i, :len(b)] = b
    return U, V


def joint_single(Yq, lo, hi, U0, V0):
    rms = np.sqrt((np.abs(Yq) ** 2).mean((1, 2)) / 2)[:, None]; s0 = torch.from_numpy(0.1 * rms)
    Ut, Vt = torch.from_numpy(U0.copy()), torch.from_numpy(V0.copy())
    lo_t, hi_t = torch.from_numpy(lo), torch.from_numpy(hi)
    X = init_gains_t(Ut, Vt, Yq, lo_t, hi_t, s0)
    Uo, Vo, Xo, F, sg = lm_joint_sigma(Ut, Vt, X, lo_t, hi_t, s0)
    return Uo.numpy(), Vo.numpy()


def run_chunk(base, Y, bits):
    P = {}; T = {}

    def tick(k, t0):
        T[k] = T.get(k, 0) + time.time() - t0
    H = A.to_ant(Y)
    t0 = time.time(); P['NOMP'] = C.uv_to_angles(*C.nomp_greedy(H, L, 4, 3)); tick('NOMP', t0)
    t0 = time.time(); Uc, Vc, _ = A.consistent_nomp(Y, bits, L, 8); P['consistent NOMP'] = C.uv_to_angles(Uc, Vc); tick('consistent NOMP', t0)
    t0 = time.time(); hf, _ = A.cnn_forward(ft, Y); pl = A.parabolic_heat_decode(hf, L); P['CNN read-out'] = pl; tick('CNN read-out', t0)
    Un, Vn = uv_from_ang(pl)
    lo, hi = mlq.cells_ri(Y, bits, A.adc_cells); B = len(Y); one = np.ones((B, 1))
    s_true = sigma_of(base['snr'])[:, None] * one
    t0 = time.time(); st, em = starts_classical(Y, bits, L, s_true); tick('ML starts (13 + EM)', t0)
    sg = {'known sigma': s_true, 'snr+5': sigma_of(base['snr'] + 5.0)[:, None] * one, 'snr-5': sigma_of(base['snr'] - 5.0)[:, None] * one,
          'nominal 10dB': sigma_of(np.full(B, 10.0))[:, None] * one}
    for nm, s in sg.items():
        t0 = time.time()
        U, V, _, _ = ml_multi(Y, lo, hi, s, st + ([em] if nm == 'known sigma' else []))
        P['ML ' + nm] = C.uv_to_angles(U, V); tick('ML ' + nm + ' (excl. starts)', t0)
        t0 = time.time()
        U, V, _, _ = ml_multi(Y, lo, hi, s, [(Un, Vn)]); P['hybrid ' + nm] = C.uv_to_angles(U, V); tick('hybrid ' + nm + ' (excl. CNN)', t0)
    t0 = time.time()
    rms = np.sqrt((np.abs(Y) ** 2).mean((1, 2)) / 2)[:, None]; s0 = 0.1 * rms; S = len(st)
    Uc_ = torch.from_numpy(np.concatenate([u for u, _ in st])); Vc_ = torch.from_numpy(np.concatenate([v for _, v in st]))
    lo_t = torch.from_numpy(np.concatenate([lo] * S)); hi_t = torch.from_numpy(np.concatenate([hi] * S)); s0t = torch.from_numpy(np.concatenate([s0] * S))
    X = init_gains_t(Uc_, Vc_, np.concatenate([Y] * S), lo_t, hi_t, s0t)
    Uo, Vo, Xo, F, _ = lm_joint_sigma(Uc_, Vc_, X, lo_t, hi_t, s0t)
    F = F.reshape(S, B); i = F.argmin(0); ar = torch.arange(B)
    P['ML joint sigma'] = C.uv_to_angles(Uo.reshape(S, B, -1)[i, ar].numpy(), Vo.reshape(S, B, -1)[i, ar].numpy()); tick('ML joint sigma (excl. starts)', t0)
    t0 = time.time(); U, V = joint_single(Y, lo, hi, Un, Vn); P['hybrid joint sigma'] = C.uv_to_angles(U, V); tick('hybrid joint sigma (excl. CNN)', t0)
    return {m: A.stats(P[m], base) for m in METHODS}, T


def sub(b, sl):
    return {k: (v[sl] if isinstance(v, np.ndarray) and v.ndim and len(v) == N else v) for k, v in b.items()}


t00 = time.time()
for snr, seed in SEEDS.items():
    cell = f'{CACHE}/b{BITS}_snr{snr}.npz'
    if os.path.exists(cell):
        print(f'b{BITS} snr{snr}: done already', flush=True); continue
    base = SS.base_scenes(N, L, float(snr), seed); Yall = SS.generate(base, 'M5', BITS)
    Ss = {m: [] for m in METHODS}; Tt = {}
    for c0 in range(0, N, CH):
        f = f'{CACHE}/b{BITS}_snr{snr}_c{c0}.npz'
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True); Sc = {m: z[m] for m in METHODS}; Tc = z['T'].item()
        else:
            sl = slice(c0, min(c0 + CH, N)); t0 = time.time()
            Sc, Tc = run_chunk(sub(base, sl), Yall[sl], BITS)
            np.savez(f, T=np.array(Tc, dtype=object), **Sc)
            print(f'b{BITS} snr{snr} chunk {c0}: {time.time() - t0:.0f}s (total {time.time() - t00:.0f}s)', flush=True)
        for m in METHODS:
            Ss[m].append(Sc[m])
        for k, v in Tc.items():
            Tt[k] = Tt.get(k, 0) + v
    Sm = {m: np.concatenate(v) for m, v in Ss.items()}
    np.savez(cell, T=np.array({k: v / N for k, v in Tt.items()}, dtype=object), **Sm)
    print(f'=== b{BITS} snr{snr} (N={N}) Pd paper: ' + ', '.join(f'{m} {C.metric_from_stats(Sm[m], np.arange(N))["pd_paper"]:.3f}' for m in METHODS), flush=True)
    del Sm, Ss, Yall, base
print(f'b{BITS} cells done in {time.time() - t00:.0f}s', flush=True)

# ---- latency, 1 thread, 12 scenes, SNR 10 dB (separate scenes seed 14900), seconds per scene
lat = os.path.join(OUT, f'latency_b{BITS}{SFX}.json')
if not os.path.exists(lat):
    nl = 12
    b = SS.base_scenes(nl, L, 10.0, 14900); Y = SS.generate(b, 'M5', BITS)
    torch.set_num_threads(1); L_ = {}
    with threadpool_limits(limits=1):
        H = A.to_ant(Y); lo, hi = mlq.cells_ri(Y, BITS, A.adc_cells); one = np.ones((nl, 1)); s = sigma_of(b['snr'])[:, None] * one

        def tm(k, fn):
            t0 = time.time(); r = fn(); L_[k] = (time.time() - t0) / nl; return r
        tm('NOMP', lambda: C.nomp_greedy(H, L, 4, 3)); tm('consistent NOMP', lambda: A.consistent_nomp(Y, BITS, L, 8))
        hf = tm('CNN read-out', lambda: A.parabolic_heat_decode(A.cnn_forward(ft, Y)[0], L)); Un, Vn = uv_from_ang(hf)
        tm('ML starts only (13, no EM)', lambda: starts_classical(Y, BITS, L, None))
        st, em = tm('ML starts only (13 + EM)', lambda: starts_classical(Y, BITS, L, s))
        tm('ML known sigma LM only', lambda: ml_multi(Y, lo, hi, s, st + [em]))
        tm('ML wrong/fixed sigma LM only (13 starts)', lambda: ml_multi(Y, lo, hi, s, st))
        tm('hybrid LM only (CNN seeds)', lambda: ml_multi(Y, lo, hi, s, [(Un, Vn)]))
        L_['ML known sigma TOTAL'] = L_['ML starts only (13 + EM)'] + L_['ML known sigma LM only']
        L_['ML wrong/fixed sigma TOTAL'] = L_['ML starts only (13, no EM)'] + L_['ML wrong/fixed sigma LM only (13 starts)']
        L_['hybrid TOTAL (CNN + one LM)'] = L_['CNN read-out'] + L_['hybrid LM only (CNN seeds)']
    json.dump(L_, open(lat, 'w'), indent=1); print('latency (s/scene, 1 thread, 12 scenes):', L_, flush=True)
print('ALL DONE', flush=True)
