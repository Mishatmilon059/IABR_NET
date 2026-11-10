"""Agent-free background job (queued 2026-10-15 at the owner's request, so CPU work continues even if the assistant's usage limit is reached).

Adapts the learned unfolded decoder (StagedSC(RefinerV3), learned_decoder/lurd.pt, 4,803 params) to low-resolution-ADC measurements,
using the CNN that the phase-2 workflow fine-tunes on ADC data (deep_research/finetune/adc/cache/adc_ft.pt).
 1. waits until that checkpoint exists, has not changed for 10 minutes and no train_ft.py process is running;
 2. fine-tunes the learned decoder from lurd.pt on fresh synthetic batches (50% nominal, 50% ADC b ~ U{1,2,3,4}, peak AGC), heat seeds from
    the fine-tuned CNN; training seeds 61_000_000+ (disjoint from every test/tuning seed used so far);
 3. evaluates on the stress M5 banks (seeds 5000 / 5015, 500 scenes, b = 4,3,2,1 and clean) and a 2000-scene official subset:
    NOMP, fine-tuned CNN + classical heat-seeded decoder, fine-tuned CNN + original learned decoder, fine-tuned CNN + adapted learned decoder,
    original CNN + original learned decoder; paper-style / strict Pd, RMSE, paired bootstrap (2000, seed 0).
Keeps the PC awake (SetThreadExecutionState) while it runs. Writes only into this folder. Usage: python ld_adapt_adc.py [--smoke]"""
import os, sys, time, json, math, ctypes
SMOKE = '--smoke' in sys.argv
NTH = '2' if SMOKE else '6'
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = NTH
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
HERE = os.path.dirname(os.path.abspath(__file__))
DR = 'D:/ai_ml_project/paper/evidence/deep_research'
LD = 'D:/ai_ml_project/paper/evidence/learned_decoder'
for p in (LD, f'{DR}/stress', 'D:/ai_ml_project/paper/evidence/new_results'):
    sys.path.insert(0, p)
CACHE = f'{HERE}/cache'; os.makedirs(CACHE, exist_ok=True)
# per-scene stats (for paired comparisons with the ADC stage) go to C: (D: is almost full); regenerable from the fixed seeds
PC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder_adapt'
os.makedirs(PC, exist_ok=True)
TAG = 'smoke' if SMOKE else 'run'
LOG = open(f'{HERE}/ld_adapt_adc_{TAG}.log', 'a', encoding='utf-8')


def log(*a):
    s = time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + '\n'); LOG.flush()


try:   # keep the system awake while this process lives (ES_CONTINUOUS | ES_SYSTEM_REQUIRED); released automatically on exit
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
except Exception as e:
    log('keep-awake not set:', e)

import numpy as np, psutil, torch
from exp5 import *            # StagedSC, RefinerV3, Staged, matched_loss, heat_peaks_np, uv_to_angles_np, sim (iabr2_sim), gm (model ns), ...
import common as C
import stress_sim as SS
torch.set_num_threads(int(NTH))

CKPT_FT = f'{DR}/finetune/adc/cache/adc_ft.pt' if not SMOKE else f'{C.ROOT}/outputs/checkpoints/IABRv2/final.pt'
STEPS = 4 if SMOKE else 1500
BS = 16 if SMOKE else 64
N_EVAL = 20 if SMOKE else 500
N_OFF = 5 if SMOKE else 250        # per SNR on the official bank
log(f'start smoke={SMOKE} ckpt={CKPT_FT} steps={STEPS} bs={BS} threads={NTH}')


# ---------------------------------------------------------------- 1. wait for the fine-tuned CNN
def trainer_running():
    for p in psutil.process_iter(['cmdline']):
        try:
            if any('train_ft' in c for c in (p.info['cmdline'] or [])):
                return True
        except Exception:
            pass
    return False


while not SMOKE:
    ok = os.path.exists(CKPT_FT) and time.time() - os.path.getmtime(CKPT_FT) > 600 and not trainer_running()
    if ok:
        break
    time.sleep(120)
log('checkpoint ready:', CKPT_FT, time.ctime(os.path.getmtime(CKPT_FT)))


def load_cnn(path):
    m = gm['IABRv2'](32, 10, 32)
    s = torch.load(path, map_location='cpu')
    if isinstance(s, dict) and 'model' in s and isinstance(s['model'], dict):
        s = s['model']
    m.load_state_dict(s); return m.eval()


cnn_ft = load_cnn(CKPT_FT)
cnn_orig = load_cnn(f'{C.ROOT}/outputs/checkpoints/IABRv2/final.pt')


def forward(model, Y, bs=250):
    X = np.ascontiguousarray(sim.to_ri(Y).transpose(0, 3, 1, 2)); heat, Hc = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            o = model(torch.from_numpy(X[s:s + bs]))
            heat.append(torch.sigmoid(o['heat']).numpy().astype(np.float64)); Hc.append(o['Hc'].numpy().astype(np.complex128))
    return np.concatenate(heat), np.concatenate(Hc)


# ---------------------------------------------------------------- 2. adapt the learned decoder
def gen_batch_adc(rng, B):
    snr = rng.integers(-15, 25, B).astype(float); clean = rng.uniform(size=B) < 0.2
    pd = np.where(clean, 0.0, rng.uniform(0, 8, B)); gd = np.where(clean, 0.0, rng.uniform(0, 3, B))
    psi, phi, alpha = sim.sample_paths(np.full(B, 3), rng, lmax=3)
    Y = sim.simulate(psi, phi, alpha, snr, rng, phase_deg_max=pd, gain_db_max=gd)['Y']
    q = rng.uniform(size=B) < 0.5; bits = rng.integers(1, 5, B)
    for b in (1, 2, 3, 4):
        m = q & (bits == b)
        if m.any():
            Y[m] = SS.adc(Y[m], b, 'peak')
    y = torch.from_numpy(np.ascontiguousarray(sim.to_ri(Y).transpose(0, 3, 1, 2)))
    with torch.no_grad():
        o = cnn_ft(y)
    heat = torch.sigmoid(o['heat']).numpy(); U0, V0 = heat_peaks_np(heat, 3)
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)
    return o['Hc'].detach().to(torch.complex64), t(U0), t(V0), t(np.pi * np.cos(psi)), t(np.pi * np.cos(phi))


def new_ld():
    m = StagedSC(RefinerV3(), pre=6, rounds=3, inner=2)
    m.load_state_dict(torch.load(f'{LD}/lurd.pt', map_location='cpu')); return m


ld_orig = new_ld().eval()
ld = new_ld()
CK = f'{CACHE}/ld_adc_{TAG}.pt'
start, hist = 0, []
if os.path.exists(CK):
    s = torch.load(CK, map_location='cpu'); ld.load_state_dict(s['model']); start, hist = s['step'], s['hist']
    log('resumed at step', start)
opt = torch.optim.Adam(ld.parameters(), 1e-3)
if os.path.exists(CK):
    opt.load_state_dict(torch.load(CK, map_location='cpu')['opt'])
rng = np.random.default_rng([61_000_000, start]); t0 = time.time()
for s in range(start, STEPS):
    ld.train(); ld.use_sc = True
    H, U0, V0, ug, vg = gen_batch_adc(rng, BS)
    for gp in opt.param_groups:
        gp['lr'] = 1e-3 * (0.5 * (1 + math.cos(math.pi * s / STEPS)) * 0.95 + 0.05)
    outs = ld.forward_all(H, U0, V0)
    loss = sum((1.0 if i == len(outs) - 1 else 0.4) * matched_loss(U, V, ug, vg) for i, (U, V) in enumerate(outs))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ld.parameters(), 1.0); opt.step(); hist.append(float(loss))
    if (s + 1) % 50 == 0 or s + 1 == STEPS:
        log(f'step {s + 1}/{STEPS} loss(avg last 50) {np.mean(hist[-50:]):.4f} lam {float(ld.lam):.3f} {time.time() - t0:.0f}s')
        torch.save(dict(model=ld.state_dict(), opt=opt.state_dict(), step=s + 1, hist=hist), CK)
torch.save(ld.state_dict(), f'{CACHE}/ld_adc_final_{TAG}.pt')
ld.eval(); ld.use_sc = True; ld_orig.use_sc = True


# ---------------------------------------------------------------- 3. evaluate
def run_ld(model, heat, Hc):
    U0, V0 = heat_peaks_np(heat, 3); Us, Vs = [], []
    with torch.no_grad():
        for i in range(0, len(U0), 200):
            H = torch.from_numpy(Hc[i:i + 200].astype(np.complex64))
            U, V = model(H, torch.tensor(U0[i:i + 200], dtype=torch.float32), torch.tensor(V0[i:i + 200], dtype=torch.float32))
            Us.append(U.numpy()); Vs.append(V.numpy())
    return uv_to_angles_np(np.concatenate(Us).astype(np.float64), np.concatenate(Vs).astype(np.float64))


def lst(p):
    return [(p[0][i], p[1][i]) for i in range(len(p[0]))]


def predict_chunk(Y):
    hf, Hf = forward(cnn_ft, Y); ho, Ho = forward(cnn_orig, Y)
    U, V = C.nomp_greedy(SS.to_ant(Y), 3, C.CFG['nomp_os'], C.CFG['nomp_rc'])
    return {'NOMP': lst(C.uv_to_angles(U, V)),
            'FT-CNN + classical heat decoder': lst(C.decode(Hf, 3, hf, 'heat', C.SC)),
            'FT-CNN + original learned decoder': lst(run_ld(ld_orig, hf, Hf)),
            'FT-CNN + adapted learned decoder': lst(run_ld(ld, hf, Hf)),
            'original CNN + original learned decoder': lst(run_ld(ld_orig, ho, Ho))}


def evaluate(Y, psi, phi, snr, name):
    b = dict(psi=psi, phi=phi, L=np.full(len(psi), 3))
    preds = {}
    for s0 in range(0, len(Y), 400):          # memory rule: NOMP / decoders on at most 400 scenes at a time
        for k, v in predict_chunk(Y[s0:s0 + 400]).items():
            preds.setdefault(k, []).extend(v)
    S = {k: C.scene_stats(p, b) for k, p in preds.items()}
    np.savez_compressed(f'{PC}/stats_{TAG}__{name}.npz', snr=np.asarray(snr), **S)
    Bt = C.boot_metrics(S, snr, B=2000, seed=0)
    res = {'methods': {}, 'diffs': {}}
    for k in S:
        per = C.metric_by_snr(S[k], snr); res['methods'][k] = per['mean']
    ref = 'FT-CNN + adapted learned decoder'
    for k in S:
        if k != ref:
            d = Bt[ref] - Bt[k]
            res['diffs'][f'{ref} - {k}'] = {kk: [res['methods'][ref][kk] - res['methods'][k][kk], *C.ci(d[:, j])[1:]] for j, kk in enumerate(C.KEYS)}
    return res


out = dict(ckpt_ft=CKPT_FT, steps=STEPS, batch=BS, train_seed=61_000_000, banks={})
for snr, seed in ((15, 5015), (0, 5000)):
    base = SS.base_scenes(N_EVAL, 3, float(snr), seed)
    for kind, sev in [('clean', None), ('M5', 4), ('M5', 3), ('M5', 2), ('M5', 1)]:
        Y = SS.generate(base, kind, sev); name = f'{kind}{"" if sev is None else "_b%d" % sev}_snr{snr}'
        r = evaluate(Y, base['psi'][:, :3], base['phi'][:, :3], base['snr'], name); out['banks'][name] = r
        m = r['methods']
        log(name, ' | '.join(f'{k}: {v["pd_paper"]:.4f}' for k, v in m.items()))
        for k, d in r['diffs'].items():
            log('   ', k, 'pd_paper %+.4f [%+.4f, %+.4f]' % tuple(d['pd_paper']))
        json.dump(out, open(f'{HERE}/ld_adapt_adc_{TAG}.json', 'w'), indent=1, default=float)

fb = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
snr_all = fb['meta'][:, 1].astype(np.float32)
idx = np.concatenate([np.where(snr_all == s)[0][:N_OFF] for s in sorted(set(snr_all.tolist()))])
Y16 = sim.downsample16(fb['data'][idx]); Y = Y16[..., 0] + 1j * Y16[..., 1]
np.save(f'{PC}/official_idx.npy', idx)
r = evaluate(Y.astype(np.complex128), fb['feat'][idx, 0, :3], fb['feat'][idx, 1, :3], snr_all[idx], 'official_subset'); out['banks']['official_subset'] = r
log('official_subset', ' | '.join(f'{k}: {v["pd_paper"]:.4f}' for k, v in r['methods'].items()))
for k, d in r['diffs'].items():
    log('   ', k, 'pd_paper %+.4f [%+.4f, %+.4f]' % tuple(d['pd_paper']))
json.dump(out, open(f'{HERE}/ld_adapt_adc_{TAG}.json', 'w'), indent=1, default=float)
log('done')
