"""Static / light audit (no heavy compute):
 (a) seed disjointness: every .py/.ipynb/.json/.md under D:/ai_ml_project (except this folder) searched for (i) the training seeds of the three
     stages (31337, 31338, 61000000 / 61_000_000, 71000000, 71000001, 71000002) and (ii) this stage's new seeds (8700..8787 set);
     the training seeds are SeedSequence entropy LISTS [seed, step] while every test bank is default_rng(int) -> different streams;
 (b) checkpoints: md5 of final.pt / lurd.pt / adc_ft.pt / imp_ft.pt / ld_adc_final_run.pt, file times, and the parameter-level difference
     between each fine-tuned file and its parent (0 would mean the 'fine-tuned' net is the original);
 (c) the ADC used in training (stress_sim.adc via train_ft.batch) is the same function as the test generator (generate(.., 'M5'));
 (d) IABR-v2 parameter count of the fine-tuned checkpoints.
usage: python s1_audit.py > s1_audit.log"""
import os, re, hashlib, glob, time
os.environ['OMP_NUM_THREADS'] = '2'
import numpy as np, torch

ROOT = 'D:/ai_ml_project'; ME = os.path.normpath('D:/ai_ml_project/paper/evidence/deep_research/finetune/adversarial')
TRAIN = ['31337', '31338', '61000000', '61_000_000', '71000000', '71000001', '71000002']
NEW = [str(s) for s in [8700, 8715, 8730, 8745, 8760, 8775] + list(range(8780, 8788))]
hits = {s: [] for s in TRAIN + NEW}
nfile = 0
for dp, dn, fn in os.walk(ROOT):
    if os.path.normpath(dp).startswith(ME) or any(x in dp for x in ['.git', '__pycache__', 'node_modules']):
        continue
    for f in fn:
        if not f.endswith(('.py', '.ipynb', '.md', '.json', '.txt', '.cfg', '.yaml')):
            continue
        p = os.path.join(dp, f)
        try:
            if os.path.getsize(p) > 5e6:
                continue
            t = open(p, encoding='utf-8', errors='ignore').read()
        except Exception:
            continue
        nfile += 1
        for s in hits:
            if re.search(r'(?<![\d.])' + re.escape(s) + r'(?![\d.])', t):
                hits[s].append(os.path.relpath(p, ROOT))
print(f'(a) searched {nfile} text files under {ROOT} (excluding this folder)')
for s in TRAIN:
    print(f'   training seed {s:11s}: {len(hits[s])} files: {hits[s][:8]}')
for s in NEW:
    print(f'   new seed {s}: {len(hits[s])} files {hits[s][:5]}')
print('   SeedSequence check: default_rng([31337, s]) vs default_rng(5000/5015/8300/8315/8400/8415/87xx) first draws differ:',
      all(np.random.default_rng([31337, 0]).integers(0, 2**31) != np.random.default_rng(t).integers(0, 2**31) for t in [5000, 5015, 8300, 8315, 8400, 8415, 8700, 8715]))


def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


CK = dict(final=f'{ROOT}/IABR_v2_extracted/IABR_v2/outputs/checkpoints/IABRv2/final.pt', lurd=f'{ROOT}/paper/evidence/learned_decoder/lurd.pt',
          adc_ft=f'{ROOT}/paper/evidence/deep_research/finetune/adc/cache/adc_ft.pt', imp_ft=f'{ROOT}/paper/evidence/deep_research/finetune/imp/cache/imp_ft.pt',
          ld_adc=f'{ROOT}/paper/evidence/deep_research/nn_decoder_adapt/cache/ld_adc_final_run.pt')
sd = {}
print('(b) checkpoints')
for k, p in CK.items():
    s = torch.load(p, map_location='cpu', weights_only=False)
    s = s['model'] if isinstance(s, dict) and 'model' in s and isinstance(s['model'], dict) else s
    sd[k] = s
    print(f'   {k:7s} md5 {md5(p)}  mtime {time.ctime(os.path.getmtime(p))}  tensors {len(s)}  params {sum(v.numel() for v in s.values() if v.dtype.is_floating_point)}')
for a, b in [('adc_ft', 'final'), ('imp_ft', 'final'), ('adc_ft', 'imp_ft'), ('ld_adc', 'lurd')]:
    dmax = max(float((sd[a][k].double() - sd[b][k].double()).abs().max()) for k in sd[b] if sd[b][k].dtype.is_floating_point)
    rel = np.sqrt(sum(float(((sd[a][k].double() - sd[b][k].double()) ** 2).sum()) for k in sd[b] if sd[b][k].dtype.is_floating_point) /
                  sum(float((sd[b][k].double() ** 2).sum()) for k in sd[b] if sd[b][k].dtype.is_floating_point))
    print(f'   {a} vs {b}: same keys {set(sd[a]) == set(sd[b])}, max |diff| {dmax:.3e}, relative L2 change {rel:.4f}')
print('(c) train_ft.batch applies stress_sim.adc(Y, b, "peak") (finetune/adc/train_ft.py line 31); the test banks use stress_sim.generate(base,'
      ' "M5", b) which calls the same adc(Y, sev, "peak") (stress_sim.py); equality of adc(generate clean) and generate(M5) checked numerically'
      ' in t0_unit.log (2).')
