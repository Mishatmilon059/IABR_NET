"""Step-time benchmark for IABC-v2 sizes (seed 61010; timing only)."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='6'
import time, numpy as np, torch
torch.set_num_threads(6)
from iabc2_model import *
b = make_train_batch(np.random.default_rng(61010), 256)
Ht = torch.complex(torch.from_numpy(b['Hr']), torch.from_numpy(b['Hi'])); tg = torch.from_numpy(b['tgt'])
# feature sanity: Tx detrended-phase feature equals minus the old IABC feature
import sys; sys.path.insert(0,'D:/ai_ml_project/paper/evidence/deep_research/iabc')
for cfg in [dict(d=128, layers=6, heads=4, ff=384), dict(d=192, layers=6, heads=6, ff=512), dict(d=128, layers=4, heads=4, ff=256)]:
    m = IABCv2(**cfg); opt = torch.optim.AdamW(m.parameters(), 3e-4)
    for i in range(6):
        if i == 1: t = time.time()
        loss = ((m(Ht) - tg) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    print(cfg, 'params', n_params(m), f'{(time.time()-t)/5*1000:.0f} ms/step (B=256)', flush=True)
t = time.time(); [make_train_batch(np.random.default_rng([61011, i]), 256) for i in range(5)]; print(f'batch gen {(time.time()-t)/5*1000:.0f} ms')
