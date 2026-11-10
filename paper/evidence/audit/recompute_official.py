"""Audit recomputation (research-understanding agent, 2026-10-02).
Recomputes paper-style Pd per SNR from the cached predictions of IABR-v2, NOMP (no network) and the paper U-Net on the official seed-42 bank,
and a paired bootstrap (scene resampling within each SNR, 2000 reps, seed 0). Expected: 0.7551 / 0.7581 / 0.7366; IABR-v2 - NOMP = -0.0029 [-0.0042,-0.0017].
Needs: D:/ai_ml_project/IABR_Net_TestSuite (evaluator + frozen bank) and IABR_v2_extracted/.../outputs/cache."""
import sys, os, ast, numpy as np
sys.path.insert(0,'D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA'); sys.path.insert(0,'D:/ai_ml_project/IABR_Net_TestSuite')
import cv2
from scipy.optimize import linear_sum_assignment
EVAL='D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA/src/TVT_Blob_Inference.py'
from src.tvt_data_generation_v3 import permute_pairs
tree=ast.parse(open(EVAL,encoding='utf-8').read())
want=['prepare_for_metric','get_ang_difference']
ns=dict(np=np,cv2=cv2)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in want],type_ignores=[]),EVAL,'exec'),ns)
prepare_for_metric,get_ang_difference=ns['prepare_for_metric'],ns['get_ang_difference']
fb=np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
feat,meta=fb['feat'],fb['meta']; snr=meta[:,1].astype(int); L=meta[:,0].astype(int)
def per_sample(model):
    z=np.load(f'D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/pred__{model}__official.npz')
    good=np.zeros(len(L)); tot=np.zeros(len(L)); drop=np.zeros(len(L),bool)
    for i in range(len(L)):
        n=z['n'][i]; pr=(z['psi'][i,:n],z['phi'][i,:n]); f=feat[i,:,:L[i]].astype(np.float64)
        gt,p=prepare_for_metric(pr,f)
        if np.isnan(p).any(): drop[i]=True; continue
        d=np.abs(get_ang_difference(gt,p,return_flat=False))<=1.0
        good[i]=d.sum(); tot[i]=d.size
    return good,tot,drop
res={}
for m in ['IABR-v2','NOMP_no_network','U-Net_paper']:
    try: res[m]=per_sample(m)
    except Exception as e: print('fail',m,e)
import glob
print([os.path.basename(p) for p in glob.glob('D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/pred__*__official.npz')])
def mean_pd(g,t,idx_by_snr):
    return np.mean([g[ix].sum()/t[ix].sum() for ix in idx_by_snr])
ib=[np.where(snr==s)[0] for s in sorted(set(snr))]
for m,(g,t,d) in res.items(): print(m,'mean Pd(paper)=%.4f'%mean_pd(g,t,ib),'dropped',int(d.sum()))
rng=np.random.default_rng(0)
def boot(a,b,B=2000):
    diffs=[]
    for _ in range(B):
        vals=[]
        for ix in ib:
            s=rng.choice(ix,len(ix))
            va=res[a][0][s].sum()/res[a][1][s].sum(); vb=res[b][0][s].sum()/res[b][1][s].sum(); vals.append(va-vb)
        diffs.append(np.mean(vals))
    diffs=np.array(diffs); return diffs.mean(),np.percentile(diffs,[2.5,97.5])
for a,b in [('IABR-v2','NOMP_no_network'),('IABR-v2','U-Net_paper'),('NOMP_no_network','U-Net_paper')]:
    if a in res and b in res:
        m,ci=boot(a,b)
        print(a,'-',b,'paired bootstrap (resample within SNR, 2000 reps): mean diff %.4f, 95%% CI [%.4f, %.4f]'%(m,ci[0],ci[1]))
