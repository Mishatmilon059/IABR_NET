import sys, os, ast, numpy as np
sys.path.insert(0,"D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA"); sys.path.insert(0,"D:/ai_ml_project/IABR_Net_TestSuite")
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
g,t,d=per_sample('U-Net_paper')
ib=[np.where(snr==s)[0] for s in sorted(set(snr))]
print('U-Net dropped per SNR', {int(s):int(d[ix].sum()) for s,ix in zip(sorted(set(snr)),ib)}, 'total', int(d.sum()))
print('mean over SNR points of paper-style Pd', np.mean([g[ix].sum()/t[ix].sum() for ix in ib]))
gg=g.sum(); tt=t.sum(); print('pooled paper-style', gg/tt)
