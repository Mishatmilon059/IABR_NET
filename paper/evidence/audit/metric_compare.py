"""Audit: relation between the paper-style Pd and the learned-decoder score of paper/evidence/README.md.
Uses the cached predictions of the v2 heatmap-init-only variant on the official bank. Expected output:
paper full 0.7503 / first-250 0.7464 ; score full 0.7494 / first-250 0.7448 ; 256 of 8000 scenes differ."""
import sys, os, ast, itertools, numpy as np
sys.path.insert(0,'D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA'); sys.path.insert(0,'D:/ai_ml_project/IABR_Net_TestSuite')
import cv2
from src.tvt_data_generation_v3 import permute_pairs
EVAL='D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA/src/TVT_Blob_Inference.py'
tree=ast.parse(open(EVAL,encoding='utf-8').read()); ns=dict(np=np,cv2=cv2)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['prepare_for_metric','get_ang_difference']],type_ignores=[]),EVAL,'exec'),ns)
fb=np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz'); feat,meta=fb['feat'],fb['meta']; snr=meta[:,1].astype(int)
z=np.load('D:/ai_ml_project/IABR_v2_extracted/IABR_v2/outputs/cache/pred__v2_heatmap_init_only__official.npz')
PERMS=list(itertools.permutations(range(3)))
def paper_ok(i):
    n=z['n'][i]; gt,p=ns['prepare_for_metric']((z['psi'][i,:n],z['phi'][i,:n]),feat[i,:,:3].astype(np.float64))
    return (np.abs(ns['get_ang_difference'](gt,p,return_flat=False))<=1.0).sum()
def score_ok(i):
    ps,ph=np.degrees(z['psi'][i,:3]),np.degrees(z['phi'][i,:3]); tp,tf=np.degrees(feat[i,0]),np.degrees(feat[i,1])
    best=min(PERMS,key=lambda p:sum(abs(ps[q]-tp[j])+abs(ph[q]-tf[j]) for j,q in enumerate(p)))
    return sum(abs(ps[q]-tp[j])<=1 for j,q in enumerate(best))+sum(abs(ph[q]-tf[j])<=1 for j,q in enumerate(best))
snrs=sorted(set(snr)); full={}; sub={}
for name,fn in [('paper',paper_ok),('score',score_ok)]:
    a=[];b=[]; disagree=0
    for s in snrs:
        ix=np.where(snr==s)[0]
        v=np.array([fn(i) for i in ix])/6.0
        a.append(v.mean()); b.append(v[:250].mean()); full[name]=full.get(name,{}); full[name][s]=v
    print(name,'full 1000/SNR mean %.4f'%np.mean(a),' first-250/SNR mean %.4f'%np.mean(b))
d=sum(int(full['paper'][s][k]!=full['score'][s][k]) for s in snrs for k in range(1000)); print('samples where the two pairing rules give different counts:',d,'of 8000')
