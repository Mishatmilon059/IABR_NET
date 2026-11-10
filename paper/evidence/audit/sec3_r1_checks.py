"""Section III revision round 1 checks. Run: python sec3_r1_checks.py > sec3_r1_checks.log"""
import sys, math, numpy as np
sys.path.insert(0, r'D:/ai_ml_project/IABR_Net_TestSuite')
import dldoa_dataset_generation as DG
N=16; cell=2*math.pi/N
fb=np.load(r'D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz')
psi=fb['feat'][:,0,:].astype(float); phi=fb['feat'][:,1,:].astype(float)
print('== A. official bank, L=3, 8000 scenes ==')
def cd(a,b): d=np.abs(a-b)%(2*np.pi); return np.minimum(d,2*np.pi-d)
U=np.pi*np.cos(psi); V=np.pi*np.cos(phi)
m=np.min(np.stack([np.maximum(cd(U[:,i],U[:,j]),cd(V[:,i],V[:,j])) for i,j in [(0,1),(0,2),(1,2)]],1),1)
k1=int((m<cell).sum()); k2=int((m<cell/2).sum())
print(f'close pair (circular, both du and dv < 2pi/N): {k1} of 8000 = {k1/8000:.4f}; below half cell: {k2} of 8000 = {k2/8000:.4f}; binomial sd {math.sqrt(0.0386*0.9614/8000):.4f}')
zone=lambda u: np.abs(u)>=np.pi-cell
for nm,a in (('psi',psi),('phi',phi)):
    print(f'end-fire zone (|u|>=pi-2pi/N) share of {nm}: {zone(np.pi*np.cos(a)).mean():.4f}')
print(f'end-fire core (|cos|>0.995) share, all 6 angle sets pooled: {np.mean(np.abs(np.cos(np.concatenate([psi.ravel(),phi.ravel()])))>0.995):.4f}; analytic 2*arccos(0.995)/pi = {2*math.acos(0.995)/math.pi:.4f}; analytic zone {2*math.acos(1-2/N)/math.pi:.4f}')
print('== B. generator L=9, first vs last placed, zone |u|>=pi-2pi/N and 10-deg zone, 20000 scenes ==')
rng=np.random.default_rng(7); F=[];La=[];F10=[];L10=[]
for _ in range(20000):
    pts=DG.generate_points(9, math.pi/6, rng=rng)
    ph=np.array([p[0] for p in pts]); ps=np.array([p[1] for p in pts])
    a=np.concatenate([ps,ph]); 
    F.append((zone(np.pi*np.cos(ps[0])), zone(np.pi*np.cos(ph[0])))); La.append((zone(np.pi*np.cos(ps[-1])), zone(np.pi*np.cos(ph[-1]))))
    d=np.rad2deg
    F10.append((d(ps[0])<10 or d(ps[0])>170, d(ph[0])<10 or d(ph[0])>170)); L10.append((d(ps[-1])<10 or d(ps[-1])>170, d(ph[-1])<10 or d(ph[-1])>170))
F=np.array(F).mean(); La=np.array(La).mean(); F10=np.array(F10).mean(); L10=np.array(L10).mean()
n=2*20000
print(f'zone share first {F:.4f} last {La:.4f} (binomial sd {math.sqrt(0.32*0.68/n):.4f}); analytic uniform 0.3222')
print(f'10-deg zone share first {F10:.4f} last {L10:.4f} (uniform 0.1111)')
print('== C. angle-difference metric (get_ang_difference: angle(exp(ja)exp(-jb))) on radians in [0,pi] ==')
a=math.radians(0.5); b=math.radians(179.5); print('0.5 vs 179.5 deg ->', abs(math.degrees(np.angle(np.exp(1j*a)*np.exp(-1j*b)))),'deg')
