"""Step 0: inventory of banks with a net__ cache, truth, and cached baseline predictions. usage: python s0_inventory.py > s0_inventory.log"""
import os, sys, json
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
from common import *
HERE = os.path.dirname(os.path.abspath(__file__)); VER = 'D:/ai_ml_project/paper/evidence/deep_research/verify'
inv = {}
for f in sorted(os.listdir(CACHE)):
    if not f.startswith('net__'):
        continue
    n = f[5:-4]
    if not bank_exists(n):
        print(f'{n:22s} NO TRUTH'); inv[n] = None; continue
    b = load_bank(n); z = np.load(f'{CACHE}/{f}')
    has = {p: os.path.exists(f'{CACHE}/pred__{p}__{n}.npz') for p in ['U-Net_paper', 'NOMP_no_network', 'IABR-v2', 'v2_network_only_no_physics']}
    has['preds_heat'] = os.path.exists(f'{OUT}/preds_heat/{n}.npz'); has['verify_heat'] = os.path.exists(f'{VER}/cache/heat__{n}.npz')
    has['verify_netonly'] = os.path.exists(f'{VER}/cache/netonly__{n}.npz')
    inv[n] = dict(N=len(b['L']), N_net=len(z['heat']), L=sorted(set(b['L'].tolist())), snr=sorted(set(b['snr'].astype(int).tolist())), has=has)
    print(f'{n:22s} N={len(b["L"]):5d} N_net={len(z["heat"]):5d} L={inv[n]["L"]} snr={inv[n]["snr"]} ' + ' '.join(k for k, v in has.items() if v), flush=True)
json.dump(inv, open(f'{HERE}/s0_inventory.json', 'w'), indent=1)
