"""Step 5 (optional): PyTorch port of the published ResNet of Lloria et al. (DL_DOA/src/tvt_models.py: Resnet(input_shape=(64,64,2)):
Conv2DTranspose(12, 5x5, stride 2, 'same') -> 64 x res_conv(12 filters, two 5x5 'same' Conv2D + BatchNorm, ReLU, identity skip, ReLU after add)
-> Conv2DTranspose(1, 5x5, stride 2, 'same'); output (256,256,1)), weights from DL_DOA/models/inf_model_007_256_resnet.h5 read with h5py.
Post-processing exactly as TVT_Blob_Inference.run_inference_and_metrics_resnet / notebook unet_predict: OpenCV blob detector on the min-max
normalised map, top-L blobs by amplitude, peaks_to_angles(sigma=0.07, grid 256). Input: upsample64(Y16) as for the cached U-Net predictions.
Checks: (1) the 'same' transposed convolution (crop 1 at the start, 2 at the end of the full output) equals TF's definition
(adjoint of conv2d SAME stride 2, pad_before = pad_total // 2), computed with autograd; (2) per-SNR Pd on the official bank vs the authors'
repo pickle (DL_DOA/figures_resnet/resnet_pd.pkl) and Table II of the TVT paper (0 dB: 0.643, 20 dB: 0.925 at delta_max 0; T2_d1/2/5 rows).
Resumable: chunk files in DRC/resnet_chunks/. usage: python -u s5_resnet.py official,T2_d1,T2_d2,T2_d5 > s5_resnet.log"""
import os, sys, ast, json, time, pickle
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '6'
os.environ['IABR2_SUITE'] = 'D:/ai_ml_project'
sys.path.insert(0, 'D:/ai_ml_project/IABR_v2_extracted/IABR_v2'); sys.path.insert(0, 'D:/ai_ml_project/paper/evidence/new_results')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as TF, h5py, cv2
torch.set_num_threads(6)
import iabr2_sim as sim
import common as C
HERE = os.path.dirname(os.path.abspath(__file__))
DRC = 'C:/Users/Milon/AppData/Local/Temp/claude/D--thesis/a5db9a29-d75e-4561-834b-cd0638a16d91/scratchpad/dr_cache/nn_decoder'
CH = f'{DRC}/resnet_chunks'; os.makedirs(CH, exist_ok=True)
H5 = 'D:/ai_ml_project/DL_DOA/models/inf_model_007_256_resnet.h5'
EVAL = 'D:/ai_ml_project/IABR_Net_TestSuite/DL_DOA/src/TVT_Blob_Inference.py'
tree = ast.parse(open(EVAL, encoding='utf-8').read()); ns = dict(np=np, cv2=cv2)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                              ['prepare_prediction_for_peaks', 'get_blob_detector', 'reorder_keypoints', 'get_blob_peaks', 'wrap_2pi_to_minus_pi', 'peaks_to_angles']],
                        type_ignores=[]), EVAL, 'exec'), ns)
DETECTOR = ns['get_blob_detector'](); get_blob_peaks, peaks_to_angles = ns['get_blob_peaks'], ns['peaks_to_angles']


def tconv_same(m, x):          # Keras Conv2DTranspose(k=5, s=2, padding='same'): full output 2n+3, keep [1, 1+2n)
    y = m(x); n2 = 2 * x.shape[-1]
    return y[..., 1:1 + n2, 1:1 + n2]


class ResNetPaper(nn.Module):
    def __init__(self, nblocks=64, f=12):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(2, f, 5, 2); self.up2 = nn.ConvTranspose2d(f, 1, 5, 2)
        self.convs = nn.ModuleList([nn.Conv2d(f, f, 5, padding=2) for _ in range(2 * nblocks)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(f, eps=1e-3) for _ in range(2 * nblocks)])

    def forward(self, x):            # (B,2,64,64) -> (B,1,256,256)
        x = tconv_same(self.up1, x)
        for i in range(0, len(self.convs), 2):
            s = x; x = torch.relu(self.bns[i](self.convs[i](x))); x = self.bns[i + 1](self.convs[i + 1](x)); x = torch.relu(x + s)
        return tconv_same(self.up2, x)

    def load_keras_h5(self, path):
        f = h5py.File(path, 'r'); g = f['model_weights'] if 'model_weights' in f else f
        names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs['layer_names']]
        wl = []
        for n in names:
            ws = [w.decode() if isinstance(w, bytes) else w for w in g[n].attrs['weight_names']]
            if ws:
                wl.append((n, [np.array(g[n][w]) for w in ws]))
        num = lambda s: int(s.rsplit('_', 1)[1])
        convs = sorted([x for x in wl if x[0].startswith('conv2d_') and not x[0].startswith('conv2d_transpose')], key=lambda x: num(x[0]))
        tconvs = sorted([x for x in wl if x[0].startswith('conv2d_transpose')], key=lambda x: num(x[0]))
        bns = sorted([x for x in wl if x[0].startswith('batch_normalization')], key=lambda x: num(x[0]))
        assert len(convs) == 128 and len(tconvs) == 2 and len(bns) == 128 and len(wl) == 258, (len(convs), len(tconvs), len(bns), len(wl))
        with torch.no_grad():
            for (_, (k, b)), m in zip(convs, self.convs):
                m.weight.copy_(torch.from_numpy(k.transpose(3, 2, 0, 1))); m.bias.copy_(torch.from_numpy(b))
            for (_, (k, b)), m in zip(tconvs, [self.up1, self.up2]):
                m.weight.copy_(torch.from_numpy(k.transpose(3, 2, 0, 1))); m.bias.copy_(torch.from_numpy(b))
            for (_, (ga, be, mu, va)), m in zip(bns, self.bns):
                m.weight.copy_(torch.from_numpy(ga)); m.bias.copy_(torch.from_numpy(be)); m.running_mean.copy_(torch.from_numpy(mu)); m.running_var.copy_(torch.from_numpy(va))
        return self


net = ResNetPaper().load_keras_h5(H5).eval()
print(f'ResNet port: {sum(p.numel() for p in net.parameters()):,} parameters (+{sum(b.numel() for b in net.buffers() if b.dtype.is_floating_point):,} BN buffers)', flush=True)

# check (1): 'same' transposed conv vs TF definition (adjoint of conv2d SAME, stride 2), on the loaded first-layer kernel
torch.manual_seed(0); x = torch.randn(2, 2, 64, 64, dtype=torch.float64); W = net.up1.weight.detach().double()
z = torch.zeros(2, 12, 128, 128, dtype=torch.float64, requires_grad=True)
out = TF.conv2d(TF.pad(z, (1, 2, 1, 2)), W, stride=2)          # TF SAME: pad_total 3 -> before 1, after 2
(gz,) = torch.autograd.grad(out, z, grad_outputs=x)
mine = tconv_same(lambda t: TF.conv_transpose2d(t, W, stride=2), x)
alt = TF.conv_transpose2d(x, W, stride=2, padding=2, output_padding=1)
print(f'check transposed-conv padding: max |mine - TF definition| {float((mine - gz).abs().max()):.2e}; '
      f'naive PyTorch padding=2,output_padding=1 differs by {float((alt - gz).abs().max()):.2e}', flush=True)


@torch.no_grad()
def predict(Y16, Ls, bs=50):
    x64 = sim.upsample64(Y16); out = []
    for s in range(0, len(x64), bs):
        pr = net(torch.from_numpy(np.ascontiguousarray(x64[s:s + bs].transpose(0, 3, 1, 2))).float()).permute(0, 2, 3, 1)
        for j in range(pr.shape[0]):
            peaks, amps = get_blob_peaks(pr[j], DETECTOR)
            peaks = peaks[np.argsort(-amps)[:int(Ls[s + j])]]
            out.append(peaks_to_angles(peaks, sigma=0.07, grid_size=256))
    return out


fb = None
for name in sys.argv[1].split(','):
    final = f'{DRC}/resnet__{name}.npz'
    if os.path.exists(final):
        print(name, 'already done', flush=True); continue
    if name == 'official':
        fb = np.load('D:/ai_ml_project/IABR_Net_TestSuite/data/frozen_banks/eval_bank.npz'); Y16 = sim.downsample16(fb['data'])
        assert np.array_equal(sim.upsample64(Y16[:50]), fb['data'][:50]), 'upsample64(downsample16(data)) != data'
    else:
        Y16 = np.load(f'D:/ai_ml_project/IABR_v2_extracted/IABR_v2/data/banks/{name}.npz')['Y16']
    b = C.load_bank(name); Ls = b['L']; lmax = int(Ls.max()); t0 = time.time(); CS = 400
    for c0 in range(0, len(Ls), CS):
        cf = f'{CH}/{name}__{c0:05d}.npz'
        if os.path.exists(cf):
            continue
        preds = predict(Y16[c0:c0 + CS], Ls[c0:c0 + CS]); n = np.array([len(p[0]) for p in preds])
        ps = np.full((len(preds), lmax), np.nan); ph = ps.copy()
        for i, p in enumerate(preds):
            ps[i, :n[i]] = p[0][:lmax]; ph[i, :n[i]] = p[1][:lmax]
        np.savez_compressed(cf, psi=ps, phi=ph, n=n)
        print(f'  {name} chunk {c0}-{c0 + len(n)} done, {time.time() - t0:.0f}s', flush=True)
    parts = [np.load(f'{CH}/{name}__{c0:05d}.npz') for c0 in range(0, len(Ls), CS)]
    np.savez_compressed(final, psi=np.concatenate([p['psi'] for p in parts]), phi=np.concatenate([p['phi'] for p in parts]), n=np.concatenate([p['n'] for p in parts]))
    S = C.scene_stats(C.load_preds(final), b); per = C.metric_by_snr(S, b['snr'])
    print(f'{name}: ResNet port mean Pd paper-style {per["mean"]["pd_paper"]:.4f}, strict {per["mean"]["pd_strict"]:.4f}, RMSE {per["mean"]["rmse"]:.4f}, '
          f'dropped {int(S[:, 2].sum())}  ({time.time() - t0:.0f}s)', flush=True)
    if name == 'official':
        pub = {k[1]: v for k, v in pickle.load(open('D:/ai_ml_project/DL_DOA/figures_resnet/resnet_pd.pkl', 'rb')).items()}
        pubr = {k[1]: v for k, v in pickle.load(open('D:/ai_ml_project/DL_DOA/figures_resnet/resnet_rmse.pkl', 'rb')).items()}
        rows = []
        for s in sorted(pub):
            rows.append(dict(snr=s, pd_pub=float(pub[s]), pd_ours=per[s]['pd_paper'], rmse_pub=float(pubr[s]), rmse_ours=per[s]['rmse']))
            print(f'   SNR {s:3d}: Pd published (repo pkl) {pub[s]:.4f} ours {per[s]["pd_paper"]:.4f} diff {per[s]["pd_paper"] - pub[s]:+.4f} | '
                  f'RMSE published {pubr[s]:.4f} ours {per[s]["rmse"]:.4f}', flush=True)
        mx = max(abs(r['pd_ours'] - r['pd_pub']) for r in rows)
        print(f'   validation vs authors repo pickle: max |d Pd| {mx:.4f} -> {"PASS" if mx < 0.01 else "FAIL"} (threshold 0.01, as for the U-Net port check)', flush=True)
        json.dump(dict(rows=rows, max_abs_diff=mx), open(f'{HERE}/s5_resnet_validation.json', 'w'), indent=1)
    if name.startswith('T2_d'):
        for s in [0, 20]:
            print(f'   {name} SNR {s}: Pd {per[s]["pd_paper"]:.4f} RMSE {per[s]["rmse"]:.4f}', flush=True)
print('done', flush=True)
