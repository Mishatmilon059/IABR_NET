"""Parameter counts of the learned modules of designs A and B (layer shapes from DESIGN.md; seed CNN 191,702 from nn_decoder/REPORT.md)."""
import torch.nn as nn
def n(m): return sum(p.numel() for p in m.parameters())
mlp = lambda d: nn.Sequential(*[nn.Linear(a, b) for a, b in zip(d[:-1], d[1:])])
parts = dict(film=nn.Linear(6, 640), scene=mlp([45, 256, 256, 256, 15]), hyper=nn.ModuleList([mlp([36, 256, 256, 256, 8]), nn.Embedding(24, 16)]),
             fusion=nn.Sequential(nn.Conv2d(5, 32, 3), nn.Conv2d(32, 32, 3), nn.Conv2d(32, 1, 3)), step=mlp([16, 256, 256, 256, 2]),
             stop=mlp([19, 128, 128, 1]),
             sel=nn.ModuleList([nn.Linear(24, 128), nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 4, 256, batch_first=True), 2, enable_nested_tensor=False), nn.Linear(128, 1)]))
c = {k: n(v) for k, v in parts.items()}; print(c)
CNN = 191702
print('A', CNN + sum(c.values()), 'B', CNN + sum(c[k] for k in ('film', 'scene', 'hyper', 'fusion', 'stop')))
