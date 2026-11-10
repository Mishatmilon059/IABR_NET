import torch, torch.nn.functional as F, traceback, re, sys
sys.path.insert(0, '.')
import user_nomp as M

B, L, N = 8, 3, 16
torch.manual_seed(0)
Hc = torch.randn(B, N, N, dtype=torch.complex64)
U0 = torch.rand(B, L) * torch.pi; V0 = torch.rand(B, L) * torch.pi
Ug = torch.rand(B, L) * torch.pi; Vg = torch.rand(B, L) * torch.pi


def attempt(tag, model):
    print(f'\n=== {tag} ===')
    try:
        U, V = model(Hc, U0, V0)
        print('forward OK')
        loss = F.mse_loss(U, Ug) + F.mse_loss(V, Vg)
        opt = torch.optim.Adam(model.parameters(), 1e-3); opt.zero_grad(); loss.backward(); opt.step()
        print('backward OK, loss', float(loss), 'params', sum(p.numel() for p in model.parameters()))
    except Exception as e:
        print(type(e).__name__, ':', str(e)[:230])


# 1) exactly as pasted
attempt('1. as pasted', M.LearnedNOMP(L))

# 2) fix only the broadcast
src = open('user_nomp.py').read()
src2 = src.replace('A[:, :, l] * gains[:, l])', 'A[:, :, l] * gains[:, l:l+1])')
ns = {}; exec(compile(src2, 'u2', 'exec'), ns)
attempt('2. broadcast fixed (gains[:, l:l+1])', ns['LearnedNOMP'](L))

# 3) also remove the in-place updates on U, V
src3 = src2.replace('''                U[:, l] = U[:, l] + du
                V[:, l] = V[:, l] + dv''', '''                U = torch.cat([U[:, :l], (U[:, l] + du)[:, None], U[:, l+1:]], 1)
                V = torch.cat([V[:, :l], (V[:, l] + dv)[:, None], V[:, l+1:]], 1)''')
ns3 = {}; exec(compile(src3, 'u3', 'exec'), ns3)
attempt('3. broadcast + in-place fixed', ns3['LearnedNOMP'](L))
