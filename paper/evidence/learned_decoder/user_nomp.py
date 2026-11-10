import torch
import torch.nn as nn
import torch.nn.functional as F


def steering(u, v, N=16):
    n = torch.arange(N, device=u.device, dtype=u.dtype)
    m = torch.arange(N, device=u.device, dtype=u.dtype)
    nn_, mm_ = torch.meshgrid(n, m, indexing="ij")
    return torch.exp(
        -1j * nn_[None] * u[:, None, None]
        + 1j * mm_[None] * v[:, None, None]
    )


def estimate_gains(H, U, V):
    B, N, _ = H.shape
    L = U.shape[1]
    A = []
    for l in range(L):
        a = steering(U[:, l], V[:, l], N)
        A.append(a.reshape(B, -1))
    A = torch.stack(A, dim=-1)
    y = H.reshape(B, -1, 1)
    AH = A.conj().transpose(1, 2)
    reg = 1e-6 * torch.eye(L, device=H.device, dtype=H.real.dtype).to(torch.complex64)
    ATA = AH @ A + reg[None]
    ATy = AH @ y
    gains = torch.linalg.solve(ATA, ATy).squeeze(-1)
    return gains, A


def reconstruct(A, gains):
    return (A * gains[:, None, :]).sum(dim=-1)


class LearnedNewton(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, features):
        delta = self.net(features)
        return delta


class LearnedNOMP(nn.Module):
    def __init__(self, num_paths, rounds=3, N=16, max_step=0.2):
        super().__init__()
        self.L = num_paths
        self.rounds = rounds
        self.N = N
        self.max_step = max_step
        self.refiner = LearnedNewton()

    def make_features(self, residual, u, v):
        B = residual.shape[0]
        n = torch.arange(self.N, device=residual.device, dtype=residual.real.dtype)
        m = torch.arange(self.N, device=residual.device, dtype=residual.real.dtype)
        nn_, mm_ = torch.meshgrid(n, m, indexing="ij")
        E = torch.exp(
            1j * nn_[None] * u[:, None, None]
            - 1j * mm_[None] * v[:, None, None]
        )
        rE = residual * E
        S = rE.sum((1, 2))
        Su = (1j * nn_[None] * rE).sum((1, 2))
        Sv = (-1j * mm_[None] * rE).sum((1, 2))
        Suu = (-(nn_[None] ** 2) * rE).sum((1, 2))
        Svv = (-(mm_[None] ** 2) * rE).sum((1, 2))
        Suv = ((nn_[None] * mm_[None]) * rE).sum((1, 2))
        gu = 2 * torch.real(S.conj() * Su)
        gv = 2 * torch.real(S.conj() * Sv)
        huu = 2 * torch.real(Su.conj() * Su + S.conj() * Suu)
        hvv = 2 * torch.real(Sv.conj() * Sv + S.conj() * Svv)
        huv = 2 * torch.real(Su.conj() * Sv + S.conj() * Suv)
        features = torch.stack([S.real, S.imag, gu, gv, huu, hvv, huv, torch.abs(S)], dim=1)
        return features

    def forward(self, Hc, U, V):
        U = U.clone()
        V = V.clone()
        for _ in range(self.rounds):
            gains, A = estimate_gains(Hc, U, V)
            for l in range(self.L):
                own = (A[:, :, l] * gains[:, l])
                total = reconstruct(A, gains)
                others = total - own
                residual = (Hc - others.reshape_as(Hc))
                features = self.make_features(residual, U[:, l], V[:, l])
                delta = self.refiner(features)
                du = delta[:, 0]
                dv = delta[:, 1]
                du = self.max_step * torch.tanh(du)
                dv = self.max_step * torch.tanh(dv)
                U[:, l] = U[:, l] + du
                V[:, l] = V[:, l] + dv
        return U, V


def uv_to_angles(U, V):
    U = torch.clamp(U / torch.pi, -1, 1)
    V = torch.clamp(V / torch.pi, -1, 1)
    return torch.acos(U), torch.acos(V)
