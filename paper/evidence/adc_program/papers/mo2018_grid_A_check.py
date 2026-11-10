"""Structure of the Mo-et-al. measurement matrix A for OUR narrowband analog beamspace model (numpy only, no data, no randomness).
Model (mlq.py docstring / iabr2_sim): Y = W^H Hm F, Hm[n,m] = sum_k x_k exp(-j n u_k + j m v_k), W, F = 16-beam DFT codebooks
(dldoa_dataset_generation.beamforming_vector_generation_Q/_P). Grid with oversampling G: u, v in {2*pi*i/(16G)}.
A[:, (i,j)] = vec(W^H a(u_i) b(v_j)^T F). Reports: |A| per column (how many beams a grid atom touches), A^H A for G=1,
and the singular values of A for G = 1, 2, 4 (Mo et al. Sec. IV-D want equal singular values)."""
import sys
sys.path.insert(0, 'D:/ai_ml_project')
import numpy as np
import dldoa_dataset_generation as DG
W = DG.beamforming_vector_generation_Q(16, 16); F = DG.beamforming_vector_generation_P(16, 16)
print('W unitary err', np.abs(W.conj().T @ W - np.eye(16)).max(), ' F unitary err', np.abs(F.conj().T @ F - np.eye(16)).max())
n = np.arange(16.)
for G in [1, 2, 4]:
    g = 2 * np.pi * np.arange(16 * G) / (16 * G)
    R = W.conj().T @ np.exp(-1j * np.outer(n, g))       # 16 x 16G   receive part (beam q, grid i)
    T = F.T @ np.exp(1j * np.outer(n, g))               # 16 x 16G   transmit part (beam p, grid j)
    A = np.kron(T, R) / 16.0                            # vec column-major: vec(R x T^T) = (T kron R) vec(x)
    s = np.linalg.svd(A, compute_uv=False)
    big = (np.abs(R) > 1e-6 * np.abs(R).max()).sum(0)
    print(f'G={G}: A is {A.shape[0]}x{A.shape[1]}; receive beams touched per grid atom: min {big.min()} max {big.max()}; '
          f'singular values min {s.min():.4f} max {s.max():.4f} (nonzero count {np.sum(s > 1e-8)})')
    if G == 1:
        AA = A.conj().T @ A
        print('   G=1: max|A^H A - I| =', f'{np.abs(AA - np.eye(256)).max():.2e}', '; nonzeros per column of A:',
              int(np.median((np.abs(A) > 1e-8).sum(0))))
