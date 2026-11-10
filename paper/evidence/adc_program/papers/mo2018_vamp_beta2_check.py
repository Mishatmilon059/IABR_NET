"""Numerical check of the VAMP LMMSE-z divergence beta2 (Mo et al. 2018 Alg. 2 line 18; Schniter-Rangan-Fletcher 2016 eq. 19).
A is Ny x Nx (complex). LMMSE with pseudo-prior x ~ CN(r2, nu2 I), pseudo-measurement p2 = A x + CN(0, tau2 I).
beta2 := (1/Ny) tr(d z2 / d p2) = (1/Ny) tr(A C A^H)/tau2 with C = (A^H A/tau2 + I/nu2)^-1, alpha2 := (1/Nx) tr(C)/nu2.
Compares beta2 with (1-alpha2)*Nx/Ny and (1-alpha2)*Ny/Nx. Seed 0."""
import numpy as np
rng = np.random.default_rng(0)
for Ny, Nx in [(256, 256), (256, 1024), (256, 4096 // 4), (512, 256)]:
    A = (rng.standard_normal((Ny, Nx)) + 1j * rng.standard_normal((Ny, Nx))) / np.sqrt(2 * Ny)
    nu2, tau2 = 0.7, 0.3
    C = np.linalg.inv(A.conj().T @ A / tau2 + np.eye(Nx) / nu2)
    alpha2 = np.trace(C).real / nu2 / Nx
    beta2 = np.trace(A @ C @ A.conj().T).real / tau2 / Ny
    print(f'Ny={Ny:4d} Nx={Nx:5d}  beta2={beta2:.6f}  (1-a2)Nx/Ny={(1-alpha2)*Nx/Ny:.6f}  (1-a2)Ny/Nx={(1-alpha2)*Ny/Nx:.6f}')
