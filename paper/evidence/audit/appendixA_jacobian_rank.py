"""Appendix A check: numerical nullity of the Jacobian of the noise-free model of Eq. (antentries)
   Htilde[n,m] = r_n t_m sum_l alpha_l exp(-j n u_l + j m v_l)
w.r.t. all real parameters (Re/Im of r, t, alpha; u; v). Expected nullity = 6 (kappa_r, kappa_t complex; beta_r, beta_t real)
for generic parameters. Also the nullity when the ramp-and-mean gauge is NOT free (angles known) = 4.
Reports the smallest singular values so the gap to the null space can be read."""
import numpy as np
N = 16
n = np.arange(N)
def model(p, L):
    r = p[0:N] + 1j*p[N:2*N]; t = p[2*N:3*N] + 1j*p[3*N:4*N]
    a = p[4*N:4*N+L] + 1j*p[4*N+L:4*N+2*L]
    u = p[4*N+2*L:4*N+3*L]; v = p[4*N+3*L:4*N+4*L]
    H = np.zeros((N, N), complex)
    for l in range(L):
        H += a[l]*np.outer(np.exp(-1j*n*u[l]), np.exp(1j*n*v[l]))
    H = r[:, None]*H*t[None, :]
    return np.concatenate([H.real.ravel(), H.imag.ravel()])
def jac(p, L, h=1e-6):
    J = np.empty((2*N*N, p.size))
    for k in range(p.size):
        e = np.zeros_like(p); e[k] = h
        J[:, k] = (model(p+e, L)-model(p-e, L))/(2*h)
    return J
rng = np.random.default_rng(0)
for L in (1, 2, 3, 4):
    out = []; tang = []
    for trial in range(3):
        g = np.exp(rng.uniform(-0.3, 0.3, 2*N)); ph = rng.uniform(-0.3, 0.3, 2*N)
        r = g[:N]*np.exp(1j*ph[:N]); t = g[N:]*np.exp(1j*ph[N:])
        a = (rng.normal(size=L)+1j*rng.normal(size=L))/np.sqrt(2*L)
        u = rng.uniform(-np.pi, np.pi, L); v = rng.uniform(-np.pi, np.pi, L)
        p = np.concatenate([r.real, r.imag, t.real, t.imag, a.real, a.imag, u, v])
        J = jac(p, L)
        # six tangent vectors of the orbit (Eq. A-tangent): d kappa_r (re, im), d kappa_t (re, im), d beta_r, d beta_t
        def tv(kind):
            d = np.zeros_like(p)
            if kind in ('kr_re', 'kr_im'):
                c = 1.0 if kind == 'kr_re' else 1j
                dr = c*r; da = -c*a; d[0:N] = dr.real; d[N:2*N] = dr.imag; d[4*N:4*N+L] = da.real; d[4*N+L:4*N+2*L] = da.imag
            if kind in ('kt_re', 'kt_im'):
                c = 1.0 if kind == 'kt_re' else 1j
                dt = c*t; da = -c*a; d[2*N:3*N] = dt.real; d[3*N:4*N] = dt.imag; d[4*N:4*N+L] = da.real; d[4*N+L:4*N+2*L] = da.imag
            if kind == 'br':
                dr = 1j*n*r; d[0:N] = dr.real; d[N:2*N] = dr.imag; d[4*N+2*L:4*N+3*L] = 1.0
            if kind == 'bt':
                dt = -1j*n*t; d[2*N:3*N] = dt.real; d[3*N:4*N] = dt.imag; d[4*N+3*L:4*N+4*L] = 1.0
            return d
        tmax = max(np.linalg.norm(J @ tv(k))/np.linalg.norm(tv(k)) for k in ('kr_re', 'kr_im', 'kt_re', 'kt_im', 'br', 'bt'))
        tang.append(tmax)
        s = np.linalg.svd(J, compute_uv=False)
        tol = s[0]*1e-7
        nullity = p.size - int((s > tol).sum())
        out.append((nullity, s[0], s[p.size-nullity-1], s[p.size-nullity]))
    print('L=%d params=%d  nullity per trial:' % (L, 4*N+4*L), [o[0] for o in out],
          ' smallest nonzero sv %.3e, largest null-space sv %.3e, s_max %.3e; max |J v|/|v| over the 6 tangent vectors (3 draws) %.2e' % (out[0][2], out[0][3], out[0][1], max(tang)))
