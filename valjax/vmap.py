import jax.numpy as np

# easily vmap'able versions of various functions (rather than explicitly array aware)

def grid_index(g, v, extrap=False):
    n = g.size

    i = np.searchsorted(g, v)
    i1 = np.clip(i, 1, n-1)
    i0 = i1 - 1

    g0, g1 = g[i0], g[i1]
    x0 = (v-g0)/(g1-g0)
    xc = np.clip(x0, 0, 1)
    x = np.where(extrap, x0, xc)

    return i0 + x

# catmull-rom spline — we lose n={0,n-2,n-1}
def cubic_spline_fit(x, f):
    # get different views
    fm1 = f[:-3]
    f00 = f[1:-2]
    fp1 = f[2:-1]
    fp2 = f[3:]

    # compute coefficients
    c0 = f00
    c1 = 0.5*(-fm1+fp1)
    c2 = 0.5*(2*fm1-5*f00+4*fp1-fp2)
    c3 = 0.5*(-fm1+3*f00-3*fp1+fp2)

    return c0, c1, c2, c3

def cubic_spline_interp(x, xp, fp, extrap=False):
    n, = xp.shape

    # fit spline coefficients
    c0, c1, c2, c3 = cubic_spline_fit(xp, fp)

    # find upper and lower bin
    i1 = np.clip(np.searchsorted(xp, x), 2, n-2)
    i0 = i1 - 1
    x0, x1 = xp[i0], xp[i1]

    # get normalized coordinates
    t0 = (x-x0)/(x1-x0)
    tc = np.clip(t0, 0, 1)
    t = np.where(extrap, t0, tc)

    # compute cubic polynomial
    ic = i0 - 1 # since we lose the first bin
    d0, d1, d2, d3 = c0[ic], c1[ic], c2[ic], c3[ic]
    ft = d0 + d1*t + d2*t**2 + d3*t**3

    return ft
