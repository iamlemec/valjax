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

def cubic_spline_fit(x, f):
    # compute bins for interior
    p0, p1 = f[:-1], f[1:]
    m0 = f[1:-1] - f[:-2]
    m1 = f[2:] - f[1:-1]

    # handle edge bins
    m0 = np.hstack([m0[0], m0])
    m1 = np.hstack([m1, m1[-1]])

    # compute coefficients
    c0, c1 = p0, m0
    c2 = -3*p0 + 3*p1 - 2*m0 - m1
    c3 = 2*p0 + m0 - 2*p1 + m1

    return c0, c1, c2, c3

def cubic_spline_interp(x, xp, fp, extrap=False):
    n, = xp.shape

    # fit spline coefficients
    c0, c1, c2, c3 = cubic_spline_fit(xp, fp)

    # find upper and lower bin
    i1 = np.clip(np.searchsorted(xp, x), 1, n-1)
    i0 = i1 - 1
    x0, x1 = xp[i0], xp[i1]

    # get normalized coordinates
    t0 = (x-x0)/(x1-x0)
    tc = np.clip(t0, 0, 1)
    t = np.where(extrap, t0, tc)

    # compute cubic polynomial
    d0, d1, d2, d3 = c0[i0], c1[i0], c2[i0], c3[i0]
    ft = d0 + d1*t + d2*t**2 + d3*t**3

    return ft
