import jax.numpy as np

# easily vmap'able version with scalar v
def grid_index(g, v, extrap=False):
    n = g.size

    i = np.sum(v >= g)
    i1 = np.clip(i, 1, n-1)
    i0 = i1 - 1

    g0, g1 = g[i0], g[i1]
    x0 = (v-g0)/(g1-g0)
    xc = np.clip(x0, 0, 1)
    x = np.where(extrap, x0, xc)

    return i0 + x

