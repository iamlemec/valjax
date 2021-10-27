from math import prod
from operator import mul, add

import jax
import jax.numpy as np
import jax.lax as lax
from jax.tree_util import tree_map, tree_reduce

from .tools import ravel_index, ensure_tuple

##
## constants
##

ϵ = 1e-7

##
## indexing routines
##

# for each element in left space (N), take elements in right space (M)
# includes bounds checking
# x  : [N..., M...]
# iv : tuple of len(M) [N...]
# ret: [N...]
def address0(x, iv):
    iv = ensure_tuple(iv)

    K = len(iv)
    sN, sM = x.shape[:-K], x.shape[-K:]
    lN, lM = prod(sN), prod(sM)

    ic = [np.clip(i, 0, n-1) for i, n in zip(iv, sM)]
    ix = lM*np.arange(lN) + ravel_index(ic, sM).flatten()

    y0 = np.take(x, ix)
    y = np.reshape(y0, sN)

    return y

# generalized axis
def address(x, iv, axis):
    iv = ensure_tuple(iv)
    axis = ensure_tuple(axis)

    # shuffle axes to end
    K = len(axis)
    end = range(-K, 0)
    xs = np.moveaxis(x, axis, end)

    # perform on last axis
    y = address0(xs, iv)

    return y

##
## discrete maximizers
##

# multi-dimensional argmax
def argmax(x, axis):
    axis = ensure_tuple(axis)

    # shuffle axes to end
    K = len(axis)
    end = range(-K, 0)
    xs = np.moveaxis(x, axis, end)

    # flatten max axes
    sN, sK = xs.shape[:-K], xs.shape[-K:]
    xf = np.reshape(xs, sN+(-1,))

    # get maxes and reshape
    i0 = np.argmax(xf, axis=-1)
    iv = np.unravel_index(i0, sK)

    return iv

# get the smooth index of the maximum (quadratic)
def smoothmax(x, axis):
    axis = ensure_tuple(axis)

    i0 = argmax(x, axis)
    y0 = address(x, i0, axis)
    ir = []

    for k, ax in enumerate(axis):
        n = x.shape[ax]

        im = [ix - 1 if j == k else ix for j, ix in enumerate(i0)]
        ip = [ix + 1 if j == k else ix for j, ix in enumerate(i0)]

        ym = address(x, im, axis)
        yp = address(x, ip, axis)

        dm = y0 - ym
        dp = y0 - yp

        ik0 = i0[k] + (1/2)*(dm-dp)/(dm+dp)
        ik = np.clip(ik0, 0, n-1)
        ir.append(ik)

    return tuple(ir)

##
## interpolation
##

# interpolate a continuous index (linear) along given axes
# y  : [N..., M...]
# iv : tuple of len(M) [N...]
# ret: [N...]
def interp_address(y, iv, axis, extrap=True):
    iv = ensure_tuple(iv)
    axis = ensure_tuple(axis)
    shape = [y.shape[a] for a in axis]

    if extrap:
        i0 = [
            np.clip(np.floor(i), 0, n-2).astype(np.int32)
            for i, n in zip(iv, shape)
        ]
    else:
        iv = [np.clip(i, 0, n-1) for i, n in zip(iv, shape)]
        i0 = [np.floor(i).astype(np.int32) for i in iv]

    y0 = address(y, i0, axis)

    vr = y0

    for k, ax in enumerate(axis):
        i0k, ivk = i0[k], iv[k]
        i1 = [i0k + 1 if j == k else ix for j, ix in enumerate(i0)]

        y1 = address(y, i1, axis)
        vr += (ivk-i0k) * (y1-y0)

    return vr

# akin to continuous array indexing
# requires len(iv) == x.ndim
# return shape is that of iv components
# this is vmap-able
def interp_index(g, iv, extrap=True):
    iv = ensure_tuple(iv)

    if extrap:
        i0 = [
            np.clip(np.floor(i), 0, n-2).astype(np.int32)
            for i, n in zip(iv, g.shape)
        ]
    else:
        iv = [np.clip(i, 0, n-1) for i, n in zip(iv, g.shape)]
        i0 = [np.floor(i).astype(np.int32) for i in iv]

    y0 = g[tuple(i0)]

    vr = y0

    for k in range(g.ndim):
        i0k, ivk = i0[k], iv[k]
        i1 = [i0k + 1 if j == k else ig for j, ig in enumerate(i0)]

        y1 = g[tuple(i1)]
        vr += (ivk-i0k) * (y1-y0)

    return vr

# find the continuous (linearly interpolated) index of value v on grid g
# requires monotonically incrasing g
# inverse of interp, basically
def grid_index(g, v, extrap=True):
    n = g.size
    v = np.asarray(v)

    gx = g.reshape((n,)+(1,)*v.ndim)
    vx = v.reshape((1,)+v.shape)

    it = np.sum(vx >= gx, axis=0)
    i1 = np.clip(it, 1, n-1)
    i0 = i1 - 1

    g0, g1 = g[i0], g[i1]
    f0 = (v-g0)/(g1-g0)
    ir = i0 + f0

    ic = np.clip(ir, 0, n-1)
    i = np.where(extrap, ir, ic)

    return i

# only 1d interp (x0.ndim == 1)
# output shape is x.shape
def interp(x0, y0, x, extrap=True):
    i = grid_index(x0, x, extrap=extrap)
    y = interp_index(y0, i, extrap=extrap)
    return y
