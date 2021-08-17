from math import prod

import jax
import jax.numpy as np
import jax.lax as lax

from .tools import get_strides, ravel_index, ensure_tuple, smoothstep

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
# x  : [N..., M...]
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
def interp(x, iv, extrap=True):
    iv = ensure_tuple(iv)

    if extrap:
        i0 = [
            np.clip(np.floor(i), 0, n-2).astype(np.int32)
            for i, n in zip(iv, x.shape)
        ]
    else:
        iv = [np.clip(i, 0, n-1) for i, n in zip(iv, x.shape)]
        i0 = [np.floor(i).astype(np.int32) for i in iv]

    y0 = x[tuple(i0)]

    vr = y0

    for k in range(x.ndim):
        i0k, ivk = i0[k], iv[k]
        i1 = [i0k + 1 if j == k else ix for j, ix in enumerate(i0)]

        y1 = x[tuple(i1)]
        vr += (ivk-i0k) * (y1-y0)

    return vr

# find the continuous (linearly interpolated) index of value v on grid g
# requires monotonically incrasing g
# inverse of interp, basically
def grid_index(g, v, extrap=False):
    n = g.size
    gx = g.reshape((n,)+(1,)*v.ndim)
    vx = v.reshape((1,)+v.shape)

    i = np.sum(vx >= gx, axis=0)
    i1 = np.clip(i, 1, n-1)
    i0 = i1 - 1

    g0, g1 = g[i0], g[i1]
    x0 = (v-g0)/(g1-g0)
    xc = np.clip(x0, 0, 1)
    x = np.where(extrap, x0, xc)

    return i0 + x

##
## solvers
##

def solve_binary(f, x0, x1, K=10):
    zx0, zx1 = x0, x1
    zf0, zf1 = f(zx0), f(zx1)
    sel = zf0 < zf1

    x0 = np.where(sel, zx0, zx1)
    x1 = np.where(sel, zx1, zx0)
    f0 = np.where(sel, zf0, zf1)
    f1 = np.where(sel, zf1, zf0)

    for k in range(K):
        xm = 0.5*(x0+x1)
        fm = f(xm)
        sel = fm > 0

        x0 = np.where(sel, x0, xm)
        x1 = np.where(sel, xm, x1)
        f0 = np.where(sel, f0, fm)
        f1 = np.where(sel, fm, f1)

    return xm

def solve_newton(f, df, x, K=10):
    for k in range(K):
        vf, vdf = f(x), df(x)
        x = x - vf/vdf
    return x

def solve_combined(f, df, x0, x1, K=10):
    x = solve_binary(f, x0, x1, K=K)
    x = solve_newton(f, df, x, K=K)
    return x

##
## continuous optimize
##

def optim_secant(df, x0, x1, K=5):
    for k in range(K):
        d0, d1 = df(x0), df(x1)
        dd = d1 - d0
        dx = np.where(dd == 0, 0.5*(x1-x0), d1*(x1-x0)/dd)
        x2 = x1 - dx
        x0, x1 = x1, x2
    return x1

def optim_newton(df, ddf, x, clip=0.0, K=5):
    for k in range(K):
        dv, ddv = df(x), ddf(x)
        d0 = -dv/ddv
        dc = np.clip(d0, -clip, clip)
        d = np.where(clip > 0.0, dc, d0)
        x = x + d
    return x

def optim_grad(df, x, step=0.01, clip=0.0, K=10):
    for k in range(K):
        d0 = df(x)
        dc = np.clip(d0, -clip, clip)
        d = np.where(clip > 0.0, dc, d0)
        x = x + step*d
    return x

##
## iteration
##

# state-only iteration
def iterate(func, st0, T, hist=True):
    tvec = np.arange(T)
    dub = lambda x, _: 2*(func(x),)
    last, path = lax.scan(dub, st0, tvec)
    if hist:
        return path
    else:
        return last

# simulate from tree of differential maps
def simdiff(func, st0, Δ, T, hist=True):
    up = lambda x, d: x + Δ*d
    def update(st):
        dst = func(st)
        stp = jax.tree_map(up, st, dst)
        return stp
    return iterate(update, st0, T, hist=hist)
