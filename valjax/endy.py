from math import prod
from operator import mul, add
from inspect import signature

import jax
import jax.numpy as np
import jax.lax as lax
from jax.tree_util import tree_map, tree_reduce

from .tools import get_strides, ravel_index, ensure_tuple, smoothstep

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

##
## differentiable solvers
##

def step_binary(f, x0, x1):
    xm = 0.5*(x0+x1)
    sel = f(xm) > 0
    x0 = np.where(sel, x0, xm)
    x1 = np.where(sel, xm, x1)
    return x0, x1

def solve_binary(f, x0, x1, K=10):
    sel = f(x0) < f(x1)
    xi0 = np.where(sel, x0, x1)
    xi1 = np.where(sel, x1, x0)

    xf0, xf1 = iterate_while(
        lambda x: step_binary(f, *x),
        (xi0, xi1), K
    )

    return 0.5*(xf0+xf1)

def solve_newton(f, df, x0, K=10):
    return iterate_while(
        lambda x: x - f(x)/df(x),
        x0, K
    )

def solve_combined(f, df, x0, x1, K=10):
    x = solve_binary(f, x0, x1, K=K)
    x = solve_newton(f, df, x, K=K)
    return x

# assumes max_x f(x, *α) form, x scalar
def solver_diff(f, x0, x1, mode='rev', K=10):
    sig = signature(f)
    nargs = len(sig.parameters)
    arg_idx = tuple(range(1, nargs))

    f_x = jax.grad(f, argnums=0)
    f_α = jax.grad(f, argnums=arg_idx)

    def solve(*α):
        return solve_combined(
            lambda x: f(x, *α),
            lambda x: f_x(x, *α),
            x0, x1, K=K
        )

    def dsolve(x, *α):
        return tree_map(lambda a: -a/f_x(x, *α), f_α(x, *α))

    if mode in ('f', 'fwd', 'forward'):
        def solve_jvp(α, dα):
            x = solve(*α)
            ds = dsolve(x, *α)
            primal_out = x
            tangent_out = tree_reduce(add, tree_map(mul, ds, dα))
            return primal_out, tangent_out

        solve1 = jax.custom_jvp(solve)
        solve1.defjvp(solve_jvp)
    elif mode in ('r', 'rev', 'reverse'):
        def solve_fwd(*α):
            x = solve(*α)
            return x, (α, x)

        def solve_bwd(res, g):
            α, x = res
            ds = dsolve(x, *α)
            return tree_map(lambda v: v*g, ds)

        solve1 = jax.custom_vjp(solve)
        solve1.defvjp(solve_fwd, solve_bwd)

    return solve1

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
## control flow
##

def iterate_while(f, x0, K):
    z0 = 0, x0
    cond = lambda z: z[0] < K
    func = lambda z: (z[0] + 1, f(z[1]))
    _, x1 = lax.while_loop(cond, func, z0)
    return x1

# state-only iteration
def iterate_scan(f, x0, K, hist=True):
    kvec = np.arange(K)
    dub = lambda x, _: 2*(f(x),)
    x1, xh = lax.scan(dub, x0, kvec)
    if hist:
        return xh
    else:
        return x1

# simulate from tree of differential maps
def simdiff(func, st0, Δ, T, hist=True):
    up = lambda x, d: x + Δ*d
    def update(st):
        dst = func(st)
        stp = tree_map(up, st, dst)
        return stp
    return iterate_scan(update, st0, T, hist=hist)

# simplify lax switch slightly
def blank_arg(f):
    return lambda _: f()

def switch(sel, paths):
    return lax.switch(sel, [blank_arg(p) for p in paths], None)
