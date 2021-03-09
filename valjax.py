from math import prod
from operator import mul
from itertools import accumulate

import jax.numpy as npx

def get_strides(shape):
    return tuple(accumulate(shape[-1:0:-1], mul, initial=1))[::-1]

# index: [N, K] matrix or tuple of K [N] vectors
def ravel_index(index, shape, np=npx):
    idxmat = np.stack(index, axis=-1)
    stride = np.array(get_strides(shape))
    return np.dot(idxmat, stride)

def ensure_tuple(x):
    if type(x) not in (tuple, list):
        return (x,)
    else:
        return x

# for each element in left space (N), take elements in right space (M)
# includes bounds checking
# x  : [N..., M...]
# iv : tuple of len(M) [N...]
# ret: [N...]
def address0(x, iv, np=npx):
    iv = ensure_tuple(iv)

    K = len(iv)
    sN, sM = x.shape[:-K], x.shape[-K:]
    lN, lM = prod(sN), prod(sM)

    ic = [np.clip(i, 0, n-1) for i, n in zip(iv, sM)]
    ix = lM*np.arange(lN) + ravel_index(ic, sM, np=np).flatten()

    y0 = np.take(x, ix)
    y = np.reshape(y0, sN)

    return y

# generalized axis
def address(x, iv, axis, np=npx):
    iv = ensure_tuple(iv)
    axis = ensure_tuple(axis)

    # shuffle axes to end
    K = len(axis)
    end = range(-K, 0)
    xs = np.moveaxis(x, axis, end)

    # perform on last axis
    y = address0(xs, iv, np=np)

    return y

# multi-dimensional argmax
def argmax(x, axis, np=npx):
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
def smoothmax(x, axis, np=npx):
    axis = ensure_tuple(axis)

    i0 = argmax(x, axis, np=np)
    y0 = address(x, i0, axis, np=np)
    ir = []

    for k, ax in enumerate(axis):
        n = x.shape[ax]

        im = [ix - 1 if j == k else ix for j, ix in enumerate(i0)]
        ip = [ix + 1 if j == k else ix for j, ix in enumerate(i0)]

        ym = address(x, im, axis, np=np)
        yp = address(x, ip, axis, np=np)

        dm = y0 - ym
        dp = y0 - yp

        ik0 = i0[k] + (1/2)*(dm-dp)/(dm+dp)
        ik = np.clip(ik0, 0, n-1)
        ir.append(ik)

    return tuple(ir)

# interpolate a continuous index (linear) along given axes
# x  : [N..., M...]
# iv : tuple of len(M) [N...]
# ret: [N...]
def interp_address(y, iv, axis, extrap=True, np=npx):
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

    y0 = address(y, i0, axis, np=np)

    vr = y0

    for k, ax in enumerate(axis):
        i0k, ivk = i0[k], iv[k]
        i1 = [i0k + 1 if j == k else ix for j, ix in enumerate(i0)]

        y1 = address(y, i1, axis, np=np)
        vr += (ivk-i0k) * (y1-y0)

    return vr

# akin to continuous array indexing
# requires len(iv) == x.ndim
# return shape is that of iv's
def interp(x, iv, extrap=True, np=npx):
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
