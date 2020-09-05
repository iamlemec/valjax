import jax
import jax.numpy as np

from math import prod
from operator import mul
from itertools import accumulate

def get_strides(shape):
    return tuple(accumulate(shape[-1:0:-1], mul, initial=1))[::-1]

# index: [N, K] matrix or tuple of K [N] vectors
def ravel_index(index, shape):
    idxmat = np.stack(index, axis=-1)
    stride = np.array(get_strides(shape))
    return np.dot(idxmat, stride)

# take element by element on axis
# includes bounds checking
# x: [N..., M...]
# i: tuple of len(M) [N...]
# r: [N...]
def address0(x, iv):
    K = len(iv)
    sN, sM = x.shape[:-K], x.shape[-K:]
    lN, lM = prod(sN), prod(sM)

    ic = [np.clip(i, 0, n-1) for i, n in zip(iv, sM)]
    ix = lM*np.arange(lN) + ravel_index(ic, sM)

    y0 = np.take(x, ix)
    y = np.reshape(y0, sN)

    return y

# generalized axis
def address(x, iv, axis):
    K = len(axis)
    end = range(-K, 0)
    xs = np.moveaxis(x, axis, end)
    y = address0(xs, iv)
    return y

# multi-dimensional argmax
def argmax(x, axis):
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

# interpolate a continuous index (linear)
def interp(y, iv, axis):
    i0 = [np.floor(i).astype(np.int32) for i in iv]
    y0 = address(y, i0, axis)

    vr = y0

    for k, ax in enumerate(axis):
        n = y.shape[ax]
        i0k, ivk = i0[k], iv[k]

        i1 = [i0k + 1 if j == k else ix for j, ix in enumerate(i0)]
        y1 = address(y, i1, axis=axis)

        vr += (ivk-i0k) * (y1-y0)

    return vr
