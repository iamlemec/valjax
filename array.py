import jax
import jax.numpy as np
from math import prod

# take element by element on axis=-1
# x: [N..., K]
# i: [N...]
# r: [N...]
def address0(x, i):
    sN = prod(i.shape)
    sK = x.shape[-1]
    ic = sK*np.arange(sN) + i.flatten()
    zc = np.take(x, ic)
    return zc.reshape(i.shape)

# generalized axis
def address(x, i, axis=-1):
    x1 = x.swapaxes(-1, axis)
    y = address0(x1, i)
    return y

# multi-dimensional argmax
def argmax(x, axis=-1):
    # take care of singleton case
    if type(axis) is int:
        return np.argmax(x, axis=axis)

    # shuffle axes to end
    K = len(axis)
    xs = np.moveaxis(x, axis, range(-K, 0))

    # flatten max axes
    sN = xs.shape[:-K]
    sK = xs.shape[-K:]
    xf = np.reshape(xs, sN+(-1,))

    # get maxes and reshape
    i0 = np.argmax(xf, axis=-1)
    iv = np.unravel_index(i0, sK)

    return iv

# get the smooth index of the maximum (quadratic)
def smoothmax(x, δ, axis=-1):
    i0 = argmax(x, axis=axis)

    n = x.shape[axis]
    im = np.clip(i0-1, 0, n-1)
    ip = np.clip(i0+1, 0, n-1)

    y0 = address(y, i0, axis=axis)
    ym = address(y, im, axis=axis)
    yp = address(y, ip, axis=axis)

    dm = y0 - ym
    dp = y0 - yp

    return i0 + (1/2)*(dm-dp)/(dm+dp)

# interpolate a continuous index
def interp(y, i, axis=-1):
    n = y.shape[0]

    i0 = np.floor(i).astype(np.int32)
    ilo = np.clip(i0, 0, n-1)
    ihi = np.clip(ilo + 1, 0, n-1)
    t = i - ilo

    ylo = address(y, ilo, axis=axis)
    yhi = address(y, ihi, axis=axis)

    return t*yhi + (1-t)*ylo
