from operator import mul
from itertools import accumulate
from collections import OrderedDict
from inspect import signature
import toml

import jax
import jax.lax as lax
import jax.numpy as np
import jax.tree_util as tree

##
## indexing tricks
##

def get_strides(shape):
    return tuple(accumulate(shape[-1:0:-1], mul, initial=1))[::-1]

# index: [N, K] matrix or tuple of K [N] vectors
def ravel_index(index, shape):
    idxmat = np.stack(index, axis=-1)
    stride = np.array(get_strides(shape))
    return np.dot(idxmat, stride)

def ensure_tuple(x):
    if type(x) not in (tuple, list):
        return (x,)
    else:
        return x

##
## special functions
##

# classic bounded smoothstep
def smoothstep(x):
    return np.where(x > 0, np.where(x < 1, 3*x**2 - 2*x**3, 1), 0)

##
## parameters
##

# (-∞,∞) → (-∞,∞)
def ident(x):
    return x

# (-∞,∞) → (0,1)
def logit(x):
    return 1/(1+np.exp(-x))

# (0,1) → (-∞,∞)
def rlogit(x):
    return np.log(x/(1-x))

# complex spec types
class SpecType:
    pass

class SpecRange(SpecType):
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1

    def encode(self, x):
        return rlogit((x-self.x0)/(self.x1-self.x0))

    def decode(self, x):
        return self.x0 + (self.x1-self.x0)*logit(x)

# returns encode/decode pair
def spec_funcs(s):
    if s == 'ident':
        return ident, ident
    elif s == 'log':
        return np.log, np.exp
    elif s == 'logit':
        return rlogit, logit
    elif isinstance(s, SpecType):
        return s.encode, s.decode
    else:
        return s

def argpos(fun, arg):
    sig = signature(fun)
    t = type(arg)
    if t is str:
        names = list(sig.parameters)
        pos = names.index(arg)
    elif t is int:
        n = len(sig.parameters)
        pos = arg + n if arg < 0 else arg
    return pos

# encode: map from true space to free space
# decode: map from free space to true space
class Spec:
    def __init__(self, spec):
        self.spec = tree.tree_map(spec_funcs, spec)

    def encode(self, x):
        return tree.tree_map(lambda v, s: s[0](v), x, self.spec)

    def decode(self, x):
        return tree.tree_map(lambda v, s: s[1](v), x, self.spec)

    def decoder(self, fun0, arg=0):
        pos = argpos(fun0, arg)
        def fun1(*args, **kwargs):
            args1 = (
                self.decode(a) if i == pos else a for i, a in enumerate(args)
            )
            return fun0(*args1, **kwargs)
        return fun1

    def decodify(self, fun=None, arg=0):
        if fun is None:
            def decor(fun0):
                return self.decoder(fun0, arg)
            return decor
        else:
            return self.decoder(fun, arg)

##
## function tools
##

def partial(fun, *args, argnums=None):
    nargs = len(args)
    if argnums is None:
        argnums = list(range(nargs))
    elif type(argnums) is int:
        argnums = [argnums]
    assert(nargs == len(argnums))

    def fun1(*args1, **kwargs1):
        ntot = nargs + len(args1)
        idx = [i for i in range(ntot) if i not in argnums]
        idx1 = {j: i for i, j in enumerate(idx1)}
        args2 = [args[i] if i in argnums else args1[idx1[i]]]
        return fun(*args2, **kwargs1)

    return fun1

##
## control flow
##

def iterate_while(f, x0, K):
    z0 = 0, x0
    cond = lambda z: z[0] < K
    func = lambda z: (z[0] + 1, f(z[1]))
    _, x1 = lax.while_loop(cond, func, z0)
    return x1

# state-only iteration - this works better with differentiation
def iterate_scan(f, x0, K, hist=False):
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
def lambdify(x):
    return (lambda: x)

def blank_arg(f):
    return lambda _: f()

def where(sel, val_true, val_false):
    return lax.cond(sel, lambda: val_true, lambda: val_false)

def choose(sel, vals):
    return lax.switch(sel, [lambdify(x) for x in vals])

def switch(sel, paths):
    return lax.switch(sel, [blank_arg(p) for p in paths], None)

##
## trees
##

# tree_where?
