from operator import mul
from itertools import accumulate
from collections import OrderedDict
from inspect import signature
import toml

import jax
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

# returns encode/decode pair
def spec_funcs(s):
    if s == 'ident':
        return ident, ident
    elif s == 'log':
        return np.log, np.exp
    elif s == 'logit':
        return rlogit, logit
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

# def partial(f, *args0, right=True):
#     if right:
#         f1 = lambda *args: f(*args, *args0)
#     else:
#         f1 = lambda *args: f(*args0, *args)
#     return f1
