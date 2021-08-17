from operator import mul
from itertools import accumulate

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

# encode: map from true space to free space
# decode: map from free space to true space
class Spec:
    def __init__(self, spec):
        self.spec = tree.tree_map(spec_funcs, spec)

    def encode(self, x):
        return tree.tree_map(lambda v, s: s[0](v), x, self.spec)

    def decode(self, x):
        return tree.tree_map(lambda v, s: s[1](v), x, self.spec)
