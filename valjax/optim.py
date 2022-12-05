import jax
import jax.numpy as np
import jax.lax as lax
from jax.tree_util import tree_map, tree_reduce
from inspect import signature

from .tools import iterate_scan, iterate_while

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

    xf0, xf1 = iterate_scan(
        lambda x: step_binary(f, *x),
        (xi0, xi1), K
    )

    return 0.5*(xf0+xf1)

def solve_newton(f, df, x0, K=10):
    return iterate_scan(
        lambda x: x - f(x)/df(x),
        x0, K
    )

def solve_combined(f, df, x0, x1, K=10):
    x = solve_binary(f, x0, x1, K=K)
    x = solve_newton(f, df, x, K=K)
    return x

# assumes f(x, *α) form, x scalar
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
## continuous optimize (maximizers)
##

def step_secant(df, x0, x1):
    d0, d1 = df(x0), df(x1)
    dd = d1 - d0
    dx = np.where(dd == 0, 0.5, d1/dd)*(x1-x0)
    x2 = x1 - dx
    return x1, x2

def optim_secant(df, x0, x1, K=10):
    xa, xb = iterate_scan(
        lambda x: step_secant(df, *x),
        (x0, x1), K
    )
    return 0.5*(xa+xb)

def optim_newton(df, ddf, x0, K=10):
    return iterate_scan(
        lambda x: x - df(x)/ddf(x),
        x0, K
    )

def optim_grad(df, x0, step=0.01, K=10):
    return iterate_scan(
        lambda x: x + step*df(x),
        x0, K
    )

def optimer_diff(f, x0, x1, mode='rev', K=10):
    df = jax.grad(f)
    return solver_diff(df, x0, x1, mode=mode, K=K)
