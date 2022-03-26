# differentiable homotopy

import jax
import jax.numpy as np
import jax.lax as lax
from functools import partial

##
## tools
##

def linear_path(p0, p1):
    return lambda t: t*p1 + (1-t)*p0

##
## HOM-HOM-HOM-HOM HOMPACK90 STYLE
##

def _row_dets_map(dets, tv):
    i, qrow = tv
    n, = qrow.shape
    d = -np.dot(qrow, dets)
    isel = np.arange(n) == i
    dets1 = np.where(isel, d, dets)
    return dets1, d

def row_dets(mat):
    n, m = mat.shape
    qr = np.linalg.qr(mat, 'r')
    msel = np.arange(m) == m - 1
    dets0 = np.where(msel, 1.0, 0.0)
    ivec = np.arange(m-2, -1, -1)
    dets, _ = lax.scan(_row_dets_map, dets0, (ivec, qr))
    qdiag = np.hstack([np.diag(qr[:,:-1]), 1.0])
    dets1 = np.sign(np.prod(qdiag))*(dets/qdiag) # just for the sign
    # dets1 = np.prod(qdiag)*dets # to get the scale exactly
    return dets1

##
## homotopy path
##

def _homotopy_step(f, fx, fp, p, dp, Δ, st, tv):
    x0, t0 = st
    p0 = p(t0)

    # compute values
    fx_val = fx(x0, p0)
    fp_val = fp(x0, p0)
    dp_val = dp(t0)

    # prediction step
    tdir_val = np.dot(fp_val, dp_val)
    jac_val = np.hstack([fx_val, tdir_val[:, None]])
    step_pred = row_dets(jac_val)

    # normalize and direct step
    bound = (t0 >= 0.0) & (t0 <= 1.0)
    Δ1 = np.where(bound, Δ, 0.0)
    step_pred = step_pred*(Δ1/np.mean(np.abs(step_pred)))
    step_pred = step_pred*np.sign(step_pred[-1])
    dxp, dtp = step_pred[:-1], step_pred[-1]

    # apply prediction
    x1 = x0 + dxp
    t1 = t0 + dtp
    p1 = p(t1)

    # compute values
    f_val = f(x1, p1)
    fx_val = fx(x1, p1)
    fp_val = fp(x1, p1)
    dp_val = dp(t1)

    # correction step
    proj_dir = np.hstack([np.zeros_like(x1), 1.0]) # step_pred
    tdir_val = np.dot(fp_val, dp_val)
    jac_val = np.hstack([fx_val, tdir_val[:, None]])
    proj_val = np.vstack([jac_val, proj_dir])
    step_corr = -np.linalg.solve(proj_val, np.hstack([f_val, 0.0]))
    dxc, dtc = step_corr[:-1], step_corr[-1]

    # apply correction
    x2 = x1 + dxc
    t2 = t1 + dtc
    p2 = p(t2)

    # return state/hist
    st1 = x2, t2
    hst = x2, p2, t2
    return st1, hst

def homotopy(f, p, x0, Δ=0.01, K=100):
    p = linear_path(*p) if type(p) is tuple else p
    fx = jax.jacobian(f, argnums=0, holomorphic=True)
    fb = jax.jacobian(f, argnums=1, holomorphic=True)
    dp = jax.jacobian(p, holomorphic=True)

    hom_step = partial(_homotopy_step, f, fx, fb, p, dp, Δ)
    state0, tvec = (x0, 0.0j), np.arange(K)
    final, hist = lax.scan(hom_step, state0, tvec)

    return hist
