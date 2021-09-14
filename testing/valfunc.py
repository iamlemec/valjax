# finite-horizon value function optimization

import valjax.endy as vj
import valjax.tools as vt

import jax
import jax.numpy as np
import jax.lax as lax

# algo params
alg0 = {
    'T': 30, # time periods
    'N': 100, # grid size
    'k_lo': 0.1, # k grid min
    'k_hi': 3.0, # k grid max
}

# simple parameters
par0 = {
    'β': 0.95, # discount rate
    'δ': 0.1, # depreciation rate
    'α': 0.35, # capital exponent
    'z': 1.0, # productivity
}

class Model:
    def __init__(self, alg=alg0):
        self.spec = vt.Spec('logit')
        self.update_alg(alg)
        self.compile_funcs()

    def update_alg(self, alg):
        self.alg = alg.copy()
        self.make_kgrid()

    # construct capital grid
    def make_kgrid(self):
        N, k_lo, k_hi = self.alg['N'], self.alg['k_lo'], self.alg['k_hi']
        self.k_grid = np.linspace(k_lo, k_hi, N)

    # vmap and jit functions
    def compile_funcs(self):
        self.l_invest = self.spec.decoder(self.invest, arg='sp')
        self.dl_invest = jax.grad(self.l_invest, argnums=-1)
        self.v_invest_opt = jax.vmap(self.invest_opt, in_axes=[None, None, 0, 0])
        self.j_solve = jax.jit(self.solve)

    # find steady state
    def get_kss(self, par):
        β, δ, z, α = par['β'], par['δ'], par['z'], par['α']
        rhs = (1-β)/β + δ
        kss = (α*z/rhs)**(1/(1-α))
        return kss

    # defined functions
    def util(self, c):
        ϵ = self.alg['ϵ']
        u0 = np.log(ϵ) + (c/ϵ-1)
        u1 = np.log(np.maximum(ϵ, c))
        return np.where(c >= ϵ, u1, u0)

    def prod(self, z, α):
        return z*self.k_grid**α

    def invest(self, kg, vn1, yd, sp):
        kp = sp*yd
        vv = vj.interp(kg, vn1, kp)
        uv = self.util(yd-kp)
        return uv + vv

    def invest_opt(self, kg, vn1, yd, sp0):
        lp0 = self.spec.encode(sp0)
        dopt = jax.partial(self.dl_invest, kg, vn1, yd)
        lp1 = vj.optim_grad(dopt, lp0, step=1.0, clip=1.0, K=10)
        sp1 = self.spec.decode(lp1)
        return sp1

    def value(self, par, grid, st, tv):
        β = par['β']
        kg = grid['kg']
        yd = grid['yd']
        vn = st['vn']

        # calculate optimal investment
        sp0 = 0.2*np.ones_like(kg)
        vn1 = β*vn
        sp = self.v_invest_opt(kg, vn1, yd, sp0)

        # compute actual values
        kp = sp*yd
        v = self.invest(kg, vn1, yd, sp)

        # compute update errors
        err = np.max(np.abs(v-vn))

        # return state and output
        stp = {
            'vn': v,
        }
        out = {
            'kp': kp,
            'v': v,
            'err': err,
        }

        return stp, out

    def solve(self, par):
        T = self.alg['T']
        α, δ, z = par['α'], par['δ'], par['z']

        # precompute grid values
        y_grid = self.prod(z, α)
        yd_grid = y_grid + (1-δ)*self.k_grid

        # partially apply grid
        grid = {
            'kg': self.k_grid,
            'yd': yd_grid,
        }
        value1 = jax.partial(self.value, par, grid)

        # scan over time (backwards)
        s0 = {
            'vn': self.util(y_grid),
        }
        tv = {
            't': np.arange(T)[::-1],
        }
        last, path = lax.scan(value1, s0, tv)

        return path

##
## main
##

## solve it
# mod = Model(alg0)
# kss = mod.get_kss(par0)
# ret = mod.j_solve(par0)

## plot it
# fig, ax = plt.subplots()
# ax.plot(mod.k_grid, sol['kp'][-1, :]-mod.k_grid)
# ax.scatter(kss, 0, color='black', zorder=10)
# ax.hlines(0, *ax.get_xlim(), linewidth=1, linestyle='--', color='black')
