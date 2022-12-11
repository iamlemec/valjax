# finite-horizon value function optimization

import valjax as vj

import jax
import jax.numpy as np
import jax.lax as lax

# algo params
alg0 = {
    'N': 100, # grid size
    'flo': 0.2, # k grid min
    'fhi': 1.5, # k grid max
    'ε': 1e-6 # consumption min
}

# simple parameters
par0 = {
    'β': 0.95, # discount rate
    'δ': 0.1, # depreciation rate
    'α': 0.35, # capital exponent
    'z': 1.0, # productivity
}

class Capital:
    def __init__(self, par, **alg):
        self.__dict__.update({**alg0, **alg, **par0, **par})
        self.calc_kss()
        self.make_grid()
        self.compile_funcs()

    # find steady state
    def calc_kss(self):
        rhs = (1-self.β)/self.β + self.δ
        self.kss = (self.α*self.z/rhs)**(1/(1-self.α))

    # construct capital grid
    def make_grid(self):
        self.klo, self.khi = self.flo*self.kss, self.fhi*self.kss
        self.k_grid = np.linspace(self.klo, self.khi, self.N)
        self.y_grid = self.prod(self.k_grid)
        self.yd_grid = self.y_grid + (1-self.δ)*self.k_grid

    # vmap and jit functions
    def compile_funcs(self):
        self.spec = vj.Spec(vj.SpecRange(self.klo, self.khi))
        self.l_bellman = self.spec.decoder(self.bellman, arg='kp')
        self.dl_bellman = jax.grad(self.l_bellman, argnums=0)
        self.v_bellman = jax.vmap(self.bellman, in_axes=(0, 0, None))
        self.v_policy = jax.vmap(self.policy, in_axes=(0, 0, None))
        self.j_value_solve = jax.jit(self.value_solve, static_argnames='K')

    # utility function
    def util(self, c):
        u0 = np.log(self.ε) + (c/self.ε-1)
        u1 = np.log(np.maximum(self.ε, c))
        return np.where(c >= self.ε, u1, u0)

    # production function
    def prod(self, k):
        return self.z*(k**self.α)

    # compute bellman update
    def bellman(self, kp, yd, v):
        vp = vj.cubic_spline_interp(kp, self.k_grid, v)
        # vp = np.interp(kp, self.k_grid, v)
        up = self.util(yd-kp)
        return up + self.β*vp

    # find optimal policy
    def policy(self, kp0, yd, v):
        lkp0 = self.spec.encode(kp0)
        dopt = lambda lkp: self.dl_bellman(lkp, yd, v)
        lkp1 = vj.optim_grad(dopt, lkp0, step=0.01, K=5)
        kp1 = self.spec.decode(lkp1)
        return kp1

    # one value iteration
    def value_step(self, v, kp):
        # calculate updates
        v1 = self.v_bellman(kp, self.yd_grid, v)
        kp1 = self.v_policy(kp, self.yd_grid, v)

        # compute output
        err = np.max(np.abs(v1-v))
        out = {'v': v1, 'kp': kp1, 'err': err}

        # return value
        return (v1, kp1), out

    # solve for value function
    def value_solve(self, K=100):
        # initial guess
        v0 = self.util(self.yd_grid-self.δ*self.k_grid)
        kp0 = 0.2*self.kss + 0.8*self.k_grid

        # run bellman iterations
        upd = lambda x, t: self.value_step(*x)
        (v1, kp1), hist = lax.scan(upd, (v0, kp0), np.arange(K))

        # return full info
        _, ret = self.value_step(v1, kp1)
        return v1, kp1, ret, hist

##
## main
##

## solve it
# mod = Model(par, **alg)
# v1, kp1, ret, hist = mod.j_solve_value()

## plot it
# fig, ax = plt.subplots()
# ax.plot(mod.k_grid, kp1-mod.k_grid)
# ax.hlines(0, *ax.get_xlim(), linewidth=1, linestyle='--', color='black')
# ax.scatter(kss, 0, color='black', zorder=10)
