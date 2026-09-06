"""
pcnn.py -- physics-constrained neural networks for steady incompressible
flow in JAX, after Sun, Gao, Pan & Wang (CMAME 2020).

The idea of the paper, and of this file:

  1. A small MLP maps coordinates (and optionally a geometry parameter)
     to raw outputs (n_u, n_v, n_p).
  2. Boundary conditions are enforced EXACTLY by construction, not by a
     penalty term: each field is written as
         field = particular_solution + distance_function * network_output
     where the distance function vanishes on the Dirichlet boundary.
  3. The only loss is the Navier-Stokes residual at random collocation
     points, evaluated with automatic differentiation. No simulation data.

Everything is per-point: derivatives are taken with jax.grad / jax.hessian
of scalar functions of a single coordinate vector, then vectorized over
the batch with jax.vmap. This is the correct way to get u_x, u_yy, ... in
JAX; taking jacfwd of a batched function gives cross-sample Jacobians.

Dependencies: jax, numpy (matplotlib for the examples).

Author: Aiden Azarnoush
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)


# ================================================================== MLP
def init_mlp(key, layer_sizes):
    """Xavier-initialized MLP parameters: list of (W, b)."""
    params = []
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        key, sub = jax.random.split(key)
        bound = jnp.sqrt(6.0 / (n_in + n_out))
        W = jax.random.uniform(sub, (n_in, n_out), minval=-bound, maxval=bound)
        params.append((W, jnp.zeros(n_out)))
    return params


def mlp(params, x):
    """Forward pass for ONE input vector x (shape (d,)). tanh hidden layers."""
    h = x
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b


# ============================================================= problems
class Problem:
    """Base class. Subclasses define the geometry, the sampler, and the
    hard boundary-condition construction in `fields`.

    fields(params, xi) -> (u, v, p) for a single input vector xi whose
    first two entries are (x, y); extra entries are parameters (e.g. the
    stenosis severity alpha).
    """
    rho = 1.0
    mu = 0.1
    n_inputs = 2

    def sample(self, n, rng):
        raise NotImplementedError

    def fields(self, params, xi):
        raise NotImplementedError


class Poiseuille(Problem):
    """Pressure-driven flow between plates y = 0 and y = H, x in [0, L].

    Hard constraints:
        u = y (H - y) n_u            (no-slip on both walls)
        v = y (H - y) n_v
        p = dp (1 - x/L) + x (L - x) n_p   (p = dp at inlet, 0 at outlet)

    Exact solution:  u = dp / (2 mu L) * y (H - y),  v = 0,  p linear.
    """
    def __init__(self, L=1.0, H=1.0, dp=1.0, rho=1.0, mu=0.1):
        self.L, self.H, self.dp, self.rho, self.mu = L, H, dp, rho, mu

    def sample(self, n, rng):
        return np.stack([rng.uniform(0, self.L, n), rng.uniform(0, self.H, n)], 1)

    def fields(self, params, xi):
        x, y = xi[0], xi[1]
        n = mlp(params, xi)
        D = y * (self.H - y)
        u = D * n[0]
        v = D * n[1]
        p = self.dp * (1.0 - x / self.L) + x * (self.L - x) * n[2]
        return u, v, p

    def exact_u(self, y):
        return self.dp / (2.0 * self.mu * self.L) * y * (self.H - y)


class Stenosis(Problem):
    """Channel with a Gaussian constriction on the upper wall,
    parameterized by the severity alpha (an INPUT to the network, so one
    trained model is a surrogate over a range of geometries).

        lower wall  y = 0
        upper wall  y = h(x) = H - alpha exp(-50 (x - 0.5 L)^2)

    Boundary conditions, as in the paper: parabolic velocity at the inlet,
    no-slip on both walls, zero pressure at the outlet. All enforced by
    construction:

        u = u_part(x, y) + x y (h - y) n_u
        v =                x y (h - y) n_v
        p =                (L - x) n_p

    with the particular solution  u_part = 4 U H y (h - y) / h^3 : the
    inlet parabola 4 U y (H - y) / H^2 at x = 0 (where h = H), zero on
    BOTH walls for every x because it follows h(x), and carrying the same
    flow rate 2 U H / 3 at every station -- a mass-conserving first guess.
    The factor x makes the correction vanish at the inlet, y (h - y) makes
    it vanish on both walls, and (L - x) pins the outlet pressure to zero.
    Prescribing the inlet velocity fixes the flow rate; driving the flow
    by a pressure difference instead lets the optimizer park the whole
    pressure drop at the outlet and keep the interior stagnant.
    """
    def __init__(self, L=1.0, H=1.0, U=1.0, rho=1.0, mu=0.1,
                 alpha=None, alpha_range=(0.1, 0.4), throat_fraction=0.5):
        """alpha=None -> parametric surrogate, alpha is a network input
        drawn from alpha_range. alpha=<number> -> one fixed geometry with
        two inputs (x, y): solve this first; it is the easier problem."""
        self.L, self.H, self.U, self.rho, self.mu = L, H, U, rho, mu
        self.alpha = alpha
        self.alpha_range = alpha_range
        self.throat_fraction = throat_fraction
        self.n_inputs = 2 if alpha is not None else 3

    def wall(self, x, alpha):
        return self.H - alpha * jnp.exp(-50.0 * (x - 0.5 * self.L) ** 2)

    def wall_np(self, x, alpha):
        return self.H - alpha * np.exp(-50.0 * (x - 0.5 * self.L) ** 2)

    def inlet_u(self, y):
        return 4.0 * self.U * y * (self.H - y) / self.H ** 2

    def u_part(self, x, y, alpha, exp=jnp.exp):
        h = self.H - alpha * exp(-50.0 * (x - 0.5 * self.L) ** 2)
        return 4.0 * self.U * self.H * y * (h - y) / h ** 3

    def sample(self, n, rng):
        """Uniform points plus a share concentrated around the throat,
        where the velocity gradients (and the residuals) are largest."""
        n_t = int(self.throat_fraction * n)
        x = np.concatenate([rng.uniform(0, self.L, n - n_t),
                            np.clip(rng.normal(0.5 * self.L, 0.12 * self.L, n_t),
                                    0, self.L)])
        if self.alpha is None:
            a = rng.uniform(*self.alpha_range, n)
            y = rng.uniform(0, 1, n) * self.wall_np(x, a)  # inside the geometry
            return np.stack([x, y, a], 1)
        y = rng.uniform(0, 1, n) * self.wall_np(x, self.alpha)
        return np.stack([x, y], 1)

    def fields(self, params, xi):
        x, y = xi[0], xi[1]
        a = xi[2] if self.alpha is None else self.alpha
        n = mlp(params, xi)
        D = x * y * (self.wall(x, a) - y)
        u = self.u_part(x, y, a) + D * n[0]
        v = D * n[1]
        p = (self.L - x) * n[2]
        return u, v, p


# ============================================================ residuals
def make_residuals(problem):
    """Return residuals(params, X) -> (rx, ry, rc), each of shape (N,),
    for a batch X of shape (N, n_inputs). Steady incompressible NS:

        rho (u u_x + v u_y) + p_x - mu (u_xx + u_yy) = 0
        rho (u v_x + v v_y) + p_y - mu (v_xx + v_yy) = 0
        u_x + v_y = 0
    """
    rho, mu = problem.rho, problem.mu

    def point(params, xi):
        u = lambda z: problem.fields(params, z)[0]
        v = lambda z: problem.fields(params, z)[1]
        p = lambda z: problem.fields(params, z)[2]
        U, V = u(xi), v(xi)
        gu, gv, gp = jax.grad(u)(xi), jax.grad(v)(xi), jax.grad(p)(xi)
        Hu, Hv = jax.hessian(u)(xi), jax.hessian(v)(xi)
        rx = rho * (U * gu[0] + V * gu[1]) + gp[0] - mu * (Hu[0, 0] + Hu[1, 1])
        ry = rho * (U * gv[0] + V * gv[1]) + gp[1] - mu * (Hv[0, 0] + Hv[1, 1])
        rc = gu[0] + gv[1]
        return rx, ry, rc

    return jax.vmap(point, in_axes=(None, 0))


def make_predict(problem):
    """predict(params, X) -> array (N, 3) of (u, v, p)."""
    def point(params, xi):
        return jnp.stack(problem.fields(params, xi))
    return jax.vmap(point, in_axes=(None, 0))


# ============================================================= training
def adam_init(params):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {'m': zeros, 'v': zeros, 't': 0}


def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state['t'] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, state['m'], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, state['v'], grads)
    mhat = jax.tree_util.tree_map(lambda m: m / (1 - b1 ** t), m)
    vhat = jax.tree_util.tree_map(lambda v: v / (1 - b2 ** t), v)
    new = jax.tree_util.tree_map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                                 params, mhat, vhat)
    return new, {'m': m, 'v': v, 't': t}


def train(problem, layers=(2, 40, 40, 40, 3), n_iter=3000, batch=512,
          lr=1e-3, seed=0, log_every=200, lbfgs_iter=0, lbfgs_points=8000):
    """Train on physics residuals only. Returns (params, history).

    Stage 1: Adam with fresh random collocation points every iteration.
    Stage 2 (if lbfgs_iter > 0): L-BFGS on a FIXED set of lbfgs_points
    points -- the standard way to drive a PINN loss down the last orders of
    magnitude once Adam has found the basin.
    """
    assert layers[0] == problem.n_inputs, 'first layer must match problem.n_inputs'
    rng = np.random.default_rng(seed)
    params = init_mlp(jax.random.PRNGKey(seed), layers)
    residuals = make_residuals(problem)

    def loss_fn(params, X):
        rx, ry, rc = residuals(params, X)
        return jnp.mean(rx ** 2) + jnp.mean(ry ** 2) + jnp.mean(rc ** 2)

    @jax.jit
    def step(params, state, X, lr):
        loss, grads = jax.value_and_grad(loss_fn)(params, X)
        params, state = adam_update(params, grads, state, lr)
        return params, state, loss

    state = adam_init(params)
    history = []
    for i in range(1, n_iter + 1):
        X = jnp.asarray(problem.sample(batch, rng))
        lr_i = lr * (0.1 ** (i / n_iter))          # smooth decay lr -> lr/10
        params, state, loss = step(params, state, X, lr_i)
        history.append(float(loss))
        if log_every and (i % log_every == 0 or i == 1):
            print(f'iter {i:6d}   loss {float(loss):.3e}')

    if lbfgs_iter > 0:
        params, extra = lbfgs_refine(loss_fn, params,
                                     jnp.asarray(problem.sample(lbfgs_points, rng)),
                                     lbfgs_iter, log_every)
        history += extra
    return params, history


def lbfgs_refine(loss_fn, params, X, n_iter, log_every=200):
    """Polish with SciPy's L-BFGS-B on a fixed batch X. loss_fn(params, X)."""
    from scipy.optimize import minimize
    from jax.flatten_util import ravel_pytree
    flat0, unravel = ravel_pytree(params)

    @jax.jit
    def value_and_grad(flat):
        return jax.value_and_grad(lambda f: loss_fn(unravel(f), X))(flat)

    history = []

    def fun(z):
        l, g = value_and_grad(jnp.asarray(z))
        history.append(float(l))
        if log_every and len(history) % log_every == 0:
            print(f'lbfgs {len(history):5d}   loss {float(l):.3e}')
        return float(l), np.asarray(g, dtype=np.float64)

    print(f'L-BFGS refinement on {X.shape[0]} fixed points ...')
    res = minimize(fun, np.asarray(flat0, dtype=np.float64), jac=True,
                   method='L-BFGS-B',
                   options={'maxiter': n_iter, 'maxfun': 2 * n_iter,
                            'maxcor': 50, 'ftol': 0, 'gtol': 1e-12})
    print(f'L-BFGS done: {res.nit} iterations, final loss {res.fun:.3e}')
    return unravel(jnp.asarray(res.x)), history


# ================================================================== io
def save_params(params, path):
    with open(path, 'wb') as f:
        pickle.dump([(np.asarray(W), np.asarray(b)) for W, b in params], f)


def load_params(path):
    with open(path, 'rb') as f:
        return [(jnp.asarray(W), jnp.asarray(b)) for W, b in pickle.load(f)]