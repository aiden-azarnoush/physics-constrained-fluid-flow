"""
check_physics.py -- verify the residual code before training anything.

The exact Poiseuille solution is fed through the SAME residual code used
in training (by constructing a Problem whose `fields` return the exact
u, v, p). All three Navier-Stokes residuals must then be ~0 at every
point. If this passes, the derivative machinery is correct.

    python check_physics.py
"""
import numpy as np
import jax.numpy as jnp
from pcnn import Problem, make_residuals


class ExactPoiseuille(Problem):
    L, H, dp, rho, mu = 1.0, 1.0, 1.0, 1.0, 0.1

    def fields(self, params, xi):
        x, y = xi[0], xi[1]
        u = self.dp / (2 * self.mu * self.L) * y * (self.H - y)
        v = 0.0 * x
        p = self.dp * (1 - x / self.L)
        return u, v, p

    def sample(self, n, rng):
        return np.stack([rng.uniform(0, self.L, n), rng.uniform(0, self.H, n)], 1)


class WrongSolution(ExactPoiseuille):
    """Same, but with a deliberately wrong velocity profile (cubic).
    Residuals must NOT vanish -- proves the check can fail."""
    def fields(self, params, xi):
        x, y = xi[0], xi[1]
        u = 5.0 * y ** 2 * (self.H - y)
        return u, 0.0 * x, self.dp * (1 - x / self.L)


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    for prob, expect_zero in [(ExactPoiseuille(), True), (WrongSolution(), False)]:
        X = jnp.asarray(prob.sample(2000, rng))
        rx, ry, rc = make_residuals(prob)(None, X)
        worst = max(float(jnp.abs(r).max()) for r in (rx, ry, rc))
        name = prob.__class__.__name__
        print(f'{name:16s} max |residual| = {worst:.2e}   '
              f'(x-mom {float(jnp.abs(rx).max()):.1e}, '
              f'y-mom {float(jnp.abs(ry).max()):.1e}, '
              f'cont {float(jnp.abs(rc).max()):.1e})')
        ok = (worst < 1e-8) if expect_zero else (worst > 1e-2)
        print('   ->', 'PASS' if ok else 'FAIL',
              '(exact solution satisfies NS)' if expect_zero
              else '(wrong solution is correctly rejected)')
        assert ok
    print('\nPhysics residual code verified.')
