"""
poiseuille.py -- train on physics only and compare with the exact solution.

    python poiseuille.py                # ~1-3 min on a laptop CPU
    python poiseuille.py --iters 6000

Writes figures/poiseuille_result.png and models/poiseuille.pkl.
"""
import argparse
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pcnn import Poiseuille, train, make_predict, save_params

p = argparse.ArgumentParser()
p.add_argument('--iters', type=int, default=3000)
p.add_argument('--batch', type=int, default=512)
args = p.parse_args()

os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

prob = Poiseuille()
params, hist = train(prob, layers=(2, 40, 40, 40, 3),
                     n_iter=args.iters, batch=args.batch)
save_params(params, 'models/poiseuille.pkl')
predict = make_predict(prob)

# ---- compare with the exact parabola at mid-channel -------------------
y = np.linspace(0, prob.H, 200)
X = jnp.asarray(np.stack([np.full_like(y, 0.5 * prob.L), y], 1))
u_nn = np.asarray(predict(params, X))[:, 0]
u_ex = prob.exact_u(y)
rel_l2 = np.linalg.norm(u_nn - u_ex) / np.linalg.norm(u_ex)
print(f'\nrelative L2 error of u(x = L/2, y):  {rel_l2:.2e}')
print(f'max |u_nn - u_exact| = {np.abs(u_nn - u_ex).max():.2e}   '
      f'(u_max exact = {u_ex.max():.4f})')

# ---- full field -------------------------------------------------------
xg, yg = np.meshgrid(np.linspace(0, prob.L, 120), np.linspace(0, prob.H, 60))
G = jnp.asarray(np.stack([xg.ravel(), yg.ravel()], 1))
U, V, P = [np.asarray(predict(params, G))[:, k].reshape(xg.shape) for k in range(3)]

fig, ax = plt.subplots(2, 2, figsize=(11, 7))
ax[0, 0].plot(u_ex, y, 'k-', lw=2.5, label='exact')
ax[0, 0].plot(u_nn, y, 'o', color='#d3541f', ms=4, markevery=8, label='network')
ax[0, 0].set_xlabel('u'); ax[0, 0].set_ylabel('y')
ax[0, 0].set_title(f'Velocity profile at x = L/2   (rel. L2 error {rel_l2:.1e})')
ax[0, 0].legend(); ax[0, 0].grid(alpha=0.3)

ax[0, 1].semilogy(hist, color='#1f5fbf')
ax[0, 1].set_xlabel('iteration'); ax[0, 1].set_ylabel('physics loss')
ax[0, 1].set_title('Training on Navier-Stokes residuals only'); ax[0, 1].grid(alpha=0.3)

c = ax[1, 0].contourf(xg, yg, U, 30, cmap='viridis')
fig.colorbar(c, ax=ax[1, 0], label='u')
ax[1, 0].set_title('u(x, y)'); ax[1, 0].set_aspect('equal')

c = ax[1, 1].contourf(xg, yg, P, 30, cmap='coolwarm')
fig.colorbar(c, ax=ax[1, 1], label='p')
ax[1, 1].set_title('p(x, y)  (linear, as it must be)'); ax[1, 1].set_aspect('equal')
fig.suptitle('Poiseuille flow learned without any simulation data', fontsize=13)
fig.tight_layout()
fig.savefig('figures/poiseuille_result.png', dpi=150)
print('saved figures/poiseuille_result.png')
