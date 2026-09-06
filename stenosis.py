"""
stenosis.py -- parametric surrogate for a stenotic channel.

The stenosis severity alpha is an INPUT to the network, so one training
run gives the flow for every alpha in the range (the paper's main point:
a surrogate over geometry, trained with no simulation data).

    python stenosis.py --alpha 0.3      # ONE geometry first (easier, ~15 min CPU)
    python stenosis.py                  # parametric surrogate over alpha in [0.1, 0.4]
    python stenosis.py --iters 5000 --lbfgs 1000        # quicker smoke test

Writes figures/stenosis_result.png and models/stenosis.pkl.
"""
import argparse
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pcnn import Stenosis, train, make_predict, save_params

p = argparse.ArgumentParser()
p.add_argument('--alpha', type=float, default=None,
               help='fixed stenosis severity (single geometry). Omit for the '
                    'parametric surrogate over --alpha-range')
p.add_argument('--alpha-range', type=float, nargs=2, default=[0.1, 0.4])
p.add_argument('--iters', type=int, default=15000, help='Adam iterations')
p.add_argument('--lbfgs', type=int, default=5000, help='L-BFGS iterations after Adam')
p.add_argument('--batch', type=int, default=1024)
p.add_argument('--plot-alphas', type=float, nargs='+', default=None)
args = p.parse_args()

os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

fixed = args.alpha is not None
prob = Stenosis(alpha=args.alpha, alpha_range=tuple(args.alpha_range))
layers = (prob.n_inputs, 64, 64, 64, 64, 3)
params, hist = train(prob, layers=layers, n_iter=args.iters,
                     batch=args.batch, lbfgs_iter=args.lbfgs)
tag = f'stenosis_alpha{args.alpha:g}' if fixed else 'stenosis'
save_params(params, f'models/{tag}.pkl')
predict = make_predict(prob)
if args.plot_alphas is None:
    args.plot_alphas = [args.alpha] if fixed else [0.1, 0.25, 0.4]


def inputs(X, Y, a):
    cols = [X.ravel(), Y.ravel()] + ([] if fixed else [np.full(X.size, a)])
    return jnp.asarray(np.stack(cols, 1))

nx, ny = 160, 80
xs = np.linspace(0, prob.L, nx)
fig, axes = plt.subplots(len(args.plot_alphas), 2,
                         figsize=(12, 3.2 * len(args.plot_alphas)))
axes = np.atleast_2d(axes)
for row, a in zip(axes, args.plot_alphas):
    h = prob.wall_np(xs, a)
    # grid that follows the wall: y = s * h(x), s in [0, 1]
    S, Xg = np.meshgrid(np.linspace(0, 1, ny), xs, indexing='ij')
    Yg = S * h[None, :]
    out = np.asarray(predict(params, inputs(Xg, Yg, a)))
    U, V, P = [out[:, k].reshape(Xg.shape) for k in range(3)]
    speed = np.sqrt(U ** 2 + V ** 2)
    # physics checks: flow rate must be the same at every x (2/3 here),
    # and the inlet-to-outlet pressure drop is the engineering output
    Q = [np.trapezoid(U[:, j], Yg[:, j]) for j in (0, nx // 2, nx - 1)]
    dp = float(np.mean(P[:, 0]))
    print(f'alpha = {a:.2f}:  flow rate at x = 0, L/2, L: '
          f'{Q[0]:.3f} {Q[1]:.3f} {Q[2]:.3f}  (exact 0.667)   '
          f'pressure drop = {dp:.3f}   throat peak speed = {speed[:, nx // 2].max():.3f}')

    c = row[0].contourf(Xg, Yg, speed, 30, cmap='viridis')
    row[0].plot(xs, h, 'k', lw=1.5); fig.colorbar(c, ax=row[0], label='|u|')
    row[0].set_title(f'speed, α = {a}'); row[0].set_aspect('equal')
    c = row[1].contourf(Xg, Yg, P, 30, cmap='coolwarm')
    row[1].plot(xs, h, 'k', lw=1.5); fig.colorbar(c, ax=row[1], label='p')
    row[1].set_title(f'pressure, α = {a}'); row[1].set_aspect('equal')
fig.suptitle('One network, many geometries: stenotic channel surrogate', fontsize=13)
fig.tight_layout()
fig.savefig(f'figures/{tag}_result.png', dpi=150)
print(f'final loss {hist[-1]:.3e}   saved figures/{tag}_result.png')