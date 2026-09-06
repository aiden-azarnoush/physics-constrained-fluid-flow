"""
uncertainty.py -- Monte Carlo uncertainty propagation through the
stenosis surrogate. Because alpha is a network input, propagating a
distribution of alpha costs one forward pass per sample: no re-training,
no simulations.

    python uncertainty.py --mean 0.4 --std 0.08 --samples 200

Needs models/stenosis.pkl from stenosis.py.
Writes figures/uncertainty_result.png.
"""
import argparse
import os
import warnings
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pcnn import Stenosis, make_predict, load_params

p = argparse.ArgumentParser()
p.add_argument('--mean', type=float, default=0.25)
p.add_argument('--std', type=float, default=0.05)
p.add_argument('--samples', type=int, default=200)
p.add_argument('--model', default='models/stenosis.pkl')
args = p.parse_args()
os.makedirs('figures', exist_ok=True)

prob = Stenosis()
params = load_params(args.model)
predict = make_predict(prob)
rng = np.random.default_rng(0)

# fixed physical grid; points above the wall for a given alpha are masked
xs, ys = np.linspace(0, prob.L, 140), np.linspace(0, prob.H, 70)
Xg, Yg = np.meshgrid(xs, ys)
lo, hi = prob.alpha_range
alphas = np.clip(rng.normal(args.mean, args.std, args.samples), lo, hi)

fields = []
for a in alphas:
    G = jnp.asarray(np.stack([Xg.ravel(), Yg.ravel(), np.full(Xg.size, a)], 1))
    out = np.array(predict(params, G)).reshape(*Xg.shape, 3)   # copy: writable
    inside = Yg <= prob.wall_np(Xg, a)
    out[~inside] = np.nan
    fields.append(out)
fields = np.array(fields)                       # (samples, ny, nx, 3)
with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN cells above the wall
    mean, std = np.nanmean(fields, 0), np.nanstd(fields, 0)
frac_inside = np.mean(~np.isnan(fields[..., 0]), 0)
mean[frac_inside < 0.5] = np.nan; std[frac_inside < 0.5] = np.nan

fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))
for k, (name, cmap) in enumerate([('u', 'viridis'), ('p', 'coolwarm')]):
    idx = 0 if name == 'u' else 2
    c = ax[k, 0].contourf(Xg, Yg, mean[..., idx], 30, cmap=cmap)
    fig.colorbar(c, ax=ax[k, 0]); ax[k, 0].set_title(f'mean {name}')
    c = ax[k, 1].contourf(Xg, Yg, std[..., idx], 30, cmap='magma')
    fig.colorbar(c, ax=ax[k, 1]); ax[k, 1].set_title(f'std {name}')
    for a_ in ax[k]:
        a_.plot(xs, prob.wall_np(xs, args.mean), 'w--', lw=1); a_.set_aspect('equal')
fig.suptitle(f'Monte Carlo over α ~ N({args.mean}, {args.std}²), '
             f'{args.samples} samples, zero extra simulations', fontsize=13)
fig.tight_layout()
fig.savefig('figures/uncertainty_result.png', dpi=150)
print('saved figures/uncertainty_result.png')