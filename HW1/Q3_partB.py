import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def find_rinf(R0, guess=0.1):
    h = lambda r: r - (1 - np.exp(-R0 * r))
    r_root, = fsolve(h, guess)  # returns array
    # ensure small negative numerical values clipped to 0
    return max(0.0, float(r_root))

R0_values = [0.9, 1.0, 1.1, 1.2]
r_grid = np.linspace(0, 1, 401)

fig, axes = plt.subplots(2, 2, figsize=(10,8))
axes = axes.ravel()
for ax, R0 in zip(axes, R0_values):
    f = r_grid
    g = 1 - np.exp(-R0 * r_grid)
    ax.plot(r_grid, f, 'k-', label='f(r)=r')
    ax.plot(r_grid, g, 'r-', label='g(r)=1-e^{-R0 r}')
    # try several initial guesses for robustness
    guesses = [0.0, 0.001, 0.1, 0.5, 0.9]
    roots = set()
    for guess in guesses:
        rroot = find_rinf(R0, guess=guess)
        roots.add(round(rroot,10))
    roots = sorted(list(roots))
    for rroot in roots:
        ax.scatter([rroot], [rroot], s=80, facecolors='none', edgecolors='b', linewidths=2)
        ax.text(rroot, rroot, f'  r∞={rroot:.4f}', va='bottom')
    ax.set_title(f'R0 = {R0}')
    ax.set_xlabel(r'$r_\infty$')
    ax.set_ylabel('value')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
