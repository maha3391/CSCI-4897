import numpy as np
import matplotlib.pyplot as plt

# define functions from parts a-c
def analytical_i(t, i0, beta, gamma):
    r = beta - gamma
    K = r / beta
    exp_rt = np.exp(r * t)
    return (K * i0 * exp_rt) / (K + i0 * (exp_rt - 1))

def forward_euler_sis(i0, beta, gamma, t_max, stepsize):
    T = np.arange(0, t_max + stepsize, stepsize)
    I = np.zeros(len(T))
    I[0] = i0
    for idx in range(1, len(T)):
        di_dt = (beta - gamma) * I[idx-1] - beta * I[idx-1]**2
        I[idx] = I[idx-1] + di_dt * stepsize
    return T, I

def max_abs_error(dt, i0=0.01, beta=3, gamma=2, t_max=25.0):
    T, I_euler = forward_euler_sis(i0, beta, gamma, t_max, dt)
    I_analytical = analytical_i(T, i0, beta, gamma)
    return np.max(np.abs(I_euler - I_analytical))

# define delta t values
dt_values = np.array([2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125])
errors = np.array([max_abs_error(dt) for dt in dt_values])

# plot
plt.figure(figsize=(6, 5))
plt.loglog(dt_values, errors, 'ro-', label='Max Abs Error')

plt.xlabel(r'$\Delta t$', fontsize=12)
plt.ylabel(r'$E(\Delta t)$', fontsize=12)
plt.title('Error vs Step Size (log–log scale)')
plt.grid(True, which="both", ls="--", lw=0.5)
plt.legend()
plt.show()