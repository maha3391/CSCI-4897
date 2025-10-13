import numpy as np

# analytical and forward euler functions from part a
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

# define maximum absolute error
def max_abs_error(dt, i0=0.01, beta=3, gamma=2, t_max=25.0):
    T, I_euler = forward_euler_sis(i0, beta, gamma, t_max, dt)
    I_analytical = analytical_i(T, i0, beta, gamma)
    return np.max(np.abs(I_euler - I_analytical))