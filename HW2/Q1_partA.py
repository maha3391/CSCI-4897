import numpy as np 
import matplotlib.pyplot as plt

# Function to compute analytical solution for SIS model
def analytical_i(t, i0, beta, gamma):
    """Analytical solution for SIS model I(t)"""
    r = beta - gamma
    K = r/beta
    return K * i0 * np.exp(r * t) / (K + i0 * (np.exp(r * t) - 1))

# Using your Forward Euler method, simulate the solution to the normalized SIS model discussed in class (Week 3) 
def forward_euler_sis(i0, beta, gamma, t_max, stepsize):
    T = np.arange(0, t_max + stepsize, stepsize)
    I = np.zeros(len(T))
    I[0] = i0
    
    for idx in range(1, len(T)):
        # Simplified SIS model: dI/dt = (β - γ)I - βI^2
        di_dt = (beta - gamma) * I[idx-1] - beta * I[idx-1]**2
        I[idx] = I[idx-1] + di_dt * stepsize
    
    return T, I

# using β = 3 and γ = 2, and with (s0, i0) = (0.99, 0.01).
beta = 3
gamma = 2
s0 = 0.99
i0 = 0.01
t_max = 25
# simulate using a step size ∆t = 2, ∆t = 1, ∆t = 1/2 . 
step_sizes = [2, 1, 0.5]

# Create three plots ranging from t = 0 to t = 25. 
# In each plot, show only your solution’s I(t) in a red solid line, labeled as “Forward Euler”, and then also plot the analytical solution from class in a black dashed line, labeled as “Analytical.” Please also set the y-axis range to [0, 0.5].

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, stepsize in enumerate(step_sizes):
    T, I_euler = forward_euler_sis(i0, beta, gamma, t_max, stepsize)
    I_analytical = analytical_i(T, i0, beta, gamma)
    
    axes[i].plot(T, I_euler, label='Forward Euler', color='red', linestyle='-')
    axes[i].plot(T, I_analytical, label='Analytical', color='black', linestyle='--')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('I(t)')
    axes[i].set_title(f'SIS Model: Δt = {stepsize}')
    axes[i].set_ylim(0, 0.5)
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()