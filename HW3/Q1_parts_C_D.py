import numpy as np
import matplotlib.pyplot as plt

def multi_group_sir(s0, i0, r0, p, beta, gamma, t_max, stepsize):
    """
    Multi-group SIR model with different susceptibilities
    p: array of susceptibility parameters for each group
    """
    T = np.arange(0, t_max + stepsize, stepsize)
    n_groups = len(p)
    n_steps = len(T)
    
    # Initialize arrays
    S = np.zeros((n_groups, n_steps))
    I = np.zeros((n_groups, n_steps))
    R = np.zeros((n_groups, n_steps))
    
    # Set initial conditions
    S[:, 0] = s0
    I[:, 0] = i0
    R[:, 0] = r0
    
    for idx in range(1, n_steps):
        # Calculate total force of infx
        total_I = np.sum(I[:, idx-1])
        
        for i in range(n_groups):
            # SIR equations with group-specific susceptibility
            dS_dt = -p[i] * beta * S[i, idx-1] * total_I
            dI_dt = p[i] * beta * S[i, idx-1] * total_I - gamma * I[i, idx-1]
            dR_dt = gamma * I[i, idx-1]
            
            S[i, idx] = S[i, idx-1] + dS_dt * stepsize
            I[i, idx] = I[i, idx-1] + dI_dt * stepsize
            R[i, idx] = R[i, idx-1] + dR_dt * stepsize
    
    return S, I, R, T

# Initialize parameters
n_groups = 4
N_per_group = 1000  # Population per group 
p = np.array([1.0, 2.0, 3.0, 4.0])  # Susceptibility parameters
c_bar = 0.45  
beta = c_bar/N_per_group # Transmission rate per contact
gamma = 3.0  # Recovery rate

# Initial conditions: 99.9% susceptible, 0.1% infected
s0 = np.full(n_groups, 0.999 * N_per_group)
i0 = np.full(n_groups, 0.001 * N_per_group)
r0 = np.zeros(n_groups)

# Simulation parameters
t_max = 40  # Adjust based on epidemic duration
stepsize = 0.1

# Run simulation
S, I, R, T = multi_group_sir(s0, i0, r0, p, beta, gamma, t_max, stepsize)

# Part D: Calculate average relative susceptibility
def calculate_avg_susceptibility(S, p):
    """Calculate average relative susceptibility among susceptibles"""
    p_bar = np.zeros(len(T))
    for t_idx in range(len(T)):
        numerator = np.sum(p * S[:, t_idx])
        denominator = np.sum(S[:, t_idx])
        p_bar[t_idx] = numerator / denominator if denominator > 0 else 0
    return p_bar

p_bar = calculate_avg_susceptibility(S, p)

colors = ["#7ACAEC", "#218CB6", "#264AB6",  '#000080']  # light blue to dark blue 

# Create figure with subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Convert to fractions
I_fraction = I / N_per_group
S_fraction = S / N_per_group

# Plot infected compartments as fractions
for i in range(n_groups):
    ax1.plot(T, I_fraction[i, :], color=colors[i], linewidth=2, 
             label=f'Group {i+1} (p={p[i]})')
ax1.set_xlabel('Time')
ax1.set_ylabel('Infected Fraction')
ax1.set_title('Infected Compartments i₁(t), i₂(t), i₃(t), i₄(t)')
ax1.legend()
ax1.grid(True)

# Plot susceptible compartments as fractions
for i in range(n_groups):
    ax2.plot(T, S_fraction[i, :], color=colors[i], linewidth=2, 
             label=f'Group {i+1} (p={p[i]})')
ax2.set_xlabel('Time')
ax2.set_ylabel('Susceptible Fraction')
ax2.set_title('Susceptible Compartments s₁(t), s₂(t), s₃(t), s₄(t)')
ax2.legend()
ax2.grid(True)

# Plot average relative susceptibility
ax3.plot(T, p_bar, color='black', linewidth=2)
ax3.set_xlabel('Time')
ax3.set_ylabel('Average Relative Susceptibility')
ax3.set_title('p̄(t)')
ax3.grid(True)

plt.tight_layout()
plt.show()