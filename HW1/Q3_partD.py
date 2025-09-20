import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def SIR(S0,I0,R0, beta, gamma, t_max, stepsize):
    T = np.arange(0,t_max+stepsize,stepsize)
    S = np.zeros(len(T))
    I = np.zeros(len(T))
    R = np.zeros(len(T))
    N = S0+I0+R0
    
    for idx,t in enumerate(T):
        if idx==0:
            S[idx] = S0
            I[idx] = I0
            R[idx] = R0
        else:
            dS_dt = -beta * S[idx-1] * I[idx-1] / N
            dI_dt = beta * S[idx-1] * I[idx-1] / N - gamma * I[idx-1]
            dR_dt = gamma * I[idx-1]
            
            S[idx] = S[idx-1] + dS_dt * stepsize
            I[idx] = I[idx-1] + dI_dt * stepsize
            R[idx] = R[idx-1] + dR_dt * stepsize
    
    return S, I, R, T

def final_size(R0):
    h = lambda r: r - (1 - np.exp(-R0*r))
    r_root, = fsolve(h, 0.5)  # initial guess
    return float(r_root)

# Declare parameters for SIR simulation
S0, I0, R0 = 999, 1, 0
beta, gamma = 1.0, 0.5
t_max, stepsize = 100, 0.1

S, I, R, T = SIR(S0, I0, R0, beta, gamma, t_max, stepsize)

# Find final size 
R0_val = beta/gamma
r_inf = final_size(R0_val)        # fraction
R_inf_pred = r_inf * (S0+I0+R0)   # number recovered

# Plot results
plt.figure(figsize=(10,6))
plt.plot(T, S, label='Susceptible S(t)', color='blue')
plt.plot(T, I, label='Infected I(t)', color='red')
plt.plot(T, R, label='Recovered R(t)', color='black')
plt.axhline(y=R_inf_pred, color='green', linestyle=':', linewidth=2,
            label=f'Predicted final size r∞ = {R_inf_pred:.1f}')
plt.xlabel("Time")
plt.ylabel("Population")
plt.title(f"SIR model (β={beta}, γ={gamma}, R0={R0_val})")
plt.legend()
plt.grid(True)
plt.show()

print(f"Predicted fraction recovered r∞ = {r_inf:.4f}")
print(f"Predicted final recovered count = {R_inf_pred:.1f}")
print(f"Simulated final recovered count = {R[-1]:.1f}")