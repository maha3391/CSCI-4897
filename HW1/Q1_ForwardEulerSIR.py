import numpy as np 
import matplotlib.pyplot as plt

def SIR_birthdeath(S0, I0, R0, beta, gamma, mu_birth, mu_death, t_max, stepsize):
    T = np.arange(0,t_max+stepsize,stepsize)
    S = np.zeros(len(T))
    I = np.zeros(len(T))
    R = np.zeros(len(T))
    N = np.zeros(len(T))
    
    # initialize idx=0
    S[0] = S0
    I[0] = I0
    R[0] = R0
    N[0] = S0 + I0 + R0
    
    for idx, t in enumerate(T):
        if idx==0:
            continue # already initialized above
        else:
            N_prev = S[idx-1] + I[idx-1] + R[idx-1]
            dS_dt = mu_birth * N_prev - beta * S[idx-1] * I[idx-1] / N_prev - mu_death * S[idx-1]
            dI_dt = beta * S[idx-1] * I[idx-1] / N_prev - gamma * I[idx-1] - mu_death * I[idx-1]
            dR_dt = gamma * I[idx-1] - mu_death * R[idx-1]
            
            S[idx] = S[idx-1] + dS_dt * stepsize
            I[idx] = I[idx-1] + dI_dt * stepsize
            R[idx] = R[idx-1] + dR_dt * stepsize
            N[idx] = S[idx] + I[idx] + R[idx]
    
    return S, I, R, N, T

# Parameters
N = 1000
I0 = 1
S0 = 999
R0=0
beta = 1
gamma = 0.5
mu_birth = 0.01
mu_death = 0.5*mu_birth
t_max = 82 # have to go long enough to reach N=1500
stepsize = 0.1
# Simulation
S, I, R, N, T = SIR_birthdeath(S0, I0, R0, beta, gamma, mu_birth, mu_death, t_max, stepsize)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(T, S, label='Susceptible', color='blue')
plt.plot(T, I, label='Infected', color='red')
plt.plot(T, R, label='Recovered', color='green')
plt.plot(T, N, label='Total Population', color='black', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR  Model with Population Growth by 50% - Malia Hayes')

# Add horizontal dotted line at population = 1500
plt.axhline(y=1500, color='gray', linestyle=':', label='Population = 1500')
# Find where N crosses 1500
cross_idx = np.where(N >= 1500)[0][0]
cross_time = T[cross_idx]
# Add vertical dotted line at crossing time
plt.axvline(x=cross_time, color='purple', linestyle=':', label=f'Time at N=1500 ({cross_time:.2f})')

plt.legend(fontsize=8)
plt.grid(True)
plt.show()

# confirm we reached 1500
print(f"Final population: {N[-1]:.2f}")