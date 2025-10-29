import numpy as np
from scipy.stats import nbinom
import matplotlib.pyplot as plt

def n_children_negbinom(R0, k, n_draws):
    """
    Generate offspring using negative binomial distribution
    Similar to demo's n_children_poisson but with NB distribution
    """
    # If no parents, no children
    if n_draws == 0:
        return 0
    
    # Limit population size to prevent memory issues
    if n_draws > 10000:
        return 100000  # Return large number which indicates non-extinction
    
    # Calculate NB parameters from R0 and k
    mean = R0
    variance = mean + (mean**2)/k
    p = mean/variance
    n = mean**2 / (variance - mean)
    
    # Draw offspring for n_draws parents
    draws = nbinom.rvs(n=n, p=p, size=n_draws)
    total_children = np.sum(draws)
    return total_children

def sample_BP(children_function, R0, k, n_generations):
    """
    Sample a branching process trajectory
    Similar to demo's sample_BP but passes k parameter
    """
    z = np.zeros(n_generations, dtype=int)
    z[0] = 1  # Start with single infection
    
    for generation in np.arange(1, n_generations):
        # If already extinct, stay extinct
        if z[generation-1] == 0:
            z[generation] = 0
        else:
            z[generation] = children_function(R0, k, z[generation-1])
            # If population explodes, stop early and mark as non-extinct
            if z[generation] > 50000:
                z[generation:] = 100000
                break
    
    return z

def estimate_extinction_probability(R0, k, n_generations, n_simulations):
    """
    Estimate probability that epidemic dies in finite time
    """
    extinctions = 0
    
    for sim in range(n_simulations):
        trajectory = sample_BP(n_children_negbinom, R0, k, n_generations)
        
        # Check if epidemic went extinct (reached 0)
        if trajectory[-1] == 0:
            extinctions += 1
    
    return extinctions / n_simulations

# Parameters
R0 = 3
k_values = [0.1, 0.5, 1.0, 5.0, 10.0]
n_generations = 15  # Reduced from 20 for speed
n_simulations = 5000

print(f"Branching Process Extinction Probability")
print(f"R0 = {R0}")
print(f"Generations = {n_generations}")
print(f"Simulations per k value = {n_simulations}")
print("\n" + "="*40)

# Calculate extinction probabilities
results = []
for k in k_values:
    print(f"Computing for k = {k}...")
    q = estimate_extinction_probability(R0, k, n_generations, n_simulations)
    results.append({'k': k, 'q': round(q, 3)})

# Display table
print("\n" + "="*40)
print("Table of Results:")
print("="*40)
print(f"{'k':<10} {'q':<10}")
print("-"*20)
for result in results:
    print(f"{result['k']:<10.1f} {result['q']:<10.3f}")
print("="*40)
