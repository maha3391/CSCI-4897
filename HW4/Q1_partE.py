import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

print("="*70)
print("Q1 Part E - Method 5: Time-Varying Rt Estimation")
print("="*70)

# Read data
df = pd.read_csv('HW4_all_weeks.csv')

# Parameters
gamma = 1/2  # Recovery rate
generation_time = 2  # weeks (same as infection duration)

print(f"\nParameters:")
print(f"  Generation time: {generation_time} weeks")
print(f"  Recovery rate γ: {gamma} per week")

# ==============================================================================
# METHOD 5: Rt from Case Ratios (Cori et al. / Wallinga-Teunis approach)
# ==============================================================================
print("\n" + "="*70)
print("METHOD 5: Time-Varying Rt Estimation")
print("="*70)
print("Theory: Rt(t) ≈ I(t) / ∑[I(t-s) × w(s)]")
print("        where w(s) is the generation time distribution")
print("        Simplified: Rt(t) ≈ I(t) / I(t-τ) for generation time τ")

# Simple approach: ratio of cases at time t to cases τ weeks ago
tau = int(generation_time)
df['Rt_simple'] = df['New Cases'] / df['New Cases'].shift(tau)

# Smoothed approach: use 3-week window
window = 3
df['Cases_smooth'] = df['New Cases'].rolling(window=window, center=True).mean()
df['Rt_smooth'] = df['Cases_smooth'] / df['Cases_smooth'].shift(tau)

# Calculate Rt with confidence intervals (using Poisson assumption)
Rt_values = []
Rt_ci_lower = []
Rt_ci_upper = []
weeks = []

for i in range(tau, len(df)):
    current_cases = df.loc[i, 'New Cases']
    past_cases = df.loc[i-tau, 'New Cases']
    
    if past_cases > 0:
        Rt = current_cases / past_cases
        
        # Approximate CI using Poisson distribution
        # Rt ~ I(t) / I(t-τ), variance ≈ I(t) / I(t-τ)²
        se_Rt = np.sqrt(current_cases) / past_cases if current_cases > 0 else 0
        
        Rt_values.append(Rt)
        Rt_ci_lower.append(max(0, Rt - 1.96 * se_Rt))
        Rt_ci_upper.append(Rt + 1.96 * se_Rt)
        weeks.append(df.loc[i, 'Week'])

# Key time periods
early_epidemic = [i for i, w in enumerate(weeks) if 5 <= w <= 15]
peak_period = [i for i, w in enumerate(weeks) if 15 <= w <= 20]
decline_period = [i for i, w in enumerate(weeks) if 20 <= w <= 30]

print(f"\nRt Estimates at Key Time Points:")
print(f"{'Week':<8} {'Rt':<8} {'95% CI':<20} {'Interpretation'}")
print("-"*60)

# Show selected weeks
key_weeks = [10, 15, 17, 20, 25, 50]
for week in key_weeks:
    idx = [i for i, w in enumerate(weeks) if w == week]
    if idx:
        i = idx[0]
        interp = ""
        if Rt_values[i] > 1:
            interp = "Growing"
        elif Rt_values[i] < 1:
            interp = "Declining"
        else:
            interp = "Stable"
        print(f"{week:<8} {Rt_values[i]:<8.3f} [{Rt_ci_lower[i]:.2f}, {Rt_ci_upper[i]:.2f}]      {interp}")

# Summary statistics
if early_epidemic:
    early_Rt = np.mean([Rt_values[i] for i in early_epidemic])
    print(f"\nEarly epidemic (weeks 5-15): Mean Rt = {early_Rt:.2f}")

if peak_period:
    peak_Rt = np.mean([Rt_values[i] for i in peak_period])
    print(f"Peak period (weeks 15-20):    Mean Rt = {peak_Rt:.2f}")

if decline_period:
    decline_Rt = np.mean([Rt_values[i] for i in decline_period])
    print(f"Decline phase (weeks 20-30):  Mean Rt = {decline_Rt:.2f}")



