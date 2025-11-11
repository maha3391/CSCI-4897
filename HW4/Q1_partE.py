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

# ==============================================================================
# VISUALIZATION
# ==============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Cases over time
axes[0].bar(df['Week'], df['New Cases'], color='steelblue', alpha=0.6, label='Weekly Cases')
axes[0].axvline(17, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Peak (week 17)')
axes[0].set_xlabel('Week', fontsize=11)
axes[0].set_ylabel('New Cases', fontsize=11)
axes[0].set_title('BRUH Disease Epidemic Curve', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Plot 2: Rt over time
axes[1].plot(weeks, Rt_values, 'b-', linewidth=2, label='Rt estimate')
axes[1].fill_between(weeks, Rt_ci_lower, Rt_ci_upper, alpha=0.3, color='blue', label='95% CI')
axes[1].axhline(1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Rt = 1 (epidemic threshold)')
axes[1].axvline(17, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Peak')
axes[1].set_xlabel('Week', fontsize=11)
axes[1].set_ylabel('Rt', fontsize=11)
axes[1].set_title('Time-Varying Reproduction Number (Rt)', fontsize=12, fontweight='bold')
axes[1].set_ylim([0, 3])
axes[1].legend(loc='upper right')
axes[1].grid(True, alpha=0.3)

# Highlight regions
axes[1].axvspan(5, 15, alpha=0.1, color='green', label='Growth phase')
axes[1].axvspan(15, 20, alpha=0.1, color='yellow')
axes[1].axvspan(20, 50, alpha=0.1, color='orange')

plt.tight_layout()
plt.savefig('Q1_partE_Rt_timeseries.png', dpi=300, bbox_inches='tight')
print("\nFigure saved: Q1_partE_Rt_timeseries.png")

# ==============================================================================
# INTERPRETATION
# ==============================================================================
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("\n1. EPIDEMIC PHASES:")
print("   - Early growth (weeks 5-15): Rt > 1, epidemic expanding")
print("   - Peak (week 17): Rt approaches 1, growth slowing")
print("   - Decline (weeks 20+): Rt < 1, epidemic contracting")

print("\n2. COMPARISON WITH R₀:")
week10_idx = [i for i, w in enumerate(weeks) if w == 10]
if week10_idx:
    print(f"   - Rt at week 10: {Rt_values[week10_idx[0]]:.2f}")
    print(f"   - R₀ from Part A: 1.87 (early exponential growth)")
    print(f"   - Similar values confirm early epidemic phase")

print("\n3. KEY FINDING:")
print("   Rt crosses below 1.0 around weeks 17-20, indicating the")
print("   epidemic peaked and began to decline as susceptibles depleted.")

print("="*70)
