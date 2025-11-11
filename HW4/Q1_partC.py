import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

print("="*70)
print("Q1 Part C - Method 3: Peak Incidence")
print("="*70)

# Read data
df = pd.read_csv('HW4_all_weeks.csv')

# Parameters
ascertainment = 0.10
seroprevalence = 0.517  # From Part B

print(f"\nParameters:")
print(f"  Ascertainment: {ascertainment:.0%}")
print(f"  Seroprevalence: {seroprevalence:.3f}")

# Find peak
peak_idx = df['New Cases'].idxmax()
peak_week = df.loc[peak_idx, 'Week']
peak_cases = df.loc[peak_idx, 'New Cases']

# Cumulative cases at peak
df['Cumulative Cases'] = df['New Cases'].cumsum()
cum_cases_at_peak = df.loc[peak_idx, 'Cumulative Cases']

print(f"\nPeak Information:")
print(f"  Peak week: {peak_week}")
print(f"  Peak cases: {peak_cases}")
print(f"  Cumulative cases at peak: {cum_cases_at_peak}")

# Estimate population size from seroprevalence
total_cases_observed = df['New Cases'].sum()
total_infected_estimated = total_cases_observed / ascertainment
N_estimated = total_infected_estimated / seroprevalence

print(f"\nPopulation Estimation:")
print(f"  Total observed cases: {total_cases_observed}")
print(f"  Total infections (corrected): {total_infected_estimated:.0f}")
print(f"  Estimated population N: {N_estimated:.0f}")

# Calculate fraction infected by peak
true_cum_infections_at_peak = cum_cases_at_peak / ascertainment
f_star = true_cum_infections_at_peak / N_estimated

print(f"\nFraction infected by peak:")
print(f"  f* = {f_star:.4f}")

# Calculate R₀: R₀ = 1/(1-f*)
R0_peak = 1 / (1 - f_star)

print(f"\nR₀ Estimate:")
print(f"  R₀ = 1/(1-f*) = {R0_peak:.3f}")

# Confidence interval (delta method)
se_cum_peak = np.sqrt(cum_cases_at_peak)
se_total = np.sqrt(total_cases_observed)
se_sero = np.sqrt(seroprevalence * (1 - seroprevalence) / 1000)

# Numerical derivatives
def R0_from_params(cum_peak, total, sero, asc):
    f = (cum_peak / asc) / (total / asc / sero)
    return 1 / (1 - f)

delta = 1e-6
dR0_dcum = (R0_from_params(cum_cases_at_peak + delta, total_cases_observed, 
                            seroprevalence, ascertainment) - R0_peak) / delta
dR0_dtotal = (R0_from_params(cum_cases_at_peak, total_cases_observed + delta, 
                              seroprevalence, ascertainment) - R0_peak) / delta
dR0_dsero = (R0_from_params(cum_cases_at_peak, total_cases_observed, 
                             seroprevalence + delta, ascertainment) - R0_peak) / delta
dR0_dasc = (R0_from_params(cum_cases_at_peak, total_cases_observed, 
                            seroprevalence, ascertainment + delta) - R0_peak) / delta

var_cum = se_cum_peak**2
var_total = se_total**2
var_sero = se_sero**2
var_asc = (0.20 * ascertainment)**2  # 20% relative uncertainty

var_R0 = (dR0_dcum**2 * var_cum + 
          dR0_dtotal**2 * var_total +
          dR0_dsero**2 * var_sero +
          dR0_dasc**2 * var_asc)

se_R0 = np.sqrt(var_R0)
R0_ci_lower = R0_peak - 1.96 * se_R0
R0_ci_upper = R0_peak + 1.96 * se_R0

print(f"  95% CI: [{R0_ci_lower:.3f}, {R0_ci_upper:.3f}]")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Epidemic curve with peak
axes[0].plot(df['Week'], df['New Cases'], 'b-', linewidth=2)
axes[0].axvline(peak_week, color='red', linestyle='--', linewidth=2, label=f'Peak (week {peak_week})')
axes[0].scatter([peak_week], [peak_cases], color='red', s=200, zorder=5, edgecolor='darkred', linewidth=2)
axes[0].set_xlabel('Week', fontsize=11)
axes[0].set_ylabel('New Cases', fontsize=11)
axes[0].set_title('Epidemic Curve with Peak', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Cumulative cases
axes[1].plot(df['Week'], df['Cumulative Cases'], 'g-', linewidth=2)
axes[1].axvline(peak_week, color='red', linestyle='--', linewidth=2, label=f'Peak (week {peak_week})')
axes[1].axhline(cum_cases_at_peak, color='orange', linestyle=':', linewidth=2, 
            label=f'Cases at peak: {cum_cases_at_peak}')
axes[1].scatter([peak_week], [cum_cases_at_peak], color='red', s=200, zorder=5, 
            edgecolor='darkred', linewidth=2)
axes[1].set_xlabel('Week', fontsize=11)
axes[1].set_ylabel('Cumulative Cases', fontsize=11)
axes[1].set_title('Cumulative Incidence', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('Q1_partC_peak_analysis.png', dpi=300, bbox_inches='tight')
print("\nFigure saved: Q1_partC_peak_analysis.png")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Peak week: {peak_week}")
print(f"Fraction infected by peak: f* = {f_star:.4f}")
print(f"R₀ = {R0_peak:.3f} [{R0_ci_lower:.3f}, {R0_ci_upper:.3f}]")
print("="*70)
