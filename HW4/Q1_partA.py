import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress

# Read the data
df = pd.read_csv('HW4_all_weeks.csv')

print("="*70)
print("Q1 Part A - Method 1: Exponential Growth")
print("="*70)

# BRUH Disease parameters
gamma = 1/2  # Recovery rate = 1/infection_duration = 1/2 per week
mu = 1/100   # Death rate = 1/lifespan = 1/100 per week

print(f"\nParameters:")
print(f"  γ = {gamma} per week")
print(f"  μ = {mu} per week")
print(f"  γ + μ = {gamma + mu} per week")

# Find exponential growth phase
peak_week = df.loc[df['New Cases'].idxmax(), 'Week']
early_data = df[df['Week'] <= peak_week].copy()
early_data['log_cases'] = np.log(early_data['New Cases'])

# Test different windows to find best exponential fit
print("\nFinding best exponential growth window...")
best_r2 = 0
best_range = None
best_slope = None

for start_week in range(1, 15):
    for end_week in range(start_week + 4, min(20, peak_week + 1)):
        subset = early_data[(early_data['Week'] >= start_week) & 
                           (early_data['Week'] <= end_week)].copy()
        
        if len(subset) < 5:
            continue
            
        slope, intercept, r_value, p_value, std_err = linregress(subset['Week'], subset['log_cases'])
        r_squared = r_value**2
        
        if r_squared > best_r2 and r_squared > 0.95:
            best_r2 = r_squared
            best_range = (start_week, end_week)
            best_slope = slope

print(f"Best fit: Weeks {best_range[0]}-{best_range[1]}, R² = {best_r2:.4f}")

# Regression analysis on best window
subset = early_data[(early_data['Week'] >= best_range[0]) & 
                   (early_data['Week'] <= best_range[1])].copy()

slope, intercept, r_value, p_value, std_err = linregress(subset['Week'], subset['log_cases'])

print(f"\n" + "="*70)
print("LINEAR REGRESSION ON ln(CASES) vs TIME")
print("="*70)
print(f"Sample size: n = {len(subset)}")
print(f"Slope (m): {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R² value: {r_value**2:.4f}")
print(f"Standard error of slope: {std_err:.4f}")
print(f"p-value: {p_value:.6f}")

# Confidence interval for slope
n = len(subset)
df_residual = n - 2
t_critical = stats.t.ppf(0.975, df=df_residual)

print(f"\n" + "="*70)
print("95% CONFIDENCE INTERVAL FOR SLOPE")
print("="*70)
print(f"Degrees of freedom: {df_residual}")
print(f"t-critical (α=0.05, two-tailed): {t_critical:.4f}")
print(f"Margin of error: {t_critical * std_err:.4f}")
print(f"\nSlope 95% CI: [{slope - t_critical * std_err:.4f}, {slope + t_critical * std_err:.4f}]")

slope_ci_lower = slope - t_critical * std_err
slope_ci_upper = slope + t_critical * std_err

# Calculate R₀: R₀ = 1 + m/(γ+μ)
print(f"\n" + "="*70)
print("R₀ CALCULATION FROM SLOPE")
print("="*70)
print(f"Theory: I(t) ≈ I(0)exp((R₀-1)(γ+μ)t)")
print(f"        ln(I(t)) ≈ ln(I(0)) + (R₀-1)(γ+μ)t")
print(f"        Slope m = (R₀-1)(γ+μ)")
print(f"        Therefore: R₀ = 1 + m/(γ+μ)")
print(f"\nPoint estimate:")
print(f"  R₀ = 1 + {slope:.4f}/{gamma + mu:.4f}")

R0_estimate = 1 + slope / (gamma + mu)
R0_ci_lower = 1 + slope_ci_lower / (gamma + mu)
R0_ci_upper = 1 + slope_ci_upper / (gamma + mu)

print(f"  R₀ = {R0_estimate:.3f}")
print(f"\n95% Confidence Interval (from slope CI):")
print(f"  Lower: R₀ = 1 + {slope_ci_lower:.4f}/{gamma + mu:.4f} = {R0_ci_lower:.3f}")
print(f"  Upper: R₀ = 1 + {slope_ci_upper:.4f}/{gamma + mu:.4f} = {R0_ci_upper:.3f}")
print(f"\nFinal Result: R₀ = {R0_estimate:.3f} [95% CI: {R0_ci_lower:.3f}, {R0_ci_upper:.3f}]")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original scale
axes[0].plot(subset['Week'], subset['New Cases'], 'ro', markersize=8, label='Data')
x_fit = np.linspace(subset['Week'].min(), subset['Week'].max(), 100)
y_fit = np.exp(slope * x_fit + intercept)
axes[0].plot(x_fit, y_fit, 'b-', linewidth=2, label=f'Fit (R²={r_value**2:.3f})')
axes[0].set_xlabel('Week')
axes[0].set_ylabel('New Cases')
axes[0].set_title(f'Exponential Growth (Weeks {best_range[0]}-{best_range[1]})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Log scale
axes[1].plot(subset['Week'], subset['log_cases'], 'ro', markersize=8, label='ln(Cases)')
axes[1].plot(x_fit, slope * x_fit + intercept, 'b-', linewidth=2, label=f'Slope={slope:.3f}')
axes[1].set_xlabel('Week')
axes[1].set_ylabel('ln(New Cases)')
axes[1].set_title('Log-Linear Fit')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Q1_partA_exponential_growth.png', dpi=300, bbox_inches='tight')
print("\nFigure saved: Q1_partA_exponential_growth.png")

print("="*70)
