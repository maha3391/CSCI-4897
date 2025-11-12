import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

print("="*70)
print("Q1 Part B - Method 2: Endemic Equilibrium")
print("="*70)

# Parameters
gamma = 1/2  # Recovery rate (1/2 per week)
mu = 1/100   # Death rate (1/100 per week)

# Study data
n_prevalence = 1000
positive_prevalence = 7

prevalence = positive_prevalence / n_prevalence

print(f"\nData:")
print(f"  Prevalence: {positive_prevalence}/{n_prevalence} = {prevalence:.3f}")

# ==============================================================================
# METHOD 2: Endemic Equilibrium (from Prevalence)
# ==============================================================================
print("\n" + "="*70)
print("METHOD 2: Endemic Equilibrium (Week 9 Lecture)")
print("="*70)
print("Theory: At endemic equilibrium, i_eq = (1 - 1/R₀) × μ/(γ + μ)")
print("        Solving for R₀: R₀ = 1 / (1 - i_eq × (γ/μ + 1))")

# Step 1: Prevalence estimate and CI
i_eq = prevalence
print(f"\nStep 1: Prevalence Estimate")
print(f"  Point estimate: p = {positive_prevalence}/{n_prevalence} = {prevalence:.4f}")

# Binomial CI for prevalence
prevalence_se = np.sqrt(prevalence * (1 - prevalence) / n_prevalence)
prevalence_ci_lower = prevalence - 1.96 * prevalence_se
prevalence_ci_upper = prevalence + 1.96 * prevalence_se
print(f"  Standard error: SE = √(p(1-p)/n) = {prevalence_se:.6f}")
print(f"  95% CI: [{prevalence_ci_lower:.4f}, {prevalence_ci_upper:.4f}]")

# Step 2: Calculate R₀
print(f"\nStep 2: Calculate R₀")
print(f"  γ/μ = {gamma/mu:.2f}")
print(f"  R₀ = 1 / (1 - {i_eq:.4f} × ({gamma/mu:.2f} + 1))")
R0_prev = 1 / (1 - i_eq * (gamma/mu + 1))
print(f"  R₀ = {R0_prev:.3f}")

# Step 3: Propagate uncertainty (delta method)
print(f"\nStep 3: Propagate Uncertainty to R₀ (Delta Method)")
print(f"  dR₀/di = (γ/μ + 1) / (1 - i(γ/μ + 1))²")
dR0_di = (gamma/mu + 1) / (1 - i_eq * (gamma/mu + 1))**2
print(f"  dR₀/di = {dR0_di:.2f}")
R0_prev_se = abs(dR0_di) * prevalence_se
print(f"  SE(R₀) = |dR₀/di| × SE(i) = {R0_prev_se:.4f}")

R0_prev_ci_lower = R0_prev - 1.96 * R0_prev_se
R0_prev_ci_upper = R0_prev + 1.96 * R0_prev_se
print(f"\n  Final: R₀ = {R0_prev:.3f} [95% CI: {R0_prev_ci_lower:.3f}, {R0_prev_ci_upper:.3f}]")

# ==============================================================================
# VISUALIZATION
# ==============================================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Single estimate with error bar
ax1 = axes[0]
ax1.errorbar(0, R0_prev, yerr=[[R0_prev - R0_prev_ci_lower], [R0_prev_ci_upper - R0_prev]], 
             fmt='o', color='orange', markersize=15, capsize=15, capthick=3, linewidth=3,
             label=f'R₀ = {R0_prev:.3f}')
ax1.set_xlim([-0.5, 0.5])
ax1.set_ylim([0, 3])
ax1.set_xticks([0])
ax1.set_xticklabels(['Method 2\n(Endemic Equilibrium)'], fontsize=12)
ax1.set_ylabel('R₀ Estimate', fontsize=13)
ax1.set_title('Method 2: R₀ from Prevalence', fontsize=14, fontweight='bold')
ax1.axhline(1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, label='Epidemic threshold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.legend(fontsize=11)

# Plot 2: Comparison with Method 1
ax2 = axes[1]
methods = ['Method 1\n(Exponential)', 'Method 2\n(Equilibrium)']
R0_vals = [1.870, R0_prev]
ci_lowers = [1.796, R0_prev_ci_lower]
ci_uppers = [1.943, R0_prev_ci_upper]
colors = ['blue', 'orange']

x_pos = np.arange(len(methods))
for i, (r0, low, high, color) in enumerate(zip(R0_vals, ci_lowers, ci_uppers, colors)):
    ax2.errorbar(i, r0, yerr=[[r0-low], [high-r0]], fmt='o', color=color,
                 markersize=12, capsize=10, capthick=2.5, linewidth=2.5)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(methods, fontsize=11)
ax2.set_ylabel('R₀ Estimate', fontsize=13)
ax2.set_title('Comparison: Methods 1 vs 2', fontsize=14, fontweight='bold')
ax2.axhline(1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 3])

plt.tight_layout()
plt.savefig('Q1_partB_method2.png', dpi=300, bbox_inches='tight')
print("Figure saved: Q1_partB_method2.png")
print("="*70)
