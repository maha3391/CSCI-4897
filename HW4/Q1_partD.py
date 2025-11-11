import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*70)
print("Q1 Part D - Comparison of R₀ Estimation Methods")
print("="*70)

# Collect all R₀ estimates from previous parts
methods = {
    'Method 1\n(Exponential Growth)': {'R0': 1.870, 'CI_lower': 1.796, 'CI_upper': 1.943},
    'Method 2\n(Equilibrium)': {'R0': 1.555, 'CI_lower': 0.918, 'CI_upper': 2.193},
    'Method 3\n(Peak Incidence)': {'R0': 1.071, 'CI_lower': 1.066, 'CI_upper': 1.076},
    'Method 4A\n(Final Size)': {'R0': 1.408, 'CI_lower': 1.199, 'CI_upper': 1.616},
    'Method 4B\n(Force of Infection)': {'R0': 1.455, 'CI_lower': 1.327, 'CI_upper': 1.584},
}

print("\nSummary of R₀ Estimates:")
print("-"*70)
for method, data in methods.items():
    method_name = method.replace('\n', ' ')
    ci_width = data['CI_upper'] - data['CI_lower']
    print(f"{method_name:40s} R₀ = {data['R0']:.3f} [{data['CI_lower']:.3f}, {data['CI_upper']:.3f}]  (CI width: {ci_width:.3f})")

# Calculate statistics
R0_values = [data['R0'] for data in methods.values()]
mean_R0 = np.mean(R0_values)
median_R0 = np.median(R0_values)
std_R0 = np.std(R0_values)

print(f"\nAggregate Statistics:")
print(f"  Mean: {mean_R0:.3f}")
print(f"  Median: {median_R0:.3f}")
print(f"  Std Dev: {std_R0:.3f}")
print(f"  Range: [{min(R0_values):.3f}, {max(R0_values):.3f}]")

# Discussion
print("\n" + "="*70)
print("DISCUSSION: Why Do Estimates Differ?")
print("="*70)

print("\n1. TEMPORAL DIFFERENCES:")
print("   - Method 1 (R₀=1.87): Early epidemic phase (weeks 9-14)")
print("     → Captures 'intrinsic' transmission when S≈N")
print("   - Methods 2, 4A, 4B (R₀≈1.4-1.6): Later/equilibrium phase")
print("     → Reduced by susceptible depletion")
print("   - Method 3 (R₀=1.07): Peak timing (week 17)")
print("     → Only 6.6% infected by peak suggests lower R₀")

print("\n2. METHODOLOGICAL ASSUMPTIONS:")
print("   - Method 1: Assumes exponential growth, S≈N")
print("   - Method 2: Assumes endemic equilibrium (questionable)")
print("   - Method 3: Assumes SIR peak dynamics")
print("   - Method 4A: Assumes completed epidemic")
print("   - Method 4B: Assumes endemic steady state")

print("\n3. UNCERTAINTY DIFFERENCES:")
print("   - Method 2: WIDEST CI (1.275) - very sensitive to small prevalence")
print("   - Method 3: NARROWEST CI (0.010) - but may be biased")
print("   - Methods 1, 4A, 4B: Moderate uncertainty (0.15-0.42)")

print("\n4. DATA SOURCE DIFFERENCES:")
print("   - Methods 1, 3: Use incidence time series")
print("   - Method 2: Uses prevalence (7/1000)")
print("   - Methods 4A, 4B: Use seroprevalence (517/1000)")

print("\n5. KEY INSIGHTS:")
print("   - High seroprevalence (51.7%) suggests major epidemic occurred")
print("   - Low current prevalence (0.7%) suggests epidemic may be waning")
print("   - Method 3's low estimate (1.07) seems inconsistent with other methods")
print("   - Methods 1 and 4A/4B bracket likely range: R₀ ≈ 1.4-1.9")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nMost reliable estimates:")
print("  • Method 1 (Exponential Growth): R₀ = 1.87 [1.80, 1.94]")
print("    - Strong statistical fit (R²=0.996)")
print("    - Clear exponential phase identified")
print("    - Reflects initial transmission potential")
print()
print("  • Method 4A (Final Size): R₀ = 1.41 [1.20, 1.62]")
print("    - Based on large seroprevalence sample")
print("    - Reasonable if epidemic is complete")
print("    - Reflects average transmission over epidemic")
print()
print("Best estimate range: R₀ ≈ 1.4-1.9")
print("  (Lower bound from final size, upper bound from exponential growth)")
print("="*70)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: R₀ estimates with confidence intervals
ax1 = axes[0]
method_names = list(methods.keys())
R0_vals = [methods[m]['R0'] for m in method_names]
ci_lowers = [methods[m]['CI_lower'] for m in method_names]
ci_uppers = [methods[m]['CI_upper'] for m in method_names]
colors = ['blue', 'orange', 'red', 'green', 'purple']

x_pos = np.arange(len(method_names))
for i, (r0, low, high, color) in enumerate(zip(R0_vals, ci_lowers, ci_uppers, colors)):
    ax1.errorbar(i, r0, yerr=[[r0-low], [high-r0]], fmt='o', color=color, 
                 markersize=12, capsize=10, capthick=2, linewidth=2.5, label=method_names[i].replace('\n', ' '))

ax1.axhline(mean_R0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Mean: {mean_R0:.2f}')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(method_names, fontsize=9)
ax1.set_ylabel('R₀ Estimate', fontsize=12)
ax1.set_title('Comparison of R₀ Estimation Methods', fontsize=13, fontweight='bold')
ax1.axhline(1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0.5, 2.5])

# Plot 2: CI widths comparison
ax2 = axes[1]
ci_widths = [ci_uppers[i] - ci_lowers[i] for i in range(len(method_names))]
bars = ax2.barh(x_pos, ci_widths, color=colors, alpha=0.7)
ax2.set_yticks(x_pos)
ax2.set_yticklabels(method_names, fontsize=9)
ax2.set_xlabel('95% Confidence Interval Width', fontsize=12)
ax2.set_title('Uncertainty Comparison', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# Add values on bars
for i, (width, bar) in enumerate(zip(ci_widths, bars)):
    ax2.text(width + 0.05, i, f'{width:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('Q1_partD_comparison.png', dpi=300, bbox_inches='tight')
print("\nFigure saved: Q1_partD_comparison.png")
