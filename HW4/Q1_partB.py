import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

print("="*70)
print("Q1 Part B - Methods 2, 4A, 4B")
print("="*70)

# Parameters
gamma = 1/2  # Recovery rate (1/2 per week)
mu = 1/100   # Death rate (1/100 per week)
lifespan = 100  # weeks

# Study data
n_prevalence = 1000
positive_prevalence = 7
n_sero = 1000
positive_sero = 517

prevalence = positive_prevalence / n_prevalence
seroprevalence = positive_sero / n_sero

print(f"\nData:")
print(f"  Prevalence: {positive_prevalence}/{n_prevalence} = {prevalence:.3f}")
print(f"  Seroprevalence: {positive_sero}/{n_sero} = {seroprevalence:.3f}")

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
# METHOD 4A: Final Size Relation (from Seroprevalence)
# ==============================================================================
print("\n" + "="*70)
print("METHOD 4A: Final Size Relation (Week 9 Lecture)")
print("="*70)
print("Theory: Final size equation z = 1 - exp(-R₀ × z)")
print("        Solving: R₀ = -ln(1-z)/z where z = seroprevalence")

# Step 1: Seroprevalence estimate and CI
z = seroprevalence
print(f"\nStep 1: Seroprevalence Estimate")
print(f"  Point estimate: z = {positive_sero}/{n_sero} = {seroprevalence:.4f}")

sero_se = np.sqrt(seroprevalence * (1 - seroprevalence) / n_sero)
sero_ci_lower = seroprevalence - 1.96 * sero_se
sero_ci_upper = seroprevalence + 1.96 * sero_se
print(f"  Standard error: SE = √(z(1-z)/n) = {sero_se:.6f}")
print(f"  95% CI: [{sero_ci_lower:.4f}, {sero_ci_upper:.4f}]")

# Step 2: Calculate R₀
print(f"\nStep 2: Calculate R₀")
print(f"  R₀ = -ln(1-z)/z = -ln(1-{z:.4f})/{z:.4f}")
R0_sero = -np.log(1 - z) / z
print(f"  R₀ = {R0_sero:.3f}")

# Step 3: Propagate uncertainty (delta method)
print(f"\nStep 3: Propagate Uncertainty to R₀ (Delta Method)")
print(f"  dR₀/dz = (z - (1-z)ln(1-z)) / (z²(1-z))")
dR0_dz = (z - (1-z)*np.log(1-z)) / (z**2 * (1-z))
print(f"  dR₀/dz = {dR0_dz:.4f}")
R0_sero_se = abs(dR0_dz) * sero_se
print(f"  SE(R₀) = |dR₀/dz| × SE(z) = {R0_sero_se:.4f}")

R0_sero_ci_lower = R0_sero - 1.96 * R0_sero_se
R0_sero_ci_upper = R0_sero + 1.96 * R0_sero_se
print(f"\n  Final: R₀ = {R0_sero:.3f} [95% CI: {R0_sero_ci_lower:.3f}, {R0_sero_ci_upper:.3f}]")

# ==============================================================================
# METHOD 4B: Force of Infection (from Seroprevalence)
# ==============================================================================
print("\n" + "="*70)
print("METHOD 4B: Force of Infection (Week 9 Lecture)")
print("="*70)
print("Theory: Under endemic conditions, sero ≈ 1 - exp(-λL/2)")
print("        where λ = force of infection, L = lifespan")
print("        Then: R₀ = λ × L")

# Step 1: Use seroprevalence from above
print(f"\nStep 1: Seroprevalence (from above)")
print(f"  z = {seroprevalence:.4f} [95% CI: {sero_ci_lower:.4f}, {sero_ci_upper:.4f}]")

# Step 2: Solve for λ
print(f"\nStep 2: Solve for Force of Infection λ")
print(f"  sero = 1 - exp(-λL/2)")
print(f"  λ = -2ln(1-sero)/L = -2ln(1-{z:.4f})/{lifespan}")
lambda_foi = -2 * np.log(1 - z) / lifespan
print(f"  λ = {lambda_foi:.6f} per week")

# Step 3: Calculate R₀
print(f"\nStep 3: Calculate R₀")
print(f"  R₀ = λ × L = {lambda_foi:.6f} × {lifespan}")
R0_foi = lambda_foi * lifespan
print(f"  R₀ = {R0_foi:.3f}")

# Step 4: Propagate uncertainty
print(f"\nStep 4: Propagate Uncertainty to R₀ (Delta Method)")
print(f"  dR₀/dz = d(λL)/dz = 2L/(1-z) / L = 2/(1-z)")
dR0_dz_foi = 2 / (1 - z)
print(f"  dR₀/dz = {dR0_dz_foi:.4f}")
R0_foi_se = abs(dR0_dz_foi) * sero_se
print(f"  SE(R₀) = |dR₀/dz| × SE(z) = {R0_foi_se:.4f}")

R0_foi_ci_lower = R0_foi - 1.96 * R0_foi_se
R0_foi_ci_upper = R0_foi + 1.96 * R0_foi_se
print(f"\n  Final: R₀ = {R0_foi:.3f} [95% CI: {R0_foi_ci_lower:.3f}, {R0_foi_ci_upper:.3f}]")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Method 2 (Equilibrium):  R₀ = {R0_prev:.3f} [{R0_prev_ci_lower:.3f}, {R0_prev_ci_upper:.3f}]")
print(f"Method 4A (Final Size):  R₀ = {R0_sero:.3f} [{R0_sero_ci_lower:.3f}, {R0_sero_ci_upper:.3f}]")
print(f"Method 4B (FOI):         R₀ = {R0_foi:.3f} [{R0_foi_ci_lower:.3f}, {R0_foi_ci_upper:.3f}]")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
methods = ['Method 2\n(Equilibrium)', 'Method 4A\n(Final Size)', 'Method 4B\n(FOI)']
R0_values = [R0_prev, R0_sero, R0_foi]
ci_lower = [R0_prev_ci_lower, R0_sero_ci_lower, R0_foi_ci_lower]
ci_upper = [R0_prev_ci_upper, R0_sero_ci_upper, R0_foi_ci_upper]
colors = ['orange', 'green', 'purple']

x_pos = np.arange(len(methods))
for i, (r0, low, high, color) in enumerate(zip(R0_values, ci_lower, ci_upper, colors)):
    ax.errorbar(i, r0, yerr=[[r0-low], [high-r0]], fmt='o', color=color, 
                 markersize=12, capsize=10, capthick=2, linewidth=2.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel('R₀ Estimate', fontsize=12)
ax.set_title('R₀ Estimates from Prevalence/Seroprevalence', fontsize=13, fontweight='bold')
ax.axhline(1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.5, 2.5])

plt.tight_layout()
plt.savefig('Q1_partB_estimates.png', dpi=300, bbox_inches='tight')
print("\nFigure saved: Q1_partB_estimates.png")
print("="*70)
