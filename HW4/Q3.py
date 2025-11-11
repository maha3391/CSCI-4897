import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

print("="*70)
print("SENSITIVITY, SPECIFICITY, AND PREVALENCE ANALYSIS")
print("="*70)

# ============================================================================
# PART A: Read data and create visualization
# ============================================================================

# Read the three CSV files
neg_controls = pd.read_csv('HW4_Q3_neg-1.csv', header=None).values.flatten()
pos_controls = pd.read_csv('HW4_Q3_pos-1.csv', header=None).values.flatten()
field_data = pd.read_csv('HW4_Q3_data-1.csv', header=None).values.flatten()

print(f"\nData Summary:")
print(f"- Negative controls: {len(neg_controls)} samples")
print(f"  Mean: {np.mean(neg_controls):.2f}, Std: {np.std(neg_controls):.2f}")
print(f"  Range: [{np.min(neg_controls):.2f}, {np.max(neg_controls):.2f}]")
print(f"\n- Positive controls: {len(pos_controls)} samples")
print(f"  Mean: {np.mean(pos_controls):.2f}, Std: {np.std(pos_controls):.2f}")
print(f"  Range: [{np.min(pos_controls):.2f}, {np.max(pos_controls):.2f}]")
print(f"\n- Field data: {len(field_data)} samples")
print(f"  Mean: {np.mean(field_data):.2f}, Std: {np.std(field_data):.2f}")
print(f"  Range: [{np.min(field_data):.2f}, {np.max(field_data):.2f}]")

# ============================================================================
# PART B: Define functions for sensitivity, specificity, and prevalence
# ============================================================================

def sensitivity(c, pos_data=pos_controls):
    """
    Sensitivity: P(test+ | disease+)
    Fraction of positive controls that test positive (above cutoff c)
    """
    return np.sum(pos_data >= c) / len(pos_data)

def specificity(c, neg_data=neg_controls):
    """
    Specificity: P(test- | disease-)
    Fraction of negative controls that test negative (below cutoff c)
    """
    return np.sum(neg_data < c) / len(neg_data)

def raw_prevalence(c, data=field_data):
    """
    Raw (apparent) prevalence: φ̂(c)
    Fraction of field samples that test positive
    """
    return np.sum(data >= c) / len(data)

def corrected_prevalence(c, data=field_data, pos_data=pos_controls, neg_data=neg_controls):
    """
    Corrected prevalence: θ̂(c)
    Using Rogan-Gladen estimator: θ = (φ + Sp - 1) / (Se + Sp - 1)
    """
    phi = raw_prevalence(c, data)
    se = sensitivity(c, pos_data)
    sp = specificity(c, neg_data)
    
    denominator = se + sp - 1
    if abs(denominator) < 1e-10:  # Avoid division by zero
        return np.nan
    
    theta = (phi + sp - 1) / denominator
    return theta

def youden_index(c, pos_data=pos_controls, neg_data=neg_controls):
    """
    Youden's J statistic: J = Se + Sp - 1
    Maximizes the vertical distance in ROC curve
    """
    return sensitivity(c, pos_data) + specificity(c, neg_data) - 1

# Find optimal Youden cutoff
cutoff_range = np.linspace(
    min(np.min(neg_controls), np.min(pos_controls), np.min(field_data)) - 1,
    max(np.max(neg_controls), np.max(pos_controls), np.max(field_data)) + 1,
    1000
)

youden_values = [youden_index(c) for c in cutoff_range]
youden_cutoff = cutoff_range[np.argmax(youden_values)]
youden_max = np.max(youden_values)

print(f"\n" + "="*70)
print("YOUDEN'S INDEX ANALYSIS")
print("="*70)
print(f"Optimal cutoff (Youden): c = {youden_cutoff:.3f}")
print(f"Youden's J at optimal c: {youden_max:.4f}")
print(f"\nAt Youden cutoff c = {youden_cutoff:.3f}:")
print(f"  Sensitivity: {sensitivity(youden_cutoff):.4f} ({sensitivity(youden_cutoff)*100:.2f}%)")
print(f"  Specificity: {specificity(youden_cutoff):.4f} ({specificity(youden_cutoff)*100:.2f}%)")
print(f"  Raw prevalence: {raw_prevalence(youden_cutoff):.4f} ({raw_prevalence(youden_cutoff)*100:.2f}%)")
print(f"  Corrected prevalence: {corrected_prevalence(youden_cutoff):.4f} ({corrected_prevalence(youden_cutoff)*100:.2f}%)")

# ============================================================================
# PART A: Visualization - Tall skinny plot
# ============================================================================

fig = plt.figure(figsize=(12, 8))

# Create jittered x-coordinates for better visualization
np.random.seed(42)
jitter_amount = 0.15

x_neg = np.zeros(len(neg_controls)) + np.random.uniform(-jitter_amount, jitter_amount, len(neg_controls))
x_pos = np.ones(len(pos_controls)) + np.random.uniform(-jitter_amount, jitter_amount, len(pos_controls))
x_field = 2 * np.ones(len(field_data)) + np.random.uniform(-jitter_amount, jitter_amount, len(field_data))

# Main plot
ax1 = plt.subplot(1, 1, 1)
ax1.scatter(x_neg, neg_controls, c='red', alpha=0.3, s=30, label='Negative Controls')
ax1.scatter(x_pos, pos_controls, c='black', alpha=0.3, s=30, label='Positive Controls')
ax1.scatter(x_field, field_data, c='blue', alpha=0.3, s=30, label='Field Data')

# Add horizontal line at Youden cutoff
ax1.axhline(y=youden_cutoff, color='green', linestyle='--', linewidth=2, 
            label=f'Youden Cutoff = {youden_cutoff:.2f}')

# Add shaded regions
y_min = min(np.min(neg_controls), np.min(pos_controls), np.min(field_data)) - 2
y_max = max(np.max(neg_controls), np.max(pos_controls), np.max(field_data)) + 2
ax1.fill_between([-0.5, 2.5], youden_cutoff, y_max, alpha=0.1, color='orange', 
                  label='Called Positive (above cutoff)')
ax1.fill_between([-0.5, 2.5], y_min, youden_cutoff, alpha=0.1, color='lightblue', 
                  label='Called Negative (below cutoff)')

ax1.set_xticks([0, 1, 2])
ax1.set_xticklabels(['Negative\nControls', 'Positive\nControls', 'Field\nData'])
ax1.set_ylabel('Assay Value')
ax1.set_title('Distribution of Assay Values: Controls and Field Data')
ax1.set_xlim(-0.5, 2.5)
ax1.set_ylim(y_min, y_max)
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('Q3_partA_distributions.png', dpi=300, bbox_inches='tight')
print("\nPart A plot saved as 'Q3_partA_distributions.png'")

# ============================================================================
# PART C: ROC Curve and Corrected Prevalence vs Cutoff
# ============================================================================

# Calculate Se, Sp, and prevalences for range of cutoffs
se_values = [sensitivity(c) for c in cutoff_range]
sp_values = [specificity(c) for c in cutoff_range]
raw_prev_values = [raw_prevalence(c) for c in cutoff_range]
corr_prev_values = [corrected_prevalence(c) for c in cutoff_range]

# For ROC curve, we need 1 - Specificity (False Positive Rate)
fpr_values = [1 - sp for sp in sp_values]

# Values at Youden cutoff
se_youden = sensitivity(youden_cutoff)
sp_youden = specificity(youden_cutoff)
fpr_youden = 1 - sp_youden
raw_youden = raw_prevalence(youden_cutoff)
corr_youden = corrected_prevalence(youden_cutoff)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
ax1 = axes[0]
ax1.plot(fpr_values, se_values, 'b-', linewidth=2, label='ROC Curve')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
ax1.plot(fpr_youden, se_youden, 'ro', markersize=12, 
         label=f'Youden (c={youden_cutoff:.2f})')

# Add annotation for Youden point
ax1.annotate(f'Youden\nc={youden_cutoff:.2f}\nSe={se_youden:.3f}\nSp={sp_youden:.3f}',
             xy=(fpr_youden, se_youden), xytext=(fpr_youden+0.15, se_youden-0.15),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax1.set_xlabel('False Positive Rate (1 - Specificity)')
ax1.set_ylabel('True Positive Rate (Sensitivity)')
ax1.set_title('Receiver Operating Characteristic (ROC) Curve')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)

# Calculate AUC
auc = np.trapz(se_values, fpr_values)
ax1.text(0.6, 0.2, f'AUC = {abs(auc):.3f}', fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Corrected Prevalence vs Cutoff
ax2 = axes[1]
ax2.plot(cutoff_range, corr_prev_values, 'b-', linewidth=2, label='Corrected Prevalence θ̂(c)')
ax2.plot(cutoff_range, raw_prev_values, 'g--', linewidth=2, alpha=0.6, label='Raw Prevalence φ̂(c)')
ax2.plot(youden_cutoff, corr_youden, 'ro', markersize=12, 
         label=f'Youden (c={youden_cutoff:.2f})')

# Add annotation for Youden point
ax2.annotate(f'Youden\nc={youden_cutoff:.2f}\nθ̂={corr_youden:.3f}',
             xy=(youden_cutoff, corr_youden), xytext=(youden_cutoff+3, corr_youden+0.05),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.axhline(y=1, color='k', linestyle='-', alpha=0.3)
ax2.axvline(x=youden_cutoff, color='red', linestyle=':', alpha=0.5)
ax2.set_xlabel('Cutoff Value (c)')
ax2.set_ylabel('Prevalence Estimate')
ax2.set_title('Prevalence Estimates vs Cutoff Value')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Q3_partC_ROC_and_prevalence.png', dpi=300, bbox_inches='tight')
print("Part C plot saved as 'Q3_partC_ROC_and_prevalence.png'")

# ============================================================================
# Additional Analysis: Se, Sp vs Cutoff
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cutoff_range, se_values, 'b-', linewidth=2, label='Sensitivity')
ax.plot(cutoff_range, sp_values, 'r-', linewidth=2, label='Specificity')
ax.plot(cutoff_range, youden_values, 'g-', linewidth=2, label="Youden's J (Se + Sp - 1)")
ax.axvline(x=youden_cutoff, color='purple', linestyle='--', linewidth=2, 
           label=f'Youden Cutoff = {youden_cutoff:.2f}')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axhline(y=1, color='k', linestyle='-', alpha=0.3)

ax.set_xlabel('Cutoff Value (c)')
ax.set_ylabel('Probability')
ax.set_title('Sensitivity and Specificity vs Cutoff Value')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('Q3_Se_Sp_vs_cutoff.png', dpi=300, bbox_inches='tight')
print("Additional plot saved as 'Q3_Se_Sp_vs_cutoff.png'")

# ============================================================================
# Explore different cutoff choices
# ============================================================================

print(f"\n" + "="*70)
print("COMPARISON OF DIFFERENT CUTOFF STRATEGIES")
print("="*70)

# Different cutoff strategies
cutoffs_to_compare = {
    'Very Low (5th percentile neg)': np.percentile(neg_controls, 5),
    'Mean of negatives': np.mean(neg_controls),
    'Midpoint (mean of means)': (np.mean(neg_controls) + np.mean(pos_controls)) / 2,
    'Youden Optimal': youden_cutoff,
    'High Specificity (99%)': cutoff_range[np.argmin(np.abs(np.array(sp_values) - 0.99))],
    'High Sensitivity (99%)': cutoff_range[np.argmin(np.abs(np.array(se_values) - 0.99))],
}

print(f"\n{'Strategy':<30} {'Cutoff':>8} {'Se':>8} {'Sp':>8} {'Raw %':>8} {'Corr %':>8}")
print("-" * 70)
for name, c in cutoffs_to_compare.items():
    se = sensitivity(c)
    sp = specificity(c)
    raw = raw_prevalence(c)
    corr = corrected_prevalence(c)
    print(f"{name:<30} {c:>8.2f} {se:>8.3f} {sp:>8.3f} {raw:>8.3f} {corr:>8.3f}")

