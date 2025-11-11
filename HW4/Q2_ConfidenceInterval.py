import statsmodels.stats.proportion as smp

def corrected_prevalence_CI(x, n, Se, Sp, alpha=0.05):
    p_obs = x/n
    ci_low, ci_high = smp.proportion_confint(x, n, alpha=alpha, method='wilson')
    p_corr = (p_obs + Sp - 1)/(Se + Sp - 1)
    ci_corr = [(ci_low + Sp - 1)/(Se + Sp - 1),
               (ci_high + Sp - 1)/(Se + Sp - 1)]
    return p_corr, ci_corr

print(corrected_prevalence_CI(39, 100, 0.9, 0.98))
