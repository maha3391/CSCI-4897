import numpy as np

# Symmetrized contact matrices from part b
C_pemic = np.array([[3.1, 42.9],
                    [4.77, 25.0]])

C_uensa = np.array([[3.0, 42.25],
                    [5.07, 25.1]])

# Compute eigenvalues
eig_pemic = np.linalg.eigvals(C_pemic)
eig_uensa = np.linalg.eigvals(C_uensa)

# Take the largest eigenvalue from each
lambda_pemic = max(eig_pemic)
lambda_uensa = max(eig_uensa)

# Compute ratio of R0 values
R0_ratio = lambda_pemic / lambda_uensa

print("Pemic largest eigenvalue:", lambda_pemic)
print("Uenza largest eigenvalue:", lambda_uensa)
print("R0 ratio (Pemic/Uenza):", R0_ratio)
