import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PARAMETERS AND SAMPLES
# -----------------------------
# Labels must match your MCMC parameters
param_names = ["H0", "log10(Lambda)", "Omega_m"]
# Use your flattened, burn-in removed chain
# Example: flat_samples from previous diagnostics
# flat_samples shape: (n_samples, n_params)
# flat_samples, R_hat = mcmc_diagnostics_with_rhat(sampler)
# For demonstration, assume flat_samples already exists

# -----------------------------
# 1. Compute Covariance & Fisher Matrix
# -----------------------------
C = np.cov(flat_samples.T)          # Covariance matrix
F = np.linalg.inv(C)                # Fisher matrix

print("\nCovariance Matrix:\n", C)
print("\nFisher Matrix (Inverse Covariance):\n", F)
print(f"\nCondition Number of Covariance: {np.linalg.cond(C):.3e}")

# Positive definiteness check
eigvals_C = np.linalg.eigvalsh(C)
if np.all(eigvals_C > 0):
    print("Covariance matrix is positive definite ✅")
else:
    print("WARNING: Covariance matrix is NOT positive definite ❌")

# -----------------------------
# 2. Scaling for numerical stability
# -----------------------------
# Scale parameters to dimensionless units
param_means = np.mean(flat_samples, axis=0)
scale_matrix = np.diag(1.0 / np.abs(param_means))
C_scaled = scale_matrix @ C @ scale_matrix
F_scaled = np.linalg.inv(C_scaled)
print(f"Condition number after scaling: {np.linalg.cond(C_scaled):.3e}")

# -----------------------------
# 3. PCA Diagnostics
# -----------------------------
eigvals, eigvecs = np.linalg.eigh(F_scaled)
# Sort eigenvalues descending
idx = np.argsort(eigvals)[::-1]
eigvals_desc = eigvals[idx]
eigvecs_desc = eigvecs[:, idx]

print("\nPrincipal Component Analysis (PCA) of Fisher Matrix:")
for i, val in enumerate(eigvals_desc):
    print(f"Mode {i+1}: Eigenvalue = {val:.3e}")
    mode_eq = " + ".join([f"{eigvecs_desc[j,i]:+.4f}*{param_names[j]}"
                          for j in range(len(param_names))])
    print(f"        Combination: {mode_eq}")

# Degeneracy check
threshold = 1e-6 * np.max(eigvals_desc)
good_modes = np.sum(eigvals_desc > threshold)
print(f"\nNumber of well-constrained modes: {good_modes}/{len(param_names)}")

# -----------------------------
# 4. Optional: Eigenvalue Plot
# -----------------------------
plt.figure(figsize=(6,4))
plt.bar(range(1, len(eigvals_desc)+1), eigvals_desc, color='skyblue')
plt.yscale('log')
plt.xlabel("Principal Mode")
plt.ylabel("Eigenvalue (log scale)")
plt.title("Fisher Matrix PCA Eigenvalues")
plt.show()