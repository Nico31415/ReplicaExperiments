import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
#  Reproduce Figure 3 (LASSO only) from Guo–Baron–Shamai (arXiv:0906.3234v3)
# ===============================================================

# -----------------------------
# 1. Experiment parameters
# -----------------------------
rng = np.random.default_rng(42)

n = 100                   # signal dimension
num_trials = 2000          # Monte-Carlo trials per beta
betas = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])   # measurement ratios
theta_grid = np.logspace(-2, 1.0, 15)                   # LASSO thresholds to sweep
max_cd_iters = 150
max_fp_iters = 50
tol_fp = 1e-6

# Bernoulli–Gaussian prior
rho = 0.1
var_nonzero = 1.0
Ex2 = rho * var_nonzero

# SNR definition from paper: 10 dB in terms of signal variance
SNR0_dB = 10.0
SNR0_lin = 10 ** (SNR0_dB / 10.0)
sigma0_2 = Ex2 / SNR0_lin         # = 0.01
sigma0 = np.sqrt(sigma0_2)

# -----------------------------
# 2. Utility functions
# -----------------------------
def sample_prior(size, rng):
    mask = rng.random(size) < rho
    x = np.zeros(size)
    x[mask] = rng.normal(0, np.sqrt(var_nonzero), size=mask.sum())
    return x

def soft_threshold(z, theta):
    return np.sign(z) * np.maximum(np.abs(z) - theta, 0.0)

# -----------------------------
# 3. Replica / State-evolution prediction
# -----------------------------
def lasso_replica_mse(beta, theta, rng):
    """Replica (state-evolution) MSE for given beta, theta."""
    sigma_eff2 = sigma0_2
    x = sample_prior(5000, rng)
    v = rng.normal(size=x.size)

    for _ in range(max_fp_iters):
        z = x + np.sqrt(sigma_eff2) * v
        xhat = soft_threshold(z, theta)
        mse = np.mean((xhat - x) ** 2)
        sigma_eff2_new = sigma0_2 + beta * mse
        if abs(sigma_eff2_new - sigma_eff2) < tol_fp:
            break
        sigma_eff2 = sigma_eff2_new

    return mse / Ex2    # normalize by signal power

def best_lasso_replica_curve(betas, theta_grid, rng):
    mse_rep, best_theta = [], []
    for beta in betas:
        best_mse = np.inf
        best_t = None
        for theta in theta_grid:
            mse = lasso_replica_mse(beta, theta, rng)
            if mse < best_mse:
                best_mse = mse
                best_t = theta
        mse_rep.append(best_mse)
        best_theta.append(best_t)
        print(f"[Replica] beta={beta:.2f}, best θ={best_t:.4g}, MSE={best_mse:.4g}")
    return np.array(mse_rep), np.array(best_theta)

# -----------------------------
# 4. Empirical Monte-Carlo simulation
# -----------------------------
def lasso_coordinate_descent(A, y, theta, max_iters=150):
    """Simple cyclic coordinate-descent LASSO solver."""
    m, n = A.shape
    x = np.zeros(n)
    r = y.copy()
    col_norms2 = np.sum(A ** 2, axis=0)

    for _ in range(max_iters):
        for j in range(n):
            aj = A[:, j]
            cj = col_norms2[j]
            r += aj * x[j]
            zj = aj.dot(r) / cj
            x_new = soft_threshold(zj, theta / cj)
            r -= aj * x_new
            x[j] = x_new
    return x

def lasso_empirical_curve(betas, best_theta, rng):
    mse_median = []
    for i, beta in enumerate(betas):
        m = int(round(n / beta))
        theta = best_theta[i]
        mse_trials = []
        for _ in range(num_trials):
            x_true = sample_prior(n, rng)
            A = rng.normal(0, 1 / np.sqrt(m), size=(m, n))
            w = rng.normal(0, sigma0, size=m)
            y = A @ x_true + w
            x_hat = lasso_coordinate_descent(A, y, theta, max_iters=max_cd_iters)
            mse_norm = np.mean((x_hat - x_true) ** 2) / np.mean(x_true ** 2)
            mse_trials.append(mse_norm)
        med = np.median(mse_trials)
        mse_median.append(med)
        print(f"[Sim] beta={beta:.2f}, θ={theta:.4g}, median normalized MSE={med:.4g}")
    return np.array(mse_median)

# -----------------------------
# 5. Run everything
# -----------------------------
mse_replica, best_theta = best_lasso_replica_curve(betas, theta_grid, rng)
mse_empirical = lasso_empirical_curve(betas, best_theta, rng)

# Convert to dB
mse_replica_db = 10 * np.log10(mse_replica)
mse_empirical_db = 10 * np.log10(mse_empirical)

# -----------------------------
# 6. Plot
# -----------------------------
plt.figure(figsize=(7.2, 4.6))
plt.plot(betas, mse_replica_db, "g-", linewidth=2, label="Lasso (replica)")
plt.plot(betas, mse_empirical_db, "g^", markersize=7, markerfacecolor="none", label="Lasso (sim.)")
plt.xlabel(r"Measurement ratio $\beta = n/m$")
plt.ylabel("Median normalized squared error (dB)")
plt.grid(True, linestyle=":", linewidth=0.7)
plt.xlim([betas.min(), betas.max()])
plt.ylim([-18, 0])
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
