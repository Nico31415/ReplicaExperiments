"""
Improved replica-only code for the LASSO curve in Fig. 3.

- Uses a dense beta grid
- Uses a dense lambda grid
- Uses many Monte-Carlo samples to approximate the scalar expectations
- Produces a smooth replica prediction curve (no simulations)

Copy–paste into a file, e.g. lasso_replica_fig3.py, and run.
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 0. Global configuration
# ============================================================

rng = np.random.default_rng(12345)

# Prior: Bernoulli–Gaussian with 90% zeros (rho = 0.1)
rho = 0.1
var_nonzero = 1.0 / rho        # so that Var(x) = 1
sigma_x2 = rho * var_nonzero
assert np.isclose(sigma_x2, 1.0)  # sanity check

# Noise level: SNR0 = 10 dB, interpreted as signal_var / noise_var
SNR0_dB = 10.0
SNR0_lin = 10 ** (SNR0_dB / 10.0)
sigma0_2 = sigma_x2 / SNR0_lin     # noise variance
sigma0 = np.sqrt(sigma0_2)

# Measurement ratios beta = n/m
betas = np.linspace(0.5, 3.0, 80)  # dense grid for smooth curve

# Replica / state evolution parameters
mc_se_samples = 200_000            # scalar samples per iteration (increase for smoother)
max_fp_iters = 100                 # max fixed-point iterations
tol_fp = 1e-6                      # convergence tolerance on sigma_eff^2

# Lambda grid for LASSO regularization search
# (wider and denser than before for smoothness)
lambda_grid = np.logspace(-3, 1.0, 200)


# ============================================================
# 1. Helper functions
# ============================================================

def bernoulli_gaussian(size, rho, var_nonzero, rng):
    """
    Sample Bernoulli-Gaussian: zero with prob 1-rho, N(0,var_nonzero) with prob rho.
    """
    mask = rng.random(size) < rho
    x = np.zeros(size, dtype=float)
    x[mask] = rng.normal(loc=0.0, scale=np.sqrt(var_nonzero), size=mask.sum())
    return x


def soft_threshold(z, theta):
    """
    Soft-thresholding operator: sign(z) * max(|z| - theta, 0)
    """
    return np.sign(z) * np.maximum(np.abs(z) - theta, 0.0)


# ============================================================
# 2. Replica / state evolution for LASSO
# ============================================================

def lasso_replica_mse(beta, lam, rng):
    """
    Replica-based / AMP-style state evolution for LASSO.

    We consider the scalar equivalent channel:
        z = x + sqrt(sigma_eff^2) * v

    and LASSO's scalar denoiser:
        x_hat = soft_threshold(z, theta),
        theta = lam * sigma_eff^2

    The fixed-point equation is:
        sigma_eff^2 = sigma0^2 + beta * E[(x - x_hat)^2].

    Parameters
    ----------
    beta : float
        Measurement ratio n/m.
    lam : float
        LASSO regularization parameter λ.
    rng : np.random.Generator
        RNG for Monte-Carlo expectation.

    Returns
    -------
    mse : float
        Replica-predicted MSE at the fixed point.
    """
    sigma_eff2 = sigma0_2  # initialize with noise variance

    for _ in range(max_fp_iters):
        # Sample prior and noise
        x = bernoulli_gaussian(mc_se_samples, rho, var_nonzero, rng)
        v = rng.normal(size=mc_se_samples)

        # Scalar channel
        z = x + np.sqrt(sigma_eff2) * v

        # LASSO soft-threshold with θ = λ * sigma_eff^2
        theta = lam * sigma_eff2
        x_hat = soft_threshold(z, theta)

        # MSE under current sigma_eff2
        mse = np.mean((x_hat - x) ** 2)

        # Fixed-point update
        sigma_eff2_new = sigma0_2 + beta * mse

        if np.abs(sigma_eff2_new - sigma_eff2) < tol_fp:
            sigma_eff2 = sigma_eff2_new
            break

        sigma_eff2 = sigma_eff2_new

    return mse


def lasso_replica_curve(betas, lambda_grid, rng):
    """
    For each beta in `betas`, search over `lambda_grid` to find the λ
    that minimizes the replica-predicted MSE.

    Returns
    -------
    mse_replica : np.ndarray
        Best MSE for each beta.
    best_lambda : np.ndarray
        Corresponding λ for each beta.
    """
    mse_replica = np.zeros_like(betas, dtype=float)
    best_lambda = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        best_mse = np.inf
        best_lam = None

        for lam in lambda_grid:
            mse = lasso_replica_mse(beta, lam, rng)
            if mse < best_mse:
                best_mse = mse
                best_lam = lam

        mse_replica[i] = best_mse
        best_lambda[i] = best_lam
        print(f"[Replica] beta = {beta:.3f}, best lambda = {best_lam:.4g}, MSE = {best_mse:.6f}")

    return mse_replica, best_lambda


# ============================================================
# 3. Main: compute replica curve and plot
# ============================================================

if __name__ == "__main__":
    # 3.1 Replica predictions for all betas
    mse_replica, best_lambda = lasso_replica_curve(betas, lambda_grid, rng)

    # Convert to dB (signal variance is 1, so it's just 10 log10(MSE))
    mse_replica_db = 10.0 * np.log10(mse_replica)

    # 3.2 Plot
    plt.figure(figsize=(7, 5))

    plt.plot(
        betas,
        mse_replica_db,
        color="green",
        linestyle="-",
        linewidth=2.0,
        label="Lasso (replica)"
    )

    # Match axes style of the paper
    plt.xlabel(r"Measurement ratio $\beta = n/m$", fontsize=12)
    plt.ylabel("Median squared error (dB)", fontsize=12)  # paper calls it 'median', here it's mean; scale same

    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.xlim([betas.min(), betas.max()])
    plt.ylim([-18, 0])      # adjust if your curve lies outside this range

    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()
