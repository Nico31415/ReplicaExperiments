import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  LASSO: Replica / State-Evolution vs Monte Carlo
#
#  y = A x + w,  A_ij ~ N(0, 1/m),  w ~ N(0, sigma0^2 I)
#  PRIOR_MODE ∈ { "bern-gauss", "bin-gauss" }
#
#  Replica side:
#      scalar channel z = x + sqrt(sigma_eff^2) v
#      denoiser      η(z; θ) = soft_threshold(z, θ)
#      fixed point   sigma_eff^2 = sigma0^2 + β * E[(η(z; θ) - x)^2]
#      λ (LASSO penalty) related to θ via
#          λ = θ * (1 - (1/δ) E[η'(z; θ)])   with δ = m/n, 1/δ = β
#          η'(z; θ) = 1_{|z| > θ}  for soft-thresholding
#
#  Simulation side:
#      Solve  x̂ = argmin_x 0.5 ||y - A x||_2^2 + λ ||x||_1
#      via coordinate descent, with λ taken from the replica mapping.
# ============================================================

# -----------------------------
# 0. Choose prior
# -----------------------------
PRIOR_MODE = "bin-gauss"   # "bern-gauss" or "bin-gauss"

# -----------------------------
# 1. Basic configuration
# -----------------------------
rng = np.random.default_rng(12345)

# ---- Parameters for Bernoulli–Gaussian prior ----
rho = 0.1                      # fraction of nonzeros in bern-gauss prior
var_nonzero_bg = 1.0 / rho     # so that Var(x) ≈ 1

# ---- Parameters for Bin vs small-Gauss mixture prior ----
rho_spiky    = 0.2
p_one        = 0.02
sigma2_small = 0.3

# SNR definition (noise variance fixed)
SNR0_dB  = 10.0
SNR0_lin = 10 ** (SNR0_dB / 10.0)
sigma0_2 = 1.0 / SNR0_lin
sigma0   = np.sqrt(sigma0_2)

# Measurement ratios to test (β = n/m)
betas = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])

# Problem size and Monte Carlo
n           = 1000    # signal dimension
num_trials  = 100     # Monte Carlo trials per beta
max_cd_iters = 150    # coordinate descent iterations

# Replica / state evolution parameters
mc_se_samples = 8000   # scalar samples per iteration for expectations
max_fp_iters  = 50     # max fixed-point iterations
tol_fp        = 1e-5   # tolerance for fixed-point convergence

# Grid of soft-threshold levels (θ) to explore for replica
theta_grid_lasso = np.logspace(-2, 1.0, 16)


# ------------------------------------------------
# 2. Helper functions: prior + soft-threshold
# ------------------------------------------------
def sample_prior(size, rng):
    """
    Sample x and 'group_mask' for the current PRIOR_MODE.

    Returns:
      x:          (size,) vector of prior draws
      group_mask: (size,) boolean

    PRIOR_MODE == "bern-gauss":
        x_j = 0 w.p. 1-rho,
        x_j ~ N(0, var_nonzero_bg) w.p. rho
        group_mask[j] = True for Gaussian (nonzero) coords.

    PRIOR_MODE == "bin-gauss":
        With prob rho_spiky:
            x_j ~ Bin(p_one)  (0 or 1)
            group_mask[j] = True  (spiky)
        With prob 1-rho_spiky:
            x_j ~ N(0, sigma2_small)
            group_mask[j] = False (smooth)
    """
    if PRIOR_MODE == "bern-gauss":
        group_mask = rng.random(size) < rho
        x = np.zeros(size, dtype=float)
        x[group_mask] = rng.normal(
            loc=0.0, scale=np.sqrt(var_nonzero_bg), size=group_mask.sum()
        )
        return x, group_mask

    elif PRIOR_MODE == "bin-gauss":
        group_mask = rng.random(size) < rho_spiky   # True -> Bin component
        x = np.zeros(size, dtype=float)

        # Spiky Bin(p_one) component
        spiky_idx = group_mask
        if spiky_idx.any():
            ones_mask = rng.random(size) < p_one
            x[spiky_idx & ones_mask] = 1.0

        # Small Gaussian component
        smooth_idx = ~group_mask
        if smooth_idx.any():
            x[smooth_idx] = rng.normal(
                loc=0.0, scale=np.sqrt(sigma2_small), size=smooth_idx.sum()
            )

        return x, group_mask

    else:
        raise ValueError(f"Unknown PRIOR_MODE: {PRIOR_MODE}")


def soft_threshold(z, theta):
    """Soft-thresholding operator."""
    return np.sign(z) * np.maximum(np.abs(z) - theta, 0.0)


# ------------------------------------------------
# 3. Replica / SE for LASSO with λ-θ mapping
# ------------------------------------------------
def lasso_se_fixed_point(beta, theta, rng):
    """
    State evolution fixed point for LASSO with soft-threshold level θ.

    We consider the scalar channel:
        z = x + sqrt(sigma_eff^2) v,
    and denoiser
        x_hat = soft_threshold(z, theta).

    Returns:
        mse:         E[(x_hat - x)^2] at the fixed point
        sigma_eff2:  fixed-point effective noise variance
        p_active:    E[η'(z; θ)] = P(|z| > θ) at the fixed point
    """
    sigma_eff2 = sigma0_2

    # Pre-sample x and v once (Monte Carlo approximation of expectations)
    x, _ = sample_prior(mc_se_samples, rng)
    v = rng.normal(size=mc_se_samples)

    for _ in range(max_fp_iters):
        z = x + np.sqrt(sigma_eff2) * v
        xhat = soft_threshold(z, theta)

        mse = np.mean((xhat - x) ** 2)
        p_active = np.mean(np.abs(z) > theta)  # E[η'(z; θ)]

        sigma_eff2_new = sigma0_2 + beta * mse  # β = n/m = 1/δ

        if np.abs(sigma_eff2_new - sigma_eff2) < tol_fp:
            sigma_eff2 = sigma_eff2_new
            break

        sigma_eff2 = sigma_eff2_new

    # One last update for mse / p_active at the converged sigma_eff2
    z = x + np.sqrt(sigma_eff2) * v
    xhat = soft_threshold(z, theta)
    mse = np.mean((xhat - x) ** 2)
    p_active = np.mean(np.abs(z) > theta)

    return mse, sigma_eff2, p_active


def lasso_replica_curve(betas, theta_grid, rng):
    """
    For each β, scan θ-grid, solve SE, and:
      - pick θ that minimizes replica-predicted MSE,
      - compute the corresponding LASSO λ from θ via the AMP / KKT mapping:
            λ = θ * (1 - (1/δ) E[η'(z; θ)]),
        where δ = m/n, so 1/δ = β.
    """
    mse_replica = np.zeros_like(betas, dtype=float)
    best_theta = np.zeros_like(betas, dtype=float)
    best_lambda = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        best_mse = np.inf
        best_th = None
        best_lam = None

        for theta in theta_grid:
            mse, sigma_eff2, p_active = lasso_se_fixed_point(beta, theta, rng)

            # δ = m/n = 1/β, so 1/δ = β
            lam = theta * (1.0 - beta * p_active)
            # Numerical safety: λ must be ≥ 0
            lam = max(lam, 0.0)

            if mse < best_mse:
                best_mse = mse
                best_th = theta
                best_lam = lam

        mse_replica[i] = best_mse
        best_theta[i] = best_th
        best_lambda[i] = best_lam

        print(f"[Replica-Lasso] beta={beta:.2f}, "
              f"best θ={best_th:.4g}, λ={best_lam:.4g}, MSE={best_mse:.4g}")

    return mse_replica, best_theta, best_lambda


# --------------------------------------------------------
# 4. Coordinate Descent solver for finite-dimensional LASSO
# --------------------------------------------------------
def lasso_coordinate_descent(A, y, lam, max_iters=100):
    """
    Cyclic coordinate descent for the LASSO problem:

        min_x  0.5 * ||y - A x||^2 + lam * ||x||_1

    where 'lam' is the L1 regularization parameter.
    """
    m, n = A.shape
    x = np.zeros(n)
    r = y.copy()
    col_norms2 = np.sum(A ** 2, axis=0)

    for _ in range(max_iters):
        for j in range(n):
            aj = A[:, j]
            cj = col_norms2[j]
            if cj == 0.0:
                continue

            # Add back old contribution of x[j] to the residual
            r += aj * x[j]

            # Least-squares estimate for coordinate j
            zj = aj.dot(r) / cj

            # Soft-threshold with parameter lam / cj
            x_new = soft_threshold(zj, lam / cj)

            # Update residual and coordinate value
            r -= aj * x_new
            x[j] = x_new

    return x


# --------------------------------------------------------
# 5. Monte-Carlo simulations for LASSO
# --------------------------------------------------------
def lasso_simulation_curve(betas, best_lambda, rng):
    """
    For each β, run Monte Carlo simulations of the finite-dimensional LASSO
    tuned with λ from the replica λ-θ mapping.

    Returns:
      mse_median: median MSE over trials for each β.
    """
    mse_median = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        lam = best_lambda[i]
        m = int(round(n / beta))
        mse_trials = []

        for _ in range(num_trials):
            # Design matrix A with entries ~ N(0, 1/sqrt(m))
            A = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, n))

            # Signal and noise
            x_true, _ = sample_prior(n, rng)
            w = rng.normal(loc=0.0, scale=sigma0, size=m)

            # Observations
            y = A @ x_true + w

            # LASSO estimate
            x_hat = lasso_coordinate_descent(A, y, lam, max_iters=max_cd_iters)

            mse_trials.append(np.mean((x_hat - x_true) ** 2))

        mse_median[i] = np.median(mse_trials)
        print(f"[Sim-Lasso] beta={beta:.2f}, λ={lam:.4g}, median MSE={mse_median[i]:.4g}")

    return mse_median


# -----------------------------
# 6. Run LASSO replica + empirical and plot
# -----------------------------
if __name__ == "__main__":
    print(f"Using PRIOR_MODE = {PRIOR_MODE}")
    print("Computing replica predictions for LASSO...")

    # Replica predictions + λ mapping
    mse_lasso_replica, best_theta_lasso, best_lambda_lasso = lasso_replica_curve(
        betas, theta_grid_lasso, rng
    )

    print("\nRunning Monte-Carlo simulations for LASSO...")
    mse_lasso_sim = lasso_simulation_curve(betas, best_lambda_lasso, rng)

    # Convert MSEs to dB
    mse_lasso_replica_db = 10.0 * np.log10(mse_lasso_replica)
    mse_lasso_sim_db     = 10.0 * np.log10(mse_lasso_sim)

    # Plot
    plt.figure(figsize=(7, 4.5))

    plt.plot(betas, mse_lasso_replica_db,
             linestyle="-", linewidth=2, label="LASSO (replica)")
    plt.plot(betas, mse_lasso_sim_db,
             linestyle="None", marker="o", markersize=7,
             markerfacecolor="none", label="LASSO (simulation)")

    plt.xlabel(r"Measurement ratio $\beta = n/m$", fontsize=12)
    plt.ylabel("MSE (dB)", fontsize=12)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.xlim([betas.min(), betas.max()])
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()
