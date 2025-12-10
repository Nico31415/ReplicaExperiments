import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Figure 3 (LASSO only): replica prediction + Monte-Carlo sim
#  *** FIXED VERSION: parameterized by the threshold theta ***
# ============================================================

# -----------------------------
# 1. Basic configuration
# -----------------------------
rng = np.random.default_rng(12345)

# Problem / prior parameters
rho = 0.1                      # fraction of nonzeros
var_nonzero = 1.0 / rho        # so that Var(x) = 1
sigma_x2 = rho * var_nonzero   # should be 1.0
assert np.isclose(sigma_x2, 1.0)

SNR0_dB = 10.0
SNR0_lin = 10 ** (SNR0_dB / 10.0)

# We'll interpret SNR0 as signal variance / noise variance = 1 / sigma0^2
sigma0_2 = 1.0 / SNR0_lin      # noise variance
sigma0 = np.sqrt(sigma0_2)

# Measurement ratios to test (β = n/m)
betas = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])

# Simulation parameters
n = 500                        # signal dimension
num_trials = 200               # Monte-Carlo trials per beta
max_cd_iters = 200             # coordinate descent iterations for LASSO

# Replica (state evolution) parameters
mc_se_samples = 20000          # scalar samples per iteration for expectations
max_fp_iters = 50              # max fixed-point iterations
tol_fp = 1e-5                  # tolerance for fixed-point convergence

# Grid of soft-thresholds θ to search over
theta_grid = np.logspace(-2, 1.0, 40)   # wide, dense range


# ------------------------------------------------
# 2. Helper functions: prior, soft-threshold, etc.
# ------------------------------------------------
def bernoulli_gaussian(size, rho, var_nonzero, rng):
    """Sample Bernoulli-Gaussian vector: zero w.p. (1-rho), N(0,var_nonzero) w.p. rho."""
    mask = rng.random(size) < rho
    x = np.zeros(size, dtype=float)
    x[mask] = rng.normal(loc=0.0, scale=np.sqrt(var_nonzero), size=mask.sum())
    return x


def soft_threshold(z, theta):
    """Soft-thresholding operator."""
    return np.sign(z) * np.maximum(np.abs(z) - theta, 0.0)


# ------------------------------------------------
# 3. Replica (state-evolution) prediction for LASSO
# ------------------------------------------------
def lasso_replica_mse(beta, theta, rng):
    """
    Compute asymptotic MSE for LASSO at given beta and threshold theta
    using scalar state evolution (RS decoupling / AMP-style).

    Scalar channel: z = x + sqrt(sigma_eff^2) v.
    Denoiser:       x_hat = soft_threshold(z, theta).

    Fixed point: sigma_eff^2 = sigma0^2 + beta * E[(x - x_hat)^2].
    """
    sigma_eff2 = sigma0_2  # initialize effective noise variance

    for _ in range(max_fp_iters):
        # Sample from prior and noise
        x = bernoulli_gaussian(mc_se_samples, rho, var_nonzero, rng)
        v = rng.normal(size=mc_se_samples)

        # Scalar AWGN channel
        z = x + np.sqrt(sigma_eff2) * v

        # Scalar estimator (LASSO denoiser)
        xhat = soft_threshold(z, theta)

        # MSE under current sigma_eff2
        mse = np.mean((xhat - x) ** 2)

        # Fixed-point update
        sigma_eff2_new = sigma0_2 + beta * mse

        if np.abs(sigma_eff2_new - sigma_eff2) < tol_fp:
            sigma_eff2 = sigma_eff2_new
            break

        sigma_eff2 = sigma_eff2_new

    # Final MSE for this (beta, theta)
    return mse


def lasso_replica_curve(betas, theta_grid, rng):
    """
    For each beta, find the θ that minimizes the replica-predicted MSE.
    Return:
      mse_replica[beta_index]  – best predicted MSE
      best_theta[beta_index]  – θ achieving it
    """
    mse_replica = np.zeros_like(betas, dtype=float)
    best_theta = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        best_mse = np.inf
        best_th = None

        for theta in theta_grid:
            mse = lasso_replica_mse(beta, theta, rng)
            if mse < best_mse:
                best_mse = mse
                best_th = theta

        mse_replica[i] = best_mse
        best_theta[i] = best_th
        print(f"[Replica] beta={beta:.2f}, best theta={best_th:.4g}, MSE={best_mse:.4g}")

    return mse_replica, best_theta


# -----------------------------------------------------------------
# 4. Coordinate Descent solver for finite-dimensional LASSO
#     min  0.5 * ||y - A x||^2 + theta * ||x||_1
# -----------------------------------------------------------------
def lasso_coordinate_descent(A, y, theta, max_iters=100):
    """
    Simple cyclic coordinate descent for LASSO.

    Objective: 0.5 * ||y - A x||^2 + theta * ||x||_1

    A: (m,n) design matrix
    y: (m,) observation
    theta: scalar L1 penalty (same θ as in replica)
    """
    m, n = A.shape
    x = np.zeros(n)
    r = y.copy()                    # residual r = y - A x (initially x=0)
    col_norms2 = np.sum(A ** 2, axis=0)  # ||a_j||^2

    for _ in range(max_iters):
        for j in range(n):
            aj = A[:, j]
            cj = col_norms2[j]
            if cj == 0.0:
                continue

            # Add back old contribution of x[j] to residual
            r += aj * x[j]

            # Least-squares "coordinate estimate"
            zj = aj.dot(r) / cj

            # Soft-threshold with effective threshold theta / cj
            x_new = soft_threshold(zj, theta / cj)

            # Update residual and coefficient
            r -= aj * x_new
            x[j] = x_new

    return x


# --------------------------------------------------------
# 5. Monte-Carlo simulations for LASSO (finite dimension)
# --------------------------------------------------------
def lasso_simulation_curve(betas, best_theta, rng):
    """
    For each beta, run Monte-Carlo simulations of LASSO with the θ
    predicted (best_theta[beta_index]) and return the median MSE.
    """
    mse_median = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        theta = best_theta[i]
        m = int(round(n / beta))

        mse_trials = []

        for _ in range(num_trials):
            # Design matrix with entries ~ N(0, 1/m)
            A = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, n))

            # True sparse vector
            x_true = bernoulli_gaussian(n, rho, var_nonzero, rng)

            # Noise and measurements
            w = rng.normal(loc=0.0, scale=sigma0, size=m)
            y = A.dot(x_true) + w

            # LASSO estimate with same θ as in replica
            x_hat = lasso_coordinate_descent(A, y, theta, max_iters=max_cd_iters)

            # Normalized squared error
            mse_trials.append(np.mean((x_hat - x_true) ** 2))

        mse_trials = np.array(mse_trials)
        mse_median[i] = np.median(mse_trials)
        print(f"[Sim] beta={beta:.2f}, theta={theta:.4g}, median MSE={mse_median[i]:.4g}")

    return mse_median


# -----------------------------
# 6. Run everything and plot
# -----------------------------
if __name__ == "__main__":
    # 6.1 Replica predictions
    mse_replica, best_theta = lasso_replica_curve(betas, theta_grid, rng)

    # 6.2 Monte-Carlo simulations using those θ values
    mse_sim = lasso_simulation_curve(betas, best_theta, rng)

    # Convert to dB (normalized by Var(x) = 1)
    mse_replica_db = 10.0 * np.log10(mse_replica)
    mse_sim_db = 10.0 * np.log10(mse_sim)

    # 6.3 Plot
    plt.figure(figsize=(7, 5))

    # Lasso replica curve
    plt.plot(
        betas,
        mse_replica_db,
        color="green",
        linestyle="-",
        linewidth=2,
        label="Lasso (replica)",
    )

    # Lasso simulation markers
    plt.plot(
        betas,
        mse_sim_db,
        color="green",
        linestyle="None",
        marker="^",
        markersize=7,
        markerfacecolor="none",
        label="Lasso (sim.)",
    )

    # Axes & styling to match the paper
    plt.xlabel(r"Measurement ratio $\beta = n/m$", fontsize=12)
    plt.ylabel("Median squared error (dB)", fontsize=12)

    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.xlim([betas.min(), betas.max()])
    plt.ylim([-18, 0])  # adjust if needed to match your data

    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.show()
