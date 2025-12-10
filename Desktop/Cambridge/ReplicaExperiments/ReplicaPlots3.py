import numpy as np
import matplotlib.pyplot as plt

# ============================================================
#  Fast Figure-3 style experiment with two possible priors:
#
#  PRIOR_MODE = "bern-gauss":
#      x_j ~ N(0, 1) w.p. ρ,  x_j = 0 w.p. (1-ρ)
#
#  PRIOR_MODE = "bin-gauss":
#      With prob rho_spiky:   x_j ~ Bin(p_one)  (0 or 1, "spiky" - LASSO-friendly)
#      With prob 1-rho_spiky: x_j ~ N(0, sigma2_small)    (small Gaussian - ridge-friendly)
#
#  Methods: Linear MMSE, LASSO, Elastic Net, Oracle Hybrid(LASSO/Ridge)
#  Replica predictions + Monte-Carlo simulations
# ============================================================

# -----------------------------
# 0. Choose prior and run mode
# -----------------------------
PRIOR_MODE = "bern-gauss"   # "bern-gauss" or "bin-gauss"
FAST_RUN = True             # If True, only run LASSO with rough parameters for quick curve check

# -----------------------------
# 1. Basic configuration
# -----------------------------
rng = np.random.default_rng(12345)

# ---- Parameters for Bernoulli–Gaussian prior (only used if PRIOR_MODE == "bern-gauss") ----
if PRIOR_MODE == "bern-gauss":
    rho = 0.1                      # fraction of nonzeros in bern-gauss prior
    var_nonzero_bg = 1.0           # variance of Gaussian component: x_j ~ N(0, 1) with prob ρ

# ---- Parameters for Bin vs small-Gauss mixture prior ----
rho_spiky = 0.2                    # prob that coordinate uses Bin(p_one)
p_one = 0.02                       # within spiky component: P(x=1); else 0
sigma2_small = 0.3                 # variance of small Gaussian component

# SNR definition
SNR0_dB = 10.0
sigma0_2 = 1.0 / (10 ** (SNR0_dB / 10.0))  # noise variance
sigma0 = np.sqrt(sigma0_2)

# Measurement ratios to test (β = n/m)
betas = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])

# -----------------------------
# 1a. "Fast mode" parameters
# -----------------------------
if FAST_RUN:
    # Moderate parameters for quick curve shape check (LASSO only)
    # Still faster than full run but with better accuracy
    n = 800                      # signal dimension (moderate, close to full)
    num_trials = 75              # Monte-Carlo trials per beta (moderate, 3/4 of full)
    max_cd_iters = 120           # coordinate descent iterations (moderate)
    
    # Replica (state evolution) parameters
    mc_se_samples = 5000         # scalar samples per iteration (moderate, 5/8 of full)
    max_fp_iters = 45            # max fixed-point iterations (moderate, close to full)
    tol_fp = 1e-5               # tolerance for fixed-point convergence (same as full)
    
    # Grid for thresholds (moderate resolution)
    theta_grid = np.logspace(-2, 1.0, 14)
else:
    # Full precision parameters
    n = 1000                     # signal dimension
    num_trials = 100             # Monte-Carlo trials per beta
    max_cd_iters = 150          # coordinate descent iterations
    
    # Replica (state evolution) parameters
    mc_se_samples = 8000         # scalar samples per iteration for expectations
    max_fp_iters = 50           # max fixed-point iterations
    tol_fp = 1e-5               # tolerance for fixed-point convergence
    
    # Grid for thresholds (used for all methods that need hyperparameter search)
    theta_grid = np.logspace(-2, 1.0, 16)

# Elastic Net mixing parameter:
# penalty = theta * (alpha * ||x||_1 + (1-alpha)/2 * ||x||_2^2)
alpha_enet = 0.5


# ------------------------------------------------
# 2. Helper functions: priors, denoisers, etc.
# ------------------------------------------------
def sample_prior(size, rng):
    """
    Sample x and a 'group mask' for the current PRIOR_MODE.

    Returns:
      x:          (size,) vector
      group_mask: (size,) boolean
          PRIOR_MODE == "bern-gauss":
              x_j ~ N(0, 1) w.p. ρ,
              x_j = 0 w.p. (1-ρ)
              group_mask[j] = True for Gaussian (nonzero) coords (for L1 group).

          PRIOR_MODE == "bin-gauss":
              With prob rho_spiky:
                  x_j ~ Bin(p_one)  (0 or 1)
                  group_mask[j] = True  (spiky / L1 group)
              With prob 1-rho_spiky:
                  x_j ~ N(0, sigma2_small)
                  group_mask[j] = False (smooth / L2 group)
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
        if group_mask.any():
            ones_mask = rng.random(size) < p_one
            x[group_mask & ones_mask] = 1.0

        # Small Gaussian component
        smooth_mask = ~group_mask
        if smooth_mask.any():
            x[smooth_mask] = rng.normal(
                loc=0.0, scale=np.sqrt(sigma2_small), size=smooth_mask.sum()
            )

        return x, group_mask

    else:
        raise ValueError(f"Unknown PRIOR_MODE: {PRIOR_MODE}")


def soft_threshold(z, theta):
    """Soft-thresholding operator."""
    return np.sign(z) * np.maximum(np.abs(z) - theta, 0.0)


def elastic_net_denoiser(z, theta, alpha):
    """
    Proximal operator for Elastic Net penalty

        f(x) = theta * (alpha * |x| + (1-alpha)/2 * x^2)

    prox_f(z) = argmin_x 0.5 |x - z|^2 + f(x)
              = (1 / (1 + lambda2)) * soft_threshold(z, lambda1),

    where lambda1 = theta * alpha, lambda2 = theta * (1-alpha).
    """
    lam1 = theta * alpha
    lam2 = theta * (1.0 - alpha)
    return (1.0 / (1.0 + lam2)) * soft_threshold(z, lam1)


# ------------------------------------------------
# 3. Replica prediction: Common infrastructure
# ------------------------------------------------
def _replica_state_evolution(beta, denoiser_fn, rng, need_group_mask=False):
    """
    Common state evolution loop for replica predictions.

    Args:
        beta: Measurement ratio n/m
        denoiser_fn: Function that takes (z, theta, ...) and returns x_hat
        rng: Random number generator
        need_group_mask: If True, sample_prior returns group_mask (for hybrid oracle)

    Returns:
        Final MSE after convergence
    """
    sigma_eff2 = sigma0_2

    # Pre-sample x and v once
    if need_group_mask:
        x, group_mask = sample_prior(mc_se_samples, rng)
    else:
        x, _ = sample_prior(mc_se_samples, rng)
        group_mask = None

    v = rng.normal(size=mc_se_samples)

    for _ in range(max_fp_iters):
        z = x + np.sqrt(sigma_eff2) * v

        if need_group_mask:
            x_hat = denoiser_fn(z, group_mask)
        else:
            x_hat = denoiser_fn(z)

        mse = np.mean((x_hat - x) ** 2)
        sigma_eff2_new = sigma0_2 + beta * mse

        if np.abs(sigma_eff2_new - sigma_eff2) < tol_fp:
            break
        sigma_eff2 = sigma_eff2_new

    return mse


def _replica_curve(betas, theta_grid, mse_fn, method_name, rng, print_fn=None, **mse_kwargs):
    """
    Common grid search pattern for replica curves.

    Args:
        betas: Array of measurement ratios
        theta_grid: Grid of hyperparameters to search
        mse_fn: Function that computes MSE for given beta, theta, ...
        method_name: String for printing (e.g., "Lasso", "ENet")
        rng: Random number generator
        print_fn: Optional custom print function (beta, best_theta_val, best_mse, **kwargs)
        **mse_kwargs: Additional keyword arguments to pass to mse_fn

    Returns:
        mse_replica: Array of best MSEs for each beta
        best_theta: Array of best thetas for each beta
    """
    mse_replica = np.zeros_like(betas, dtype=float)
    best_theta = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        best_mse = np.inf
        best_theta_val = None

        for theta in theta_grid:
            mse = mse_fn(beta, theta, rng, **mse_kwargs)
            if mse < best_mse:
                best_mse = mse
                best_theta_val = theta

        mse_replica[i] = best_mse
        best_theta[i] = best_theta_val
        
        if print_fn:
            print_fn(beta, best_theta_val, best_mse, **mse_kwargs)
        else:
            print(f"[Replica-{method_name}] beta={beta:.2f}, best theta={best_theta_val:.4g}, MSE={best_mse:.4g}")

    return mse_replica, best_theta


# ------------------------------------------------
# 3a. Replica prediction: LASSO
# ------------------------------------------------
def lasso_replica_mse(beta, theta, rng):
    """
    Asymptotic MSE for LASSO at given beta and threshold theta
    using scalar state evolution.

    Scalar channel: z = x + sqrt(sigma_eff^2) v.
    Denoiser:       x_hat = soft_threshold(z, theta).
    """
    def denoiser(z):
        return soft_threshold(z, theta)

    return _replica_state_evolution(beta, denoiser, rng)


def lasso_replica_curve(betas, theta_grid, rng):
    return _replica_curve(betas, theta_grid, lasso_replica_mse, "Lasso", rng)


# ------------------------------------------------
# 3b. Replica prediction: Linear MMSE
# ------------------------------------------------
def linear_replica_mse(beta):
    """
    Asymptotic MSE for the linear MMSE estimator (quadratic regularizer).

    For S = I and Var(x) ≈ 1, the RS/Tse–Hanly fixed point is:
        sigma_eff^2 = sigma0^2 + beta * mse
        mse = sigma_eff^2 / (1 + sigma_eff^2)
    We keep the same formula here even if Var(x) != 1 exactly.
    """
    sigma_eff2 = sigma0_2

    for _ in range(max_fp_iters):
        mse = sigma_eff2 / (1.0 + sigma_eff2)
        sigma_eff2_new = sigma0_2 + beta * mse

        if abs(sigma_eff2_new - sigma_eff2) < tol_fp:
            break
        sigma_eff2 = sigma_eff2_new

    # Final MSE (in case loop didn't converge, use last computed value)
    mse = sigma_eff2 / (1.0 + sigma_eff2)
    return mse


def linear_replica_curve(betas):
    mse_replica = np.zeros_like(betas, dtype=float)
    for i, beta in enumerate(betas):
        mse = linear_replica_mse(beta)
        mse_replica[i] = mse
        print(f"[Replica-Linear] beta={beta:.2f}, MSE={mse:.4g}")
    return mse_replica


# ------------------------------------------------
# 3c. Replica prediction: Elastic Net
# ------------------------------------------------
def enet_replica_mse(beta, theta, alpha, rng):
    """
    Asymptotic MSE for Elastic Net with given beta, theta, alpha
    using scalar state evolution.
    """
    def denoiser(z):
        return elastic_net_denoiser(z, theta, alpha)

    return _replica_state_evolution(beta, denoiser, rng)


def enet_replica_curve(betas, theta_grid, alpha, rng):
    def mse_fn(beta, theta, rng):
        return enet_replica_mse(beta, theta, alpha, rng)
    
    def print_fn(beta, best_theta_val, best_mse, alpha=alpha):
        print(f"[Replica-ENet] beta={beta:.2f}, alpha={alpha:.2f}, "
              f"best theta={best_theta_val:.4g}, MSE={best_mse:.4g}")

    mse_replica, best_theta = _replica_curve(betas, theta_grid, mse_fn, "ENet", rng,
                                            print_fn=print_fn, alpha=alpha)
    return mse_replica, best_theta


# ------------------------------------------------
# 3d. Replica prediction: Oracle Hybrid(LASSO/Ridge)
# ------------------------------------------------
def hybrid_oracle_replica_mse(beta, theta, rng):
    """
    Asymptotic MSE for the Oracle Hybrid(LASSO/Ridge) estimator
    at given beta, theta.

    Oracle rule (depends on prior component):
      - If coordinate is from 'group_mask=True' (spiky / Bin)  → LASSO
      - If coordinate is from 'group_mask=False' (smooth / Gauss) → Ridge
    """
    def denoiser(z, group_mask):
        x_hat = np.empty_like(z)
        # LASSO on group_mask == True
        x_hat[group_mask] = soft_threshold(z[group_mask], theta)
        # Ridge on group_mask == False (prox of 0.5(x-z)^2 + (theta/2)x^2)
        x_hat[~group_mask] = z[~group_mask] / (1.0 + theta)
        return x_hat

    return _replica_state_evolution(beta, denoiser, rng, need_group_mask=True)


def hybrid_oracle_replica_curve(betas, theta_grid, rng):
    return _replica_curve(betas, theta_grid, hybrid_oracle_replica_mse, "HybridOracle", rng)


# -----------------------------------------------------------------
# 4. Coordinate Descent solvers for finite-dimensional estimators
# -----------------------------------------------------------------
def lasso_coordinate_descent(A, y, theta, max_iters=150):
    """Cyclic coordinate descent for LASSO."""
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

            r += aj * x[j]
            zj = aj.dot(r) / cj
            x_new = soft_threshold(zj, theta / cj)
            r -= aj * x_new
            x[j] = x_new

    return x


def enet_coordinate_descent(A, y, theta, alpha, max_iters=150):
    """Cyclic coordinate descent for Elastic Net."""
    m, n = A.shape
    x = np.zeros(n)
    r = y.copy()
    col_norms2 = np.sum(A ** 2, axis=0)

    lam1 = theta * alpha
    lam2 = theta * (1.0 - alpha)

    for _ in range(max_iters):
        for j in range(n):
            aj = A[:, j]
            cj = col_norms2[j]
            if cj == 0.0:
                continue

            r += aj * x[j]
            denom = cj + lam2
            u = aj.dot(r) / denom
            x_new = soft_threshold(u, lam1 / denom)
            r -= aj * x_new
            x[j] = x_new

    return x


def hybrid_oracle_coordinate_descent(A, y, theta, group_mask, max_iters=150):
    """
    Cyclic coordinate descent for Oracle Hybrid(LASSO/Ridge).

    Objective (oracle, component-dependent):
        0.5 * ||y - A x||^2
        + theta * sum_{j in group_mask} |x_j|
        + (theta/2) * sum_{j not in group_mask} x_j^2
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

            r += aj * x[j]

            if group_mask[j]:
                # LASSO update
                zj = aj.dot(r) / cj
                x_new = soft_threshold(zj, theta / cj)
            else:
                # Ridge update
                denom = cj + theta
                x_new = aj.dot(r) / denom

            r -= aj * x_new
            x[j] = x_new

    return x


def linear_mmse_estimator(A, y, sigma0_2):
    """Linear MMSE estimator."""
    m, n = A.shape
    M = A @ A.T + sigma0_2 * np.eye(m)
    z = np.linalg.solve(M, y)
    x_hat = A.T @ z
    return x_hat


# --------------------------------------------------------
# 5. Monte-Carlo simulations: Common infrastructure
# --------------------------------------------------------
def _generate_simulation_trial(beta, rng, need_group_mask=False):
    """
    Generate a single simulation trial: A, x_true, w, y.

    Args:
        beta: Measurement ratio n/m
        rng: Random number generator
        need_group_mask: If True, return group_mask as well

    Returns:
        A: (m, n) measurement matrix
        x_true: (n,) true signal
        w: (m,) noise vector
        y: (m,) observations
        group_mask: (n,) boolean mask (only if need_group_mask=True)
    """
    m = int(round(n / beta))
    A = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, n))

    if need_group_mask:
        x_true, group_mask = sample_prior(n, rng)
    else:
        x_true, _ = sample_prior(n, rng)
        group_mask = None

    w = rng.normal(loc=0.0, scale=sigma0, size=m)
    y = A @ x_true + w

    if need_group_mask:
        return A, x_true, w, y, group_mask
    else:
        return A, x_true, w, y


def _simulation_curve(betas, trial_fn, method_name, rng, best_theta=None):
    """
    Common simulation curve pattern.

    Args:
        betas: Array of measurement ratios
        trial_fn: Function that takes (beta, theta, rng) and returns MSE for one trial
        method_name: String for printing
        rng: Random number generator
        best_theta: Array of best thetas (None for methods without hyperparameters)

    Returns:
        mse_median: Array of median MSEs for each beta
    """
    mse_median = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        theta = best_theta[i] if best_theta is not None else None
        mse_trials = []

        for _ in range(num_trials):
            mse = trial_fn(beta, theta, rng)
            mse_trials.append(mse)

        mse_median[i] = np.median(mse_trials)
        theta_str = f", theta={theta:.4g}" if theta is not None else ""
        print(f"[Sim-{method_name}] beta={beta:.2f}{theta_str}, median MSE={mse_median[i]:.4g}")

    return mse_median


def lasso_simulation_curve(betas, best_theta, rng):
    def trial_fn(beta, theta, rng):
        A, x_true, w, y = _generate_simulation_trial(beta, rng)
        x_hat = lasso_coordinate_descent(A, y, theta, max_iters=max_cd_iters)
        return np.mean((x_hat - x_true) ** 2)
    
    return _simulation_curve(betas, trial_fn, "Lasso", rng, best_theta=best_theta)


def enet_simulation_curve(betas, best_theta, alpha, rng):
    def trial_fn(beta, theta, rng):
        A, x_true, w, y = _generate_simulation_trial(beta, rng)
        x_hat = enet_coordinate_descent(A, y, theta, alpha, max_iters=max_cd_iters)
        return np.mean((x_hat - x_true) ** 2)
    
    return _simulation_curve(betas, trial_fn, "ENet", rng, best_theta=best_theta)


def hybrid_oracle_simulation_curve(betas, best_theta, rng):
    def trial_fn(beta, theta, rng):
        A, x_true, w, y, group_mask = _generate_simulation_trial(beta, rng, need_group_mask=True)
        x_hat = hybrid_oracle_coordinate_descent(A, y, theta, group_mask, max_iters=max_cd_iters)
        return np.mean((x_hat - x_true) ** 2)
    
    return _simulation_curve(betas, trial_fn, "HybridOracle", rng, best_theta=best_theta)


def linear_simulation_curve(betas, rng):
    def trial_fn(beta, theta, rng):
        A, x_true, w, y = _generate_simulation_trial(beta, rng)
        x_hat = linear_mmse_estimator(A, y, sigma0_2)
        return np.mean((x_hat - x_true) ** 2)
    
    return _simulation_curve(betas, trial_fn, "Linear", rng)


# -----------------------------
# 6. Run everything and plot
# -----------------------------
if __name__ == "__main__":
    print(f"Using PRIOR_MODE = {PRIOR_MODE}")
    if FAST_RUN:
        print("FAST_RUN mode: Only running LASSO with rough parameters for quick curve check")

    if FAST_RUN:
        # ---- Fast run: LASSO only ----
        mse_lasso_replica, best_theta_lasso = lasso_replica_curve(betas, theta_grid, rng)
        mse_lasso_sim = lasso_simulation_curve(betas, best_theta_lasso, rng)

        # Convert to dB
        mse_lasso_replica_db = 10.0 * np.log10(mse_lasso_replica)
        mse_lasso_sim_db = 10.0 * np.log10(mse_lasso_sim)

        # Plot
        plt.figure(figsize=(8, 5))
        plt.plot(betas, mse_lasso_replica_db, color="green", linestyle="-", linewidth=2,
                 label="Lasso (replica)")
        plt.plot(betas, mse_lasso_sim_db, color="green", linestyle="None",
                 marker="^", markersize=7, markerfacecolor="none", label="Lasso (sim.)")

    else:
        # ---- Full run: All methods ----
        # Replica predictions
        mse_lasso_replica, best_theta_lasso = lasso_replica_curve(betas, theta_grid, rng)
        mse_linear_replica = linear_replica_curve(betas)
        mse_enet_replica, best_theta_enet = enet_replica_curve(betas, theta_grid, alpha_enet, rng)
        mse_hybrid_replica, best_theta_hybrid = hybrid_oracle_replica_curve(betas, theta_grid, rng)

        # Monte-Carlo simulations
        mse_lasso_sim = lasso_simulation_curve(betas, best_theta_lasso, rng)
        mse_linear_sim = linear_simulation_curve(betas, rng)
        mse_enet_sim = enet_simulation_curve(betas, best_theta_enet, alpha_enet, rng)
        mse_hybrid_sim = hybrid_oracle_simulation_curve(betas, best_theta_hybrid, rng)

        # Convert all MSEs to dB
        mse_results = {
            'lasso': (mse_lasso_replica, mse_lasso_sim),
            'linear': (mse_linear_replica, mse_linear_sim),
            'enet': (mse_enet_replica, mse_enet_sim),
            'hybrid': (mse_hybrid_replica, mse_hybrid_sim),
        }

        mse_db = {}
        for method, (mse_replica, mse_sim) in mse_results.items():
            mse_db[method] = {
                'replica': 10.0 * np.log10(mse_replica),
                'sim': 10.0 * np.log10(mse_sim)
            }

        # Plot
        plt.figure(figsize=(8, 5))

        plot_configs = [
            ('linear', 'blue', 'o', 'Linear'),
            ('lasso', 'green', '^', 'Lasso'),
            ('enet', 'orange', 's', f'Elastic Net (α={alpha_enet:.2f})'),
            ('hybrid', 'purple', 'd', 'Oracle Hybrid L1/L2'),
        ]

        for method, color, marker, label_base in plot_configs:
            plt.plot(betas, mse_db[method]['replica'], color=color, linestyle="-", linewidth=2,
                     label=f"{label_base} (replica)")
            plt.plot(betas, mse_db[method]['sim'], color=color, linestyle="None",
                     marker=marker, markersize=7, markerfacecolor="none",
                     label=f"{label_base} (sim.)")

    # Common plot settings
    plt.xlabel(r"Measurement ratio $\beta = n/m$", fontsize=12)
    plt.ylabel("Median squared error (dB)", fontsize=12)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.xlim([betas.min(), betas.max()])
    plt.ylim([-18, 0])
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()
