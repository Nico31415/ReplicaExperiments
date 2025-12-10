"""
Improved replica-only code for the LASSO curve in Fig. 3.

- Uses a dense beta grid
- Uses a dense lambda grid
- Uses many Monte-Carlo samples to approximate the scalar expectations
- Produces a smooth replica prediction curve (no simulations)

Copy–paste into a file, e.g. lasso_replica_fig3.py, and run.
"""

from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge  # ridge code disabled for now

import numpy as np

import matplotlib

# Use a non-interactive backend so the script can run on headless machines.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 0. Global configuration
# ============================================================

rng = np.random.default_rng(12345)

# Prior configurations
# Prior 1: Bernoulli–Gaussian with 90% zeros (rho = 0.1)
rho_bg = 0.1
var_nonzero_bg = 1.0 / rho_bg        # so that Var(x) = 1
sigma_x2_bg = rho_bg * var_nonzero_bg
assert np.isclose(sigma_x2_bg, 1.0)  # sanity check

# Prior 2: Binomial-Gaussian mixture
rho_spiky = 0.2                    # prob that coordinate uses Bin(p_one)
p_one = 0.02                       # within spiky component: P(x=1); else 0
sigma2_small = 0.3                 # variance of small Gaussian component

# Noise level: SNR0 = 10 dB
SNR0_dB = 10.0
SNR0_lin = 10 ** (SNR0_dB / 10.0)
# Noise variance (same for both priors, as in ReplicaPlots3.py)
sigma0_2 = 1.0 / SNR0_lin     # noise variance
sigma0 = np.sqrt(sigma0_2)

# Measurement ratios beta = n/m (coarser grid to speed up)
betas = np.arange(0.5, 3.01, 0.25)

# Replica / state evolution parameters
mc_se_samples = 2000            # scalar samples per iteration (increase for smoother)
max_fp_iters = 50                # max fixed-point iterations
tol_fp = 1e-5                      # convergence tolerance on sigma_eff^2

# Threshold grid (θ) for LASSO regularization search
# (wider and dense enough; same θ used in sim)
theta_grid = np.logspace(-2, 1.0, 40)
alpha_enet = 0.5  # Elastic Net mixing parameter

# Implicit regularizer pretraining parameters
beta_PT = 1.0
lambda_PT = 0.5
c_PT = 0.8
gamma = 0.3

# Monte-Carlo simulation parameters (finite-dimensional LASSO)
sim_n = 200              # signal dimension for simulations
sim_num_trials = 100     # Monte-Carlo trials per beta
max_cd_iters = 150       # coordinate descent iterations for LASSO solver


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


def sample_prior_with_mask(size, rho, var_nonzero, rng):
    """
    Sample Bernoulli–Gaussian vector and return (x, mask) where mask indicates active entries.
    """
    mask = rng.random(size) < rho
    x = np.zeros(size, dtype=float)
    x[mask] = rng.normal(loc=0.0, scale=np.sqrt(var_nonzero), size=mask.sum())
    return x, mask


def sample_bin_gauss(size, rho_spiky, p_one, sigma2_small, rng):
    """
    Sample Binomial-Gaussian mixture prior:
    - With prob rho_spiky: x_j ~ Bin(p_one) (0 or 1, "spiky" - LASSO-friendly)
    - With prob 1-rho_spiky: x_j ~ N(0, sigma2_small) (small Gaussian - ridge-friendly)
    
    Returns (x, group_mask) where group_mask[j] = True for spiky/Bin component.
    """
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


def elastic_net_denoiser(z, theta, alpha):
    """
    Proximal operator for Elastic Net penalty:
        theta * (alpha * |x| + (1-alpha)/2 * x^2)
    """
    lam1 = theta * alpha
    lam2 = theta * (1.0 - alpha)
    return (1.0 / (1.0 + lam2)) * soft_threshold(z, lam1)


# ============================================================
# Implicit-bias regularizer functions
# ============================================================

def compute_k(beta_PT, lambda_PT, c_PT, gamma):
    """
    Compute k from (beta_PT, lambda_PT, c_PT, gamma).
    """
    inner = 1.0 + np.sqrt(1.0 + (beta_PT / c_PT) ** 2)
    k = (2.0 * (lambda_PT + c_PT) * inner + gamma ** 2) ** 2
    return k

def q_scalar(z):
    """
    The scalar function q(z) from the theorem:
        q(z) = 2 - sqrt(4 + z^2) + z * asinh(z/2)
    """
    return 2.0 - np.sqrt(4.0 + z ** 2) + z * np.arcsinh(z / 2.0)

def implicit_prox(z, lam, k, tol=1e-8, max_iters=50):
    """
    Vectorized proximal operator for the penalty lam * q(2x/sqrt(k)).
    """
    if lam == 0.0:
        return z
    x = z.copy()
    s_k = np.sqrt(k)
    for _ in range(max_iters):
        u = x / s_k
        asinh_u = np.arcsinh(u)
        f = x - z + lam * (2.0 / s_k) * asinh_u
        max_res = np.max(np.abs(f))
        if max_res < tol:
            break
        df = 1.0 + lam * (2.0 / k) * (1.0 / np.sqrt(1.0 + u ** 2))
        x = x - f / df
    return x

def implicit_prox_scalar(z, lam, k, tol=1e-8, max_iters=50):
    """
    Scalar version of the prox (for coordinate descent).
    """
    if lam == 0.0:
        return z
    x = float(z)
    s_k = np.sqrt(k)
    for _ in range(max_iters):
        u = x / s_k
        asinh_u = np.arcsinh(u)
        f = x - z + lam * (2.0 / s_k) * asinh_u
        if abs(f) < tol:
            break
        df = 1.0 + lam * (2.0 / k) * (1.0 / np.sqrt(1.0 + u ** 2))
        x = x - f / df
    return x


def run_replica_grid(betas, theta_grid, mse_fn, print_prefix, **mse_kwargs):
    """
    Generic grid search for replica/state-evolution MSE over theta_grid.
    Returns mse_replica, best_theta arrays.
    """
    mse_replica = np.zeros_like(betas, dtype=float)
    best_theta = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        best_mse = np.inf
        best_th = None

        for theta in theta_grid:
            mse = mse_fn(beta, theta, **mse_kwargs)
            if mse < best_mse:
                best_mse = mse
                best_th = theta

        mse_replica[i] = best_mse
        best_theta[i] = best_th
        print(f"[Replica-{print_prefix}] beta = {beta:.3f}, best theta = {best_th:.4g}, MSE = {best_mse:.6f}")

    return mse_replica, best_theta


def lasso_coordinate_descent(A, y, theta, max_iters=150):
    """
    Cyclic coordinate descent for LASSO:
        min_x 0.5||y - A x||_2^2 + theta * ||x||_1
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

            # Update coordinate j
            r += aj * x[j]          # add back old contribution
            zj = aj.dot(r) / cj
            x_new = soft_threshold(zj, theta / cj)
            r -= aj * x_new         # subtract new contribution
            x[j] = x_new

    return x


def enet_coordinate_descent(A, y, theta, alpha, max_iters=150):
    """
    Cyclic coordinate descent for Elastic Net:
        min_x 0.5||y - A x||_2^2 + theta*(alpha*||x||_1 + (1-alpha)/2 ||x||_2^2)
    """
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


# ============================================================
# 2. Replica / state evolution for LASSO
# ============================================================

def _generic_state_evolution(beta, denoiser_fn, sample_prior_fn, rng, need_mask=False):
    """
    Generic state evolution loop for replica predictions.
    
    Args:
        beta: Measurement ratio n/m
        denoiser_fn: Function that takes (z, ...) and returns x_hat
        sample_prior_fn: Function that samples from prior, returns (x,) or (x, mask)
        rng: Random number generator
        need_mask: If True, denoiser_fn expects (z, mask) and sample_prior_fn returns (x, mask)
    
    Returns:
        Final MSE after convergence
    """
    sigma_eff2 = sigma0_2

    for _ in range(max_fp_iters):
        if need_mask:
            x, mask = sample_prior_fn(mc_se_samples, rng)
        else:
            x = sample_prior_fn(mc_se_samples, rng)
            mask = None
        
        v = rng.normal(size=mc_se_samples)
        z = x + np.sqrt(sigma_eff2) * v

        if need_mask:
            x_hat = denoiser_fn(z, mask)
        else:
            x_hat = denoiser_fn(z)

        mse = np.mean((x_hat - x) ** 2)
        sigma_eff2_new = sigma0_2 + beta * mse

        if np.abs(sigma_eff2_new - sigma_eff2) < tol_fp:
            break
        sigma_eff2 = sigma_eff2_new

    return mse


def lasso_replica_mse(beta, theta, sample_prior_fn, rng):
    """
    Replica-based / AMP-style state evolution for LASSO.

    We consider the scalar equivalent channel:
        z = x + sqrt(sigma_eff^2) * v

    and LASSO's scalar denoiser:
        x_hat = soft_threshold(z, theta)

    The fixed-point equation is:
        sigma_eff^2 = sigma0^2 + beta * E[(x - x_hat)^2].

    Parameters
    ----------
    beta : float
        Measurement ratio n/m.
    theta : float
        LASSO regularization threshold θ.
    sample_prior_fn : callable
        Function that samples from prior: sample_prior_fn(size, rng) -> x
    rng : np.random.Generator
        RNG for Monte-Carlo expectation.

    Returns
    -------
    mse : float
        Replica-predicted MSE at the fixed point.
    """
    def denoiser(z):
        return soft_threshold(z, theta)
    
    return _generic_state_evolution(beta, denoiser, sample_prior_fn, rng)


# Ridge replica code disabled for now
# def ridge_replica_mse(beta, lam, rng):
#     """
#     Replica-based / AMP-style state evolution for ridge (linear) estimator:
#
#         x_hat = z / (1 + lam * sigma_eff^2),
#
#     with fixed-point:
#         sigma_eff^2 = sigma0^2 + beta * E[(x - x_hat)^2].
#     """
#     sigma_eff2 = sigma0_2  # start from noise variance
#
#     for _ in range(max_fp_iters):
#         x = bernoulli_gaussian(mc_se_samples, rho, var_nonzero, rng)
#         v = rng.normal(size=mc_se_samples)
#
#         z = x + np.sqrt(sigma_eff2) * v
#
#         # linear shrinkage denoiser
#         x_hat = z / (1.0 + lam * sigma_eff2)
#
#         mse = np.mean((x_hat - x) ** 2)
#         sigma_eff2_new = sigma0_2 + beta * mse
#
#         if np.abs(sigma_eff2_new - sigma_eff2) < tol_fp:
#             sigma_eff2 = sigma_eff2_new
#             break
#
#         sigma_eff2 = sigma_eff2_new
#
#     return mse



def lasso_replica_curve(betas, theta_grid, sample_prior_fn, rng):
    """Wrapper using generic replica grid search for LASSO."""
    return run_replica_grid(
        betas, 
        theta_grid, 
        lambda beta, theta, rng: lasso_replica_mse(beta, theta, sample_prior_fn, rng),
        "Lasso", 
        rng=rng
    )


def enet_replica_mse(beta, theta, alpha, sample_prior_fn, rng):
    """
    Replica state-evolution MSE for Elastic Net at given beta, theta, alpha.
    """
    def denoiser(z):
        return elastic_net_denoiser(z, theta, alpha)

    return _generic_state_evolution(beta, denoiser, sample_prior_fn, rng)


def enet_replica_curve(betas, theta_grid, alpha, sample_prior_fn, rng):
    return run_replica_grid(
        betas,
        theta_grid,
        lambda beta, theta, rng: enet_replica_mse(beta, theta, alpha, sample_prior_fn, rng),
        print_prefix=f"ENet(alpha={alpha:.2f})",
        rng=rng,
    )


def hybrid_oracle_replica_mse(beta, theta, sample_prior_with_mask_fn, rng):
    """
    Oracle hybrid: L1 on active coords (mask=True), L2 on inactive coords (mask=False).
    """
    def denoiser(z, mask):
        x_hat = np.empty_like(z)
        x_hat[mask] = soft_threshold(z[mask], theta)
        x_hat[~mask] = z[~mask] / (1.0 + theta)
        return x_hat
    
    return _generic_state_evolution(beta, denoiser, sample_prior_with_mask_fn, rng, need_mask=True)


def hybrid_oracle_replica_curve(betas, theta_grid, sample_prior_with_mask_fn, rng):
    return run_replica_grid(
        betas,
        theta_grid,
        lambda beta, theta, rng: hybrid_oracle_replica_mse(beta, theta, sample_prior_with_mask_fn, rng),
        print_prefix="HybridOracle",
        rng=rng,
    )


def implicit_replica_mse(beta, lam, k, sample_prior_fn, rng):
    """
    Replica state-evolution MSE for implicit regularizer at given beta, lam, k.
    """
    def denoiser(z):
        return implicit_prox(z, lam, k)
    
    return _generic_state_evolution(beta, denoiser, sample_prior_fn, rng)

def implicit_replica_curve(betas, lambda_grid, k, sample_prior_fn, rng):
    """Wrapper using generic replica grid search for implicit regularizer."""
    return run_replica_grid(
        betas,
        lambda_grid,
        lambda beta, theta, rng: implicit_replica_mse(beta, theta, k, sample_prior_fn, rng),
        "Implicit",
        rng=rng,
    )


# Linear (ridge) replica prediction (no tuning parameter)
def linear_replica_mse(beta):
    """
    Asymptotic MSE for linear MMSE estimator:
        sigma_eff^2 = sigma0^2 + beta * mse
        mse = sigma_eff^2 / (1 + sigma_eff^2)
    """
    sigma_eff2 = sigma0_2

    for _ in range(max_fp_iters):
        mse = sigma_eff2 / (1.0 + sigma_eff2)
        sigma_eff2_new = sigma0_2 + beta * mse

        if abs(sigma_eff2_new - sigma_eff2) < tol_fp:
            break
        sigma_eff2 = sigma_eff2_new

    return sigma_eff2 / (1.0 + sigma_eff2)


def linear_replica_curve(betas):
    mse_replica = np.zeros_like(betas, dtype=float)
    for i, beta in enumerate(betas):
        mse = linear_replica_mse(beta)
        mse_replica[i] = mse
        print(f"[Replica-Linear] beta = {beta:.3f}, MSE = {mse:.6f}")
    return mse_replica


# def ridge_replica_curve(betas, lambda_grid, rng):
#     """
#     For each beta in `betas`, search over `lambda_grid` to find the λ
#     that minimizes the replica-predicted MSE for ridge.
#     """
#     mse_replica = np.zeros_like(betas, dtype=float)
#     best_lambda = np.zeros_like(betas, dtype=float)
#
#     for i, beta in enumerate(betas):
#         best_mse = np.inf
#         best_lam = None
#
#         for lam in lambda_grid:
#             mse = ridge_replica_mse(beta, lam, rng)
#             if mse < best_mse:
#                 best_mse = mse
#                 best_lam = lam
#
#         mse_replica[i] = best_mse
#         best_lambda[i] = best_lam
#         print(f"[Ridge]  beta = {beta:.3f}, best lambda = {best_lam:.4g}, MSE = {best_mse:.6f}")
#
#     return mse_replica, best_lambda



# Unused function - kept for reference but not called
# def lasso_empirical_curve(betas, lam_func, n=100, ntrials=1000, rng=None):
#     """
#     Empirically measure median SE for LASSO for each beta.
#     (Old function using sklearn - replaced by coordinate descent version)
#     """
#     pass


# ============================================================
# 3b. Monte-Carlo simulations (finite-dimensional LASSO)
# ============================================================
def _generate_simulation_trial(beta, sample_prior_fn, sample_prior_with_mask_fn, rng, need_group_mask=False):
    """
    Generate a single simulation trial (A, x_true, w, y) for given beta.
    """
    m = int(round(sim_n / beta))
    A = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, sim_n))

    if need_group_mask:
        x_true, mask = sample_prior_with_mask_fn(sim_n, rng)
    else:
        x_true = sample_prior_fn(sim_n, rng)
        mask = None
    w = rng.normal(loc=0.0, scale=sigma0, size=m)
    y = A @ x_true + w

    return (A, x_true, w, y, mask) if need_group_mask else (A, x_true, w, y)


def _simulation_curve(betas, trial_fn, rng, best_theta, label):
    """
    Run Monte-Carlo simulations across betas using provided trial_fn.
    """
    mse_median = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        theta = best_theta[i]
        mse_trials = []

        for _ in range(sim_num_trials):
            mse = trial_fn(beta, theta, rng)
            mse_trials.append(mse)

        mse_median[i] = np.median(mse_trials)
        theta_str = f", theta={theta:.4g}" if theta is not None else ""
        print(f"[Sim-{label}] beta={beta:.3f}{theta_str}, median MSE={mse_median[i]:.6f}")

    return mse_median


def lasso_simulation_curve(betas, best_theta, sample_prior_fn, rng):
    """
    Monte-Carlo curve for LASSO using coordinate descent solver.
    """
    def trial_fn(beta, theta, rng):
        A, x_true, _, y = _generate_simulation_trial(beta, sample_prior_fn, None, rng)
        x_hat = lasso_coordinate_descent(A, y, theta, max_iters=max_cd_iters)
        return np.mean((x_hat - x_true) ** 2)

    return _simulation_curve(betas, trial_fn, rng, best_theta, label="Lasso")


def enet_simulation_curve(betas, best_theta, alpha, sample_prior_fn, rng):
    """
    Monte-Carlo curve for Elastic Net using coordinate descent solver.
    """
    def trial_fn(beta, theta, rng):
        A, x_true, _, y = _generate_simulation_trial(beta, sample_prior_fn, None, rng)
        x_hat = enet_coordinate_descent(A, y, theta, alpha, max_iters=max_cd_iters)
        return np.mean((x_hat - x_true) ** 2)

    return _simulation_curve(betas, trial_fn, rng, best_theta, label="ENet")


def linear_mmse_estimator(A, y, sigma0_2):
    """Linear MMSE estimator."""
    m, n = A.shape
    M = A @ A.T + sigma0_2 * np.eye(m)
    z = np.linalg.solve(M, y)
    x_hat = A.T @ z
    return x_hat


def linear_simulation_curve(betas, sample_prior_fn, rng):
    """
    Monte-Carlo curve for linear MMSE (ridge) estimator (no hyperparameters).
    """
    mse_median = np.zeros_like(betas, dtype=float)

    for i, beta in enumerate(betas):
        m = int(round(sim_n / beta))
        mse_trials = []

        for _ in range(sim_num_trials):
            A = rng.normal(loc=0.0, scale=1.0 / np.sqrt(m), size=(m, sim_n))
            x_true = sample_prior_fn(sim_n, rng)
            w = rng.normal(loc=0.0, scale=sigma0, size=m)
            y = A @ x_true + w

            x_hat = linear_mmse_estimator(A, y, sigma0_2)
            mse_trials.append(np.mean((x_hat - x_true) ** 2))

        mse_median[i] = np.median(mse_trials)
        print(f"[Sim-Linear] beta={beta:.3f}, median MSE={mse_median[i]:.6f}")

    return mse_median


def hybrid_oracle_coordinate_descent(A, y, theta, mask, max_iters=150):
    """
    Cyclic coordinate descent for Oracle Hybrid:
      L1 on mask==True, L2 on mask==False.
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

            if mask[j]:
                zj = aj.dot(r) / cj
                x_new = soft_threshold(zj, theta / cj)
            else:
                denom = cj + theta
                x_new = aj.dot(r) / denom

            r -= aj * x_new
            x[j] = x_new

    return x


def hybrid_oracle_simulation_curve(betas, best_theta, sample_prior_with_mask_fn, rng):
    """
    Monte-Carlo curve for Oracle Hybrid L1/L2 estimator.
    """
    def trial_fn(beta, theta, rng):
        A, x_true, _, y, mask = _generate_simulation_trial(beta, None, sample_prior_with_mask_fn, rng, need_group_mask=True)
        x_hat = hybrid_oracle_coordinate_descent(A, y, theta, mask, max_iters=max_cd_iters)
        return np.mean((x_hat - x_true) ** 2)

    return _simulation_curve(betas, trial_fn, rng, best_theta, label="Hybrid")


def implicit_coordinate_descent(A, y, lambda_reg, k, max_iters=150):
    """
    Coordinate descent for:
        min_x 0.5||y - A x||^2 + lambda_reg * sum_j q(2 x_j / sqrt(k))
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
            z = aj.dot(r) / cj
            lam_eff = lambda_reg / cj
            x_new = implicit_prox_scalar(z, lam_eff, k)
            r -= aj * x_new
            x[j] = x_new

    return x

def implicit_simulation_curve(betas, best_lambda, k, sample_prior_fn, rng):
    """
    Monte-Carlo curve for implicit regularizer using coordinate descent solver.
    """
    def trial_fn(beta, theta, rng):
        A, x_true, _, y = _generate_simulation_trial(beta, sample_prior_fn, None, rng)
        x_hat = implicit_coordinate_descent(A, y, theta, k, max_iters=max_cd_iters)
        return np.mean((x_hat - x_true) ** 2)

    return _simulation_curve(betas, trial_fn, rng, best_lambda, label="Implicit")


# ============================================================
# 3. Main: compute replica curve and plot
# ============================================================

def run_all_methods_for_prior(betas, theta_grid, alpha_enet, k, sample_prior_fn, sample_prior_with_mask_fn, rng, prior_name):
    """
    Run all methods (LASSO, Elastic Net, Linear, Hybrid Oracle, Implicit) for a given prior.
    Returns dictionary with all results.
    """
    print(f"\n{'='*60}")
    print(f"Running all methods for prior: {prior_name}")
    print(f"{'='*60}\n")
    
    # Replica predictions
    mse_replica_lasso, best_theta_lasso = lasso_replica_curve(betas, theta_grid, sample_prior_fn, rng)
    mse_replica_enet, best_theta_enet = enet_replica_curve(betas, theta_grid, alpha_enet, sample_prior_fn, rng)
    mse_replica_linear = linear_replica_curve(betas)
    mse_replica_hybrid, best_theta_hybrid = hybrid_oracle_replica_curve(betas, theta_grid, sample_prior_with_mask_fn, rng)
    mse_replica_implicit, best_lambda_implicit = implicit_replica_curve(betas, theta_grid, k, sample_prior_fn, rng)
    
    # Convert to dB
    mse_replica_db_lasso = 10.0 * np.log10(mse_replica_lasso)
    mse_replica_db_enet = 10.0 * np.log10(mse_replica_enet)
    mse_replica_db_linear = 10.0 * np.log10(mse_replica_linear)
    mse_replica_db_hybrid = 10.0 * np.log10(mse_replica_hybrid)
    mse_replica_db_implicit = 10.0 * np.log10(mse_replica_implicit)
    
    # Monte-Carlo simulations
    mse_sim = lasso_simulation_curve(betas, best_theta_lasso, sample_prior_fn, rng)
    mse_enet_sim = enet_simulation_curve(betas, best_theta_enet, alpha_enet, sample_prior_fn, rng)
    mse_linear_sim = linear_simulation_curve(betas, sample_prior_fn, rng)
    mse_hybrid_sim = hybrid_oracle_simulation_curve(betas, best_theta_hybrid, sample_prior_with_mask_fn, rng)
    mse_implicit_sim = implicit_simulation_curve(betas, best_lambda_implicit, k, sample_prior_fn, rng)
    
    mse_sim_db = 10 * np.log10(mse_sim)
    mse_enet_sim_db = 10 * np.log10(mse_enet_sim)
    mse_linear_sim_db = 10 * np.log10(mse_linear_sim)
    mse_hybrid_sim_db = 10 * np.log10(mse_hybrid_sim)
    mse_implicit_sim_db = 10 * np.log10(mse_implicit_sim)
    
    return {
        'replica': {
            'lasso': mse_replica_db_lasso,
            'enet': mse_replica_db_enet,
            'linear': mse_replica_db_linear,
            'hybrid': mse_replica_db_hybrid,
            'implicit': mse_replica_db_implicit,
        },
        'sim': {
            'lasso': mse_sim_db,
            'enet': mse_enet_sim_db,
            'linear': mse_linear_sim_db,
            'hybrid': mse_hybrid_sim_db,
            'implicit': mse_implicit_sim_db,
        }
    }


if __name__ == "__main__":
    # Compute k from pretraining parameters
    k = compute_k(beta_PT, lambda_PT, c_PT, gamma)
    print(f"Computed k = {k:.6f} from (beta_PT={beta_PT}, lambda_PT={lambda_PT}, "
          f"c_PT={c_PT}, gamma={gamma})\n")
    
    # Define prior sampling functions
    def sample_prior_bg(size, rng):
        """Bernoulli-Gaussian prior sampling."""
        return bernoulli_gaussian(size, rho_bg, var_nonzero_bg, rng)
    
    def sample_prior_bg_with_mask(size, rng):
        """Bernoulli-Gaussian prior sampling with mask."""
        return sample_prior_with_mask(size, rho_bg, var_nonzero_bg, rng)
    
    def sample_prior_bin_gauss(size, rng):
        """Binomial-Gaussian mixture prior sampling (returns only x)."""
        x, _ = sample_bin_gauss(size, rho_spiky, p_one, sigma2_small, rng)
        return x
    
    def sample_prior_bin_gauss_with_mask(size, rng):
        """Binomial-Gaussian mixture prior sampling with mask."""
        return sample_bin_gauss(size, rho_spiky, p_one, sigma2_small, rng)
    
    # Run all methods for both priors
    results_bg = run_all_methods_for_prior(
        betas, theta_grid, alpha_enet, k,
        sample_prior_bg, sample_prior_bg_with_mask, rng,
        "Bernoulli-Gaussian (rho=0.1)"
    )
    
    results_bin_gauss = run_all_methods_for_prior(
        betas, theta_grid, alpha_enet, k,
        sample_prior_bin_gauss, sample_prior_bin_gauss_with_mask, rng,
        "Binomial-Gaussian mixture (rho_spiky=0.2, p_one=0.02, sigma2_small=0.3)"
    )
    
    # Create two subplots (one above the other)
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    
    # Plot 1: Bernoulli-Gaussian prior
    ax1 = axes[0]
    ax1.plot(betas, results_bg['replica']['lasso'], color="green", linestyle="-", linewidth=2.0, label="Lasso (replica)")
    ax1.plot(betas, results_bg['sim']['lasso'], "g^", markersize=6, label="Lasso (sim.)")
    ax1.plot(betas, results_bg['replica']['enet'], color="orange", linestyle="-", linewidth=2.0, label=f"Elastic Net α={alpha_enet:.2f} (replica)")
    ax1.plot(betas, results_bg['sim']['enet'], "s", color="orange", markersize=6, markerfacecolor="none", label=f"Elastic Net α={alpha_enet:.2f} (sim.)")
    ax1.plot(betas, results_bg['replica']['linear'], color="blue", linestyle="-", linewidth=2.0, label="Linear (replica)")
    ax1.plot(betas, results_bg['sim']['linear'], "bo", markersize=6, markerfacecolor="none", label="Linear (sim.)")
    ax1.plot(betas, results_bg['replica']['hybrid'], color="purple", linestyle="-", linewidth=2.0, label="Hybrid Oracle (replica)")
    ax1.plot(betas, results_bg['sim']['hybrid'], "d", color="purple", markersize=6, markerfacecolor="none", label="Hybrid Oracle (sim.)")
    ax1.plot(betas, results_bg['replica']['implicit'], color="red", linestyle="-", linewidth=2.0, label="Implicit (replica)")
    ax1.plot(betas, results_bg['sim']['implicit'], "r*", markersize=6, markerfacecolor="none", label="Implicit (sim.)")
    
    ax1.set_xlabel(r"Measurement ratio $\beta = n/m$", fontsize=12)
    ax1.set_ylabel("Median squared error (dB)", fontsize=12)
    ax1.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax1.set_xlim([betas.min(), betas.max()])
    ax1.set_ylim([-18, 0])
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(
        "Prior: Bernoulli–Gaussian, rho=0.1, active var=1/rho; SNR0=10 dB",
        fontsize=11
    )
    
    # Plot 2: Binomial-Gaussian mixture prior
    ax2 = axes[1]
    ax2.plot(betas, results_bin_gauss['replica']['lasso'], color="green", linestyle="-", linewidth=2.0, label="Lasso (replica)")
    ax2.plot(betas, results_bin_gauss['sim']['lasso'], "g^", markersize=6, label="Lasso (sim.)")
    ax2.plot(betas, results_bin_gauss['replica']['enet'], color="orange", linestyle="-", linewidth=2.0, label=f"Elastic Net α={alpha_enet:.2f} (replica)")
    ax2.plot(betas, results_bin_gauss['sim']['enet'], "s", color="orange", markersize=6, markerfacecolor="none", label=f"Elastic Net α={alpha_enet:.2f} (sim.)")
    ax2.plot(betas, results_bin_gauss['replica']['linear'], color="blue", linestyle="-", linewidth=2.0, label="Linear (replica)")
    ax2.plot(betas, results_bin_gauss['sim']['linear'], "bo", markersize=6, markerfacecolor="none", label="Linear (sim.)")
    ax2.plot(betas, results_bin_gauss['replica']['hybrid'], color="purple", linestyle="-", linewidth=2.0, label="Hybrid Oracle (replica)")
    ax2.plot(betas, results_bin_gauss['sim']['hybrid'], "d", color="purple", markersize=6, markerfacecolor="none", label="Hybrid Oracle (sim.)")
    ax2.plot(betas, results_bin_gauss['replica']['implicit'], color="red", linestyle="-", linewidth=2.0, label="Implicit (replica)")
    ax2.plot(betas, results_bin_gauss['sim']['implicit'], "r*", markersize=6, markerfacecolor="none", label="Implicit (sim.)")
    
    ax2.set_xlabel(r"Measurement ratio $\beta = n/m$", fontsize=12)
    ax2.set_ylabel("Median squared error (dB)", fontsize=12)
    ax2.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax2.set_xlim([betas.min(), betas.max()])
    ax2.set_ylim([-18, 0])
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_title(
        "Prior: Binomial-Gaussian mixture, rho_spiky=0.2, p_one=0.02, sigma2_small=0.3; SNR0=10 dB",
        fontsize=11
    )
    
    plt.tight_layout()
    
    # Save plot instead of displaying (better for headless lab machines).
    output_path = "lasso_replica.jpg"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")
