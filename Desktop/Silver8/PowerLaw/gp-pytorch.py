import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel
from gpytorch.means import LinearMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import NormalPrior, GammaPrior
import pandas as pd

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and prepare data
df = pd.read_csv('PowerLaw/btc_data_bitinfocharts.csv')
filtered_df = df.dropna()
log_days = filtered_df['log_days_since_genesis'].values.reshape(-1, 1)
log_prices = filtered_df['log_price_BTC'].values

# Define number of bins and points per bin for downsampling
num_bins = 100
points_per_bin = 10
bins = np.linspace(log_days.min(), log_days.max(), num_bins)
downsampled_log_days, downsampled_log_prices = [], []

for i in range(len(bins) - 1):
    mask = (log_days.flatten() >= bins[i]) & (log_days.flatten() < bins[i + 1])
    indices = np.where(mask)[0]
    if len(indices) > 0:
        sampled_indices = np.random.choice(indices, min(points_per_bin, len(indices)), replace=False)
        downsampled_log_days.extend(log_days[sampled_indices].flatten())
        downsampled_log_prices.extend(log_prices[sampled_indices])

downsampled_log_days = np.array(downsampled_log_days).reshape(-1, 1)
downsampled_log_prices = np.array(downsampled_log_prices)
sorted_indices = np.argsort(downsampled_log_days.flatten())
downsampled_log_days = downsampled_log_days[sorted_indices]
downsampled_log_prices = downsampled_log_prices[sorted_indices]

# Prepare data for GP
x = np.log10(np.exp(downsampled_log_days))
y = np.log10(np.exp(downsampled_log_prices))
x = torch.tensor(x.flatten(), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Standardize the data
x_mean, x_std = x.mean(), x.std()
y_mean, y_std = y.mean(), y.std()
x_standardized = (x - x_mean) / x_std
y_standardized = (y - y_mean) / y_std

# Split into training, validation, and test data
n_train, n_val = 400, 100
x_train, y_train = x_standardized[:n_train], y_standardized[:n_train]
x_val, y_val = x_standardized[n_train:n_train + n_val], y_standardized[n_train:n_train + n_val]
x_test, y_true = x_standardized[n_train + n_val:], y_standardized[n_train + n_val:]

# Perform OLS regression on training data
print("\nOLS Regression Results:")
X_train = np.column_stack([x_train.numpy(), np.ones_like(x_train.numpy())])
beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train.numpy()
ols_gradient = beta[0]
ols_intercept = beta[1]

# Calculate R-squared
y_pred = X_train @ beta
r_squared = 1 - np.sum((y_train.numpy() - y_pred) ** 2) / np.sum((y_train.numpy() - y_train.numpy().mean()) ** 2)

print(f"OLS Gradient: {ols_gradient:.3f}")
print(f"OLS Intercept: {ols_intercept:.3f}")
print(f"R-squared: {r_squared:.3f}")

# Define the GP model
class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        
        # Set up linear mean function with very conservative priors
        self.mean_module = LinearMean(input_size=1)
        self.mean_module.weights.prior = NormalPrior(ols_gradient, 0.1)  # Use OLS gradient as prior mean
        self.mean_module.bias.prior = NormalPrior(ols_intercept, 0.1)    # Use OLS intercept as prior mean
        
        # Set up periodic kernel with conservative priors and ScaleKernel
        self.covar_module = ScaleKernel(
            PeriodicKernel(
                lengthscale_prior=GammaPrior(5.0, 2.0),  # Encourage smoother functions
                period_length_prior=NormalPrior(1.0, 1.0),  # Wider prior for period
                outputscale_prior=GammaPrior(1.0, 0.5)  # More conservative amplitude
            )
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Initialize model and likelihood with more conservative noise prior
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
likelihood.noise.prior = GammaPrior(2.0, 0.1)  # Increased noise prior to allow more uncertainty

# Initialize and train model
model = GPModel(x_train, y_train, likelihood)
model.train()
likelihood.train()

# Use Adam optimizer with smaller learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Even smaller learning rate
mll = ExactMarginalLogLikelihood(likelihood, model)

# Training loop with early stopping
best_loss = float('inf')
patience = 30  # Increased patience
patience_counter = 0

for i in range(300):  # Increased max epochs
    optimizer.zero_grad()
    output = model(x_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % 50 == 0:
        print(f'Epoch {i+1}/300 - Loss: {loss.item():.3f}')
    
    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {i+1}')
            model.load_state_dict(best_model_state)
            break

# Generate prior samples before training
model.eval()
with torch.no_grad():
    prior_dist = MultivariateNormal(
        model.mean_module(x_standardized),
        model.covar_module(x_standardized)
    )
    prior_samples = prior_dist.sample(torch.Size([5]))
    prior_mean = prior_dist.mean
    prior_lower, prior_upper = prior_dist.confidence_region()
    # Unstandardize the prior samples and statistics
    prior_samples = prior_samples * y_std + y_mean
    prior_mean = prior_mean * y_std + y_mean
    prior_lower = prior_lower * y_std + y_mean
    prior_upper = prior_upper * y_std + y_mean

# Final evaluation
model.eval()
likelihood.eval()
with torch.no_grad():
    train_output = model(x_train)
    val_output = model(x_val)
    train_nlml = -mll(train_output, y_train)
    val_nlml = -mll(val_output, y_val)

print("\nFinal Model Performance:")
print(f"Training NLML: {train_nlml:.3f}")
print(f"Validation NLML: {val_nlml:.3f}")

# Print parameters
print("\nModel Parameters:")
print("Linear Mean Function:")
print(f"Gradient: {model.mean_module.weights.item():.3f}")
print(f"Intercept: {model.mean_module.bias.item():.3f}")

print("\nPeriodic Kernel:")
print(f"Lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}")
print(f"Period Length: {model.covar_module.base_kernel.period_length.item():.3f}")
print(f"Outputscale: {model.covar_module.outputscale.item():.3f}")

print(f"\nLikelihood Noise: {likelihood.noise.item():.3f}")

# Use model for predictions
model.eval()
with torch.no_grad():
    full_pred = likelihood(model(x_standardized))
    mean_full = full_pred.mean
    lower_full, upper_full = full_pred.confidence_region()
    posterior_samples = full_pred.sample(torch.Size([5]))
    
    # Unstandardize the predictions
    mean_full = mean_full * y_std + y_mean
    lower_full = lower_full * y_std + y_mean
    upper_full = upper_full * y_std + y_mean
    posterior_samples = posterior_samples * y_std + y_mean

# Plotting
plt.figure(figsize=(20, 5))
plt.subplot(141)
plt.plot(x.numpy(), y.numpy(), 'k--', label='True Function')
plt.scatter(x[:n_train].numpy(), y[:n_train].numpy(), c='r', marker='x', label='Training Data')
plt.scatter(x[n_train:n_train+n_val].numpy(), y[n_train:n_train+n_val].numpy(), c='g', marker='o', label='Validation Data')
plt.title('Training and Validation Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(142)
for i in range(5):
    plt.plot(x.numpy(), prior_samples[i].numpy(), 'b-', alpha=0.5)
plt.plot(x.numpy(), prior_mean.numpy(), 'k-', label='Mean')
plt.fill_between(x.numpy(), prior_lower.numpy(), prior_upper.numpy(), color='b', alpha=0.2)
plt.plot(x.numpy(), y.numpy(), 'r--', label='True Function')
plt.title('Prior Samples')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(143)
plt.plot(x.numpy(), y.numpy(), 'k--', label='True Function')
plt.plot(x.numpy(), mean_full.numpy(), 'b-', label='Mean')
plt.fill_between(x.numpy(), lower_full.numpy(), upper_full.numpy(), color='b', alpha=0.2)
plt.scatter(x[:n_train].numpy(), y[:n_train].numpy(), c='r', marker='x', label='Training Data')
plt.scatter(x[n_train:n_train+n_val].numpy(), y[n_train:n_train+n_val].numpy(), c='g', marker='o', label='Validation Data')
plt.title('Posterior Mean and Uncertainty')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(144)
for i in range(5):
    plt.plot(x.numpy(), posterior_samples[i].numpy(), 'b-', alpha=0.5)
plt.plot(x.numpy(), mean_full.numpy(), 'k-', label='Mean')
plt.fill_between(x.numpy(), lower_full.numpy(), upper_full.numpy(), color='b', alpha=0.2)
plt.scatter(x[:n_train].numpy(), y[:n_train].numpy(), c='r', marker='x', label='Training Data')
plt.scatter(x[n_train:n_train+n_val].numpy(), y[n_train:n_train+n_val].numpy(), c='g', marker='o', label='Validation Data')
plt.title('Posterior Samples')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
