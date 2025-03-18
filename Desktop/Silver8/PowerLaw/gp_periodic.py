import torch
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate training data
n_train = 100
x_train = torch.linspace(0, 1, n_train)
y_train = x_train + torch.sin(x_train) + 0.1 * torch.randn(n_train)

# Generate test data
n_test = 200
x_test = torch.linspace(0, 2, n_test)
y_true = x_test + torch.sin(x_test)

# Define the GP model
class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = RBFKernel()
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModel(x_train, y_train, likelihood)

# Training
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 50
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(x_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
    if (i + 1) % 10 == 0:
        print(f'Iteration {i+1}/{training_iterations} - Loss: {loss.item():.3f}')

# Set model to evaluation mode
model.eval()
likelihood.eval()

# Generate predictions
with torch.no_grad():
    observed_pred = likelihood(model(x_test))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

# Generate samples from prior and posterior
with torch.no_grad():
    prior_samples = model(x_test).sample(torch.Size([5]))
    posterior_samples = observed_pred.sample(torch.Size([5]))

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Prior samples
plt.subplot(131)
for i in range(5):
    plt.plot(x_test.numpy(), prior_samples[i].numpy(), 'b-', alpha=0.5)
plt.title('Prior Samples (RBF Kernel)')
plt.xlabel('x')
plt.ylabel('y')

# Plot 2: Fitted GP with confidence intervals
plt.subplot(132)
plt.plot(x_test.numpy(), y_true.numpy(), 'k--', label='True Function')
plt.plot(x_test.numpy(), mean.numpy(), 'b-', label='Mean')
plt.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), color='b', alpha=0.2)
plt.scatter(x_train.numpy(), y_train.numpy(), c='r', marker='x', label='Training Data')
plt.title('Fitted GP with Confidence Intervals')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Plot 3: Posterior samples
plt.subplot(133)
for i in range(5):
    plt.plot(x_test.numpy(), posterior_samples[i].numpy(), 'b-', alpha=0.5)
plt.plot(x_test.numpy(), mean.numpy(), 'k-', label='Mean')
plt.title('Posterior Samples')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show() 