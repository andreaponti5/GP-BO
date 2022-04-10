import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

SEED = 1
n_samples = [1, 10, 100]
bounds = (-20, 20)

# Generate samples of different size and take their means.
x_mean = np.zeros((len(n_samples), 1000))
# data = np.zeros((len(n_samples), 1000))
for idx, n in enumerate(n_samples):
    np.random.seed(SEED)
    # data[idx] = np.random.randint(*bounds, (n, 1000))
    x_mean[idx] = np.mean(np.random.randint(*bounds, (n, 1000)), axis=0)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for idx, ax in enumerate(axs):
    # Plot the histogram
    ax.hist(x_mean[idx], 15, density=True, color="coral", alpha=0.6, edgecolor="crimson", linewidth=1, rwidth=0.7)
    ax.set_title("Sample size = " + str(n_samples[idx]), fontsize=18)
    ax.set_xlabel("Sample mean", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # Plot the normal bell
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    mu, std = norm.fit(x_mean[idx])
    ax.plot(x, norm.pdf(x, mu, std), color="crimson", linewidth=3)
axs[0].set_ylabel("Density", fontsize=16)
plt.tight_layout()
plt.savefig("figures/CLT_1_10_100")
plt.show()
