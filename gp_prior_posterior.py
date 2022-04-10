import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF


def plot_gpr_samples(gpr_model, n_samples, ax):
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)
    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)
    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            linewidth=3,
            alpha=0.7,
        )
    ax.plot(x, y_mean, color="black", linewidth=3, label=r"$\mu$")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ $\sigma$",
    )
    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("f(x)", fontsize=16)


rng = np.random.RandomState(4)
X_train = rng.uniform(0, 5, 10).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
n_samples = 5

kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

# Plot prior
_, ax = plt.subplots(1, 1, figsize=(8, 4))
plot_gpr_samples(gpr, n_samples=n_samples, ax=ax)
ax.set_ylim([-3.2, 3.2])
plt.tight_layout()
plt.savefig("figures/prior_samples")
plt.show()

# Plot posterior
_, ax = plt.subplots(1, 1, figsize=(8, 4))
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=ax)
ax.scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
ax.legend(loc="upper right")
ax.set_ylim([-2, 3.2])
plt.tight_layout()
plt.savefig("figures/posterior_samples")
plt.show()
