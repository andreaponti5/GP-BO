import matplotlib.pyplot as plt
import numpy as np

from skopt import gp_minimize
from skopt.acquisition import _gaussian_acquisition
from skopt.plots import _evenly_sample


def objective(x, noise_level=None):
    # Bound: [0, 10]
    # Optima: 7.9787
    x = np.array(x)
    y = -x * np.sin(x)
    return y[0]


np.random.seed(1)
x = np.arange(0, 10, 0.01)

plt.figure(figsize=(6, 3))
plt.plot(x, list(map(objective, x.reshape((-1, 1)))), color="royalblue", linewidth=3)
plt.axvline(7.9787, ls="dashed", color="red", linewidth=2)
plt.xlabel(r"x", fontsize=14)
plt.ylabel(r"f(x)", fontsize=14)
plt.tight_layout()
plt.savefig("figures/objective")
plt.show()

res = gp_minimize(objective,
                  [(0.0, 10.0)],
                  acq_func="EI",
                  n_calls=15,
                  n_random_starts=5,
                  noise=0,
                  random_state=1)

x, x_model = _evenly_sample(res.space.dimensions[0], 1000)
x = x.reshape((-1, 1))
x_model = x_model.reshape((-1, 1))

for n_calls in range(len(res.models)):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4), gridspec_kw={'height_ratios': [2, 1]})

    n_random = len(res.x_iters) - len(res.models)

    # PLOT THE TRUE FUNCTION
    ax1.plot(x, list(map(objective, x)), color="royalblue", linewidth=3)

    # PLOT THE GP MODEL WITH THE OBSERVATION
    y_pred, sigma = res.models[n_calls].predict(x_model, return_std=True)
    curr_x_iters = res.x_iters[:n_random + n_calls + 1]
    curr_func_vals = res.func_vals[:n_random + n_calls + 1]
    # GP mean
    ax1.plot(x, y_pred, color="coral", linestyle="--", linewidth=3)
    # GP standard deviation
    ax1.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.2, fc="coral", ec="None")
    # Observation
    ax1.plot(curr_x_iters, curr_func_vals, ".", color="orangered", markersize=15,
             markeredgewidth=1.5, markeredgecolor="darkred")
    ax1.set_ylabel(r"$f(x)$", fontsize=14)
    ax1.set_ylim(-10, 10)
    ax1.set_xticks([])

    # PLOT THE ACQUISITION FUNCTION
    acq_func = res.specs["args"].get("acq_func", "EI")
    acq_func_kwargs = res.specs["args"].get("acq_func_kwargs", {})
    acq = _gaussian_acquisition(x_model, res.models[n_calls],
                                y_opt=np.min(curr_func_vals),
                                acq_func=acq_func,
                                acq_func_kwargs=acq_func_kwargs)
    next_x = x[np.argmin(acq)]
    next_acq = acq[np.argmin(acq)]
    acq = - acq
    next_acq = -next_acq
    ax2.plot(x, acq, color="teal")
    ax2.fill_between(x.ravel(), 0, acq.ravel(),
                     alpha=0.3, color='teal')
    ax2.plot(next_x, next_acq, "X", color="darkslategrey", markersize=6,
             label="Next query point")
    ax2.set_xlabel(r"$x$", fontsize=14)
    ax2.set_ylabel(r"$EI(x)$", fontsize=14)

    ax1.set_title(f"Iteration {n_calls}", fontsize=16)
    plt.tight_layout()
    plt.savefig("figures/bo_ei_" + str(n_calls))
    plt.show()
