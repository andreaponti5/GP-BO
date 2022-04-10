import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, DotProduct


def plot_heatmap(matrix, labels, title, ax):
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=20)
    return ax


lengthscale = 5.0
kernel_SE = RBF(lengthscale)
kernel_M_12 = Matern(lengthscale, nu=1/2)
kernel_M_32 = Matern(lengthscale, nu=3/2)
kernel_RQ = RationalQuadratic(lengthscale, alpha=1.0)
kernel_DP = DotProduct()

x = np.arange(-10, 11, 1).reshape((-1, 1))

K_SE = kernel_SE(x, x)
K_M_12 = kernel_M_12(x, x)
K_M_32 = kernel_M_32(x, x)
K_RQ = kernel_RQ(x, x)
K_DP = kernel_DP(x, x)

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
plot_heatmap(K_SE, np.arange(-10, 11, 2), r"$K_{SE}$", axs[0])
plot_heatmap(K_M_12, np.arange(-10, 11, 2), r"$K_{Mat}$, $\nu = 1/2$", axs[1])
plot_heatmap(K_M_32, np.arange(-10, 11, 2), r"$K_{Mat}$, $\nu = 3/2$", axs[2])
plot_heatmap(K_RQ, np.arange(-10, 11, 2), r"$K_{RQ}$", axs[3])
plt.tight_layout()
plt.savefig("figures/kernel_heatmap")
plt.show()
# plot_heatmap(K_DP, np.arange(-10, 11, 2), "Dot Product")
