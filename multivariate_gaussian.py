import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    return np.exp(-fac / 2) / N


# Set a bivariate normal distribution over X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., 1.])
sigma = np.array([[1., -0.5], [-0.5, 1.5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, sigma)

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
                cmap=cm.plasma)

ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.plasma)
ax.contourf(X, Y, Z, zdir='x', offset=-3, cmap=cm.plasma)
ax.contourf(X, Y, Z, zdir='y', offset=4, cmap=cm.plasma)

# Adjust the limits, ticks and view angle
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("F(x1,x2)")
ax.set_zlim(-0.15, 0.2)
ax.set_zticks(np.linspace(0, 0.2, 5))
ax.view_init(27, -21)

plt.tight_layout()
plt.savefig("figures/multivariate_gaussian")
plt.show()
