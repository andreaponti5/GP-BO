# Gaussian Processes and Bayesian Optimization
## Intro

## Files
- _clt.py_: contains the code to empirically proof the central limit theorem. You can set the number of samples used and the bounds of data.
- _multivariate_gaussian.py_: contains the code to generate a multivariate Gaussian distribution, with the countour plot and the marginal. You can set the mean and covariance of the distribution.
- _gp_prior_posterior.py_: contains the code to sample both, from the prior and from the posterior of a Gaussian Process. You can set the data on which train the GP and the number of samples to extract.
- _kernel.py_: contains the code to plot the heatmap of some kernel functions. You can set the data on which compute the covariance matrix.
- _bo.py_: contains the code to optimize an objective function, using Bayesian optimization provided by scikit-optimize library.
