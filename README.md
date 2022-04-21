# Gaussian Processes and Bayesian Optimization
This repository contains some basic Python code to experiment with Gaussian Processes and Bayesian Optimization using the [scikit-optimize](https://scikit-optimize.github.io/stable/ "scikit-optimize homepage") library.
## Functionalities
- **Central Limit Theorem**: using `clt.py` it is possible to empirically proof the central limit theorem. You can set the number of samples used and the bounds of data.
- **Multivariate Gaussian Distributions**: with `multivariate_gaussian.py` you can generate a multivariate Gaussian distribution, with the countour plot and the marginal. You can set the mean and covariance of the distribution.
- **Sampling from a GP**: the file `gp_prior_posterior.py` contains the code to sample both, from the prior and from the posterior of a Gaussian Process. You can set the data on which train the GP and the number of samples to extract.  
- **Kernel functions**: using `kernel.py` you can plot the heatmap of some kernel functions. You can set the data on which compute the covariance matrix.  
- **Bayesian Optimization**: the file `bo.py` contains the code to optimize an objective function, using Bayesian optimization provided by scikit-optimize library. 

## Examples
The following figures show 3 iterations of Bayesian Optimization, considering the objective function _f(x) = -x*sin(x)_.  
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_0.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_1.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_2.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_3.png" width="200" height="200">
