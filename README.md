# Gaussian Processes and Bayesian Optimization

## Functionalities
- **Central Limit Theorem**: using _clt.py_ it is possible to empirically proof the central limit theorem. You can set the number of samples used and the bounds of data.
- **Multivariate Gaussian Distributions**: with _multivariate_gaussian.py_ you can generate a multivariate Gaussian distribution, with the countour plot and the marginal. You can set the mean and covariance of the distribution.
- **Sampling from a GP**: the file _gp_prior_posterior.py_ contains the code to sample both, from the prior and from the posterior of a Gaussian Process. You can set the data on which train the GP and the number of samples to extract.  
- **Kernel functions**: using _kernel.py_ you can plot the heatmap of some kernel functions. You can set the data on which compute the covariance matrix.  
- **Bayesian Optimization**: the file _bo.py_ contains the code to optimize an objective function, using Bayesian optimization provided by scikit-optimize library. 

## Examples
The following figures show 3 iterations of Bayesian Optimization, considering the objective function _f(x) = -x*sin(x)_.  
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_0.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_1.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_2.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_3.png" width="200" height="200">
