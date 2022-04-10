# Gaussian Processes and Bayesian Optimization
## Intro

## Examples
Using _clt.py_ it is possible to empirically proof the central limit theorem. You can set the number of samples used and the bounds of data.  
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/CLT_1_10_100.png" width="600" height="200">

With _multivariate_gaussian.py_ you can generate a multivariate Gaussian distribution, with the countour plot and the marginal. You can set the mean and covariance of the distribution.  
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/multivariate_gaussian.png" width="320" height="240">

The file _gp_prior_posterior.py_ contains the code to sample both, from the prior and from the posterior of a Gaussian Process. You can set the data on which train the GP and the number of samples to extract.  
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/prior_samples.png" width="400" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/posterior_samples.png" width="400" height="200">

Using _kernel.py_ you can plot the heatmap of some kernel functions. You can set the data on which compute the covariance matrix.  
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/kernel_heatmap.png" width="600" height="150">

The file _bo.py_ contains the code to optimize an objective function, using Bayesian optimization provided by scikit-optimize library.  
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_0.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_1.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_3.png" width="200" height="200">
<img src= "https://github.com/andreaponti5/GP-BO/blob/main/figures/bo_ei_4.png" width="200" height="200">
