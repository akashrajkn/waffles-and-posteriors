<h2 align="center">
  Analysis of priors for Multiplicative Normalizing flows in Bayesian neural networks
</h2>

We explore Multiplicative Normalizing flows [1] in Bayesian neural networks with different prior distributions over the network weights. The prior over the parameters can not only influence how the network behaves, but can also affect the uncertainty calibration and the achievable compression rate. We experiment with uniform, Cauchy, log-uniform, Gaussian, and standard Gumbel priors on predictive accuracy and predictive uncertainty.

### Code organization
We use the code implemented by authors available here: [AMLab-Amsterdam/MNF_VBNN](https://github.com/AMLab-Amsterdam/MNF_VBNN). `src` folder contains the codes for MNF, LeNet and soft weight sharing. To run all experiments with default parameters
```
cd src/mnf
python mnf_lenet_mnist.py
```
To specify the prior distribution, modify PARAMS in `constants.py`. Available options are `['standard_normal', 'log_uniform', 'standard_cauchy', 'standard_gumbel', 'uniform']` (`gaussian_mixture` support will be added soon.)

**Dependencies**: The code requires tensorflow. We have created a `environment.yml` file with the (working) package versions. It can be installed using conda.

### Experiments

### Results
Entropy of the predictive distribution for theMNIST-rottest set. The left figure is thehistogram of entropy values and the right figure shows the corresponding cumulative distribution function.
<div align="center">
  <img src="results/entropy_notmnist.png" height=200/>
  <img src="results/cdf_notmnist.png" height=200/>
</div>

### References
[1] *Multiplicative Normalizing Flows for Variational Bayesian Neural Networks*. Christos Louizos & Max Welling. [ArXiV 1703.01961](https://arxiv.org/abs/1703.01961)
