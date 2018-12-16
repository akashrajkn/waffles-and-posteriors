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

### Experiments and Results

**Predictive performance**: Table below shows the validation and test accuracy achieved on the `MNIST` dataset.
<table class="tg">
<tr>
  <th class="tg-xldj">Prior</th>
  <th class="tg-xldj">Validation Acc.</th>
  <th class="tg-xldj">Test Acc.</th>
</tr>
<tr>
  <td class="tg-xldj">Standard normal</td>
  <td class="tg-xldj">0.987</td>
  <td class="tg-xldj">0.992</td>
</tr>
<tr>
  <td class="tg-0pky">Log uniform</td>
  <td class="tg-0pky">0.984</td>
  <td class="tg-0pky">0.984</td>
</tr>
<tr>
  <td class="tg-0pky">Standard Cauchy</td>
  <td class="tg-0pky">0.990</td>
  <td class="tg-0pky">0.989</td>
</tr>
<tr>
  <td class="tg-0pky">Standard Gumbel</td>
  <td class="tg-0pky">0.985</td>
  <td class="tg-0pky">0.987</td>
</tr>
<tr>
  <td class="tg-0pky">Unirorm</td>
  <td class="tg-0pky">0.990</td>
  <td class="tg-0pky">0.991</td>
</tr>
</table>

**Uncertainty evaluation**
For the task of uncertainty evaluation,  we use the trained network to predict the distribution forunseen classes. We train the models on `MNIST` dataset and evaluate on the `notMNIST`[2] and `MNIST-rot`[3] datasets.
<div align="center">
  <img src="results/entropy_notmnist.png" height=200/>
  <img src="results/cdf_notmnist.png" height=200/>
</div>
Entropy of the predictive distribution for the `MNIST-rot` test set. The left figure is the histogram of entropy values and the right figure shows the corresponding cumulative distribution function.

### References
1. *Multiplicative Normalizing Flows for Variational Bayesian Neural Networks*. Christos Louizos & Max Welling. [arXiv 1703.01961](https://arxiv.org/abs/1703.01961)
2. Dataset available at: http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html
3. Dataset available at: http://www-labs.iro.umontreal.ca/~lisa/icml2007data/
