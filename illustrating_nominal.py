# The following script was used to generate Figure 1, left, in the main text


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from utilities import (sample_rkhs_func_from_kernels, 
                       dataset_generation_uniform_normal,
                       aposteriori_scaling)

# Reproducibility
np.random.seed(1234567)

# These are the settings from Experiment 1.1 a)
kernel = RBF(length_scale=0.2)
noise_level = 0.5
noise_level_train = noise_level
rkhs_norm = 2
n_samples = 50
n_kernels = 200
xs = np.linspace(-1, 1, 1000)

# Example confidence level
delta = 0.01

# Build function
ys = sample_rkhs_func_from_kernels(xs, rkhs_norm, n_kernels, kernel)

# Build one training set
(xs_train, ys_train) = dataset_generation_uniform_normal(xs, ys, n_samples, noise_level)

# Learn function from training set
gpr = GaussianProcessRegressor(
        kernel=kernel, 
        alpha=noise_level_train, 
        optimizer=None).fit(xs_train.reshape([-1,1]),ys_train)
(post_mean, post_sd) = gpr.predict(xs.reshape([-1,1]), return_std=True)
K = kernel(xs_train)

# Get uncertainty bound
beta = aposteriori_scaling(K, rkhs_norm, noise_level, noise_level, delta)

# Plot
matplotlib.rcParams.update({'font.size': 50})
matplotlib.rcParams.update({'lines.linewidth': 5})
matplotlib.rcParams.update({'axes.linewidth': 3})
plt.figure(figsize=(20,15))
plt.plot(xs, ys, 'g-')
plt.plot(xs_train, ys_train, 'ro', markersize=14)
plt.plot(xs, post_mean, 'b-')
plt.plot(xs, post_mean[:,0] + beta*post_sd, 'k')
plt.plot(xs, post_mean[:,0] - beta*post_sd, 'k')
plt.xlabel('Input $x$')
plt.ylabel('Output')
# Save plot
plt.savefig('figure_example_nominal.pdf')
