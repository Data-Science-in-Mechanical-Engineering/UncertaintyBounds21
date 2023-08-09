#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# The following script was used to generate Figure 1, right, in the main text
# Note: As an example, we use a randomly generated RKHS function from a misspecified
# kernel stored in .npz file that has to be in the same directory as this script.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from utilities import (sample_rkhs_func_from_kernels, 
                       dataset_generation_uniform_normal,
                       aposteriori_scaling,
                       check_bound_on_grid)

# Reproducability
np.random.seed(12345)

# Load an instance that failed quite often
with open('exp_1_4_b_search_func_0.npz', 'rb') as file:
    func = np.load(file)
    xs = func['xs']
    ys = func['ys']
    
# These are the settings from experiment 1.4b)
kernel_gp = RBF(length_scale=0.5)
noise_level = 1
noise_level_train = noise_level
rkhs_norm = 2
n_samples = 50

# Example confidence level
delta = 0.01

# To demonstrate how a failure looks like, we perform some repetitions of the
# learning instances now
n_rep_max = 10000

for i_rep in range(n_rep_max):
    # Build one training set
    (xs_train, ys_train) = dataset_generation_uniform_normal(xs, ys, n_samples, noise_level)
    
    # Learn function from training set
    gpr = GaussianProcessRegressor(
            kernel=kernel_gp, 
            alpha=noise_level_train, 
            optimizer=None).fit(xs_train.reshape([-1,1]),ys_train)
    (post_mean, post_sd) = gpr.predict(xs.reshape([-1,1]), return_std=True)
    K = kernel_gp(xs_train)
    
    # Get uncertainty bound
    beta = aposteriori_scaling(K, rkhs_norm, noise_level, noise_level_train, delta)
    upper_bound = post_mean[:,0] + beta*post_sd
    lower_bound = post_mean[:,0] - beta*post_sd
    
    # Check bound
    if check_bound_on_grid(ys[:,0], upper_bound, lower_bound):
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
        plt.savefig('figure_example_misspec.pdf')
        
        print(f'Failure at repetition {i_rep}')
        break
else:
    print(f'No failure within {n_rep_max} learning repetitions.')


