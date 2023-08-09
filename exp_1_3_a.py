import numpy as np
import scipy.linalg as la

from utilities import sample_rkhs_func_from_kernels, dataset_generation_uniform_normal, aposteriori_scaling, aposteriori_scalings_generator, aposteriori_rescalings_generator, check_bounds_on_grid
    
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from experiments import run_learning_instance_experiment, run_function_instance

# Config 
kernel_true = RBF(length_scale=0.5)
kernel_gpr = RBF(length_scale=0.2)

# Since we need a valid RKHS norm bound, we have to adjust for the different length scales
# We use Proposition 4.46, (4.44) in
# Steinwart, Christmann, Support Vector Machines, 2008
B = np.sqrt(0.5/0.2)

dataset_generation_config = {
    'n_samples': 50,
    'dataset_generator': lambda xs, ys, n_samples: dataset_generation_uniform_normal(xs, ys, 50, 0.5)
}

training_config = {
    'kernel': kernel_gpr,
    'noise_level_train': 0.5
}

#scalings_generator = lambda K: aposteriori_rescalings_generator(K, 20, 0.01, low=2, B=1, R=1, alpha=1)
scalings_generator = lambda K: aposteriori_scalings_generator(K, [0.1, 0.01, 0.001, 0.0001], B, 0.5, 0.5)

func_config = {
    'xs': np.linspace(-1, 1, 1000),
    'kernel': kernel_true,
    'rkhs_norm': 2,
    'n_kernels': 200
}

config = {
    'target_function': func_config,
    'dataset_generation': dataset_generation_config,
    'training': training_config,
    'scalings_generator': scalings_generator,
    'n_jobs': 14,
    'n_rep_training': 10000,
    'n_rep_funcs': 50,
    'experiment_prefix': 'exp_1_3_a'
}

# Run and store
for i in range(config['n_rep_funcs']):
    run_function_instance(config, i)
    


