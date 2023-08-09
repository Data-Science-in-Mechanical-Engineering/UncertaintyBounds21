import numpy as np
import scipy.linalg as la
    
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from utilities import dataset_generation_uniform_normal, aposteriori_scaling, aposteriori_scalings_generator, aposteriori_rescalings_generator, check_bounds_on_grid
from utilities import sample_rkhs_se_onb
from experiments import run_learning_instance_experiment

from joblib import Parallel, delayed

noise_level = 0.1
rkhs_norm = 2

def run_function_instance(config, func_id):
    # Generate function
    func_config = config['target_function']
    xs = func_config['xs'].reshape([-1,1])
    kernel = func_config['kernel']
    rkhs_norm = func_config['rkhs_norm']
    n_kernels = func_config['n_kernels']
    ys = sample_rkhs_se_onb(xs, rkhs_norm, 0.2)
    
    # Run learning instances
    instance_config = {
        'dataset_generation': config['dataset_generation'],
        'training': config['training'],
        'scalings_generator': config['scalings_generator']
    }
    instance_experiment = lambda: run_learning_instance_experiment(xs, ys, instance_config)
    results = Parallel(n_jobs=config['n_jobs'])(delayed(instance_experiment)()
                                                for _ in range(config['n_rep_training']))
    
    scalings_list = [result[0] for result in results]
    failures_list = [result[1] for result in results]
    scalings_array = np.vstack(scalings_list)
    failures_array = np.vstack(failures_list)
    
    # Save
    with open('outputs/' + config['experiment_prefix'] + '_func_' + str(func_id) + '.npz', 'wb') as f:
        np.savez(f, xs=xs, ys=ys, scalings=scalings_array, failures=failures_array)
        
# Config 
kernel_gpr = RBF(length_scale=0.5)

dataset_generation_config = {
    'n_samples': 50,
    'dataset_generator': lambda xs, ys, n_samples: dataset_generation_uniform_normal(xs, ys, 50, noise_level)
}

training_config = {
    'kernel': kernel_gpr,
    'noise_level_train': noise_level
}

scalings_generator = lambda K: aposteriori_scalings_generator(K, [0.1, 0.01, 0.001, 0.0001], rkhs_norm, noise_level, noise_level)

func_config = {
    'xs': np.linspace(-1, 1, 1000),
    'kernel': kernel_gpr,
    'rkhs_norm': rkhs_norm,
    'n_kernels': 100
}

config = {
    'target_function': func_config,
    'dataset_generation': dataset_generation_config,
    'training': training_config,
    'scalings_generator': scalings_generator,
    'n_jobs': 14,
    'n_rep_training': 10000,
    'n_rep_funcs': 100,
    'experiment_prefix': 'exp_1_4_b'
}

# Run and store
for i in range(config['n_rep_funcs']):
    run_function_instance(config, i)
    
    


