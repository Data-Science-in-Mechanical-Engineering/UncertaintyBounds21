import numpy as np
import scipy.linalg as la

# Sampling functions for data generation
def sample_rkhs_func_from_kernels2(xs, rkhs_norm=1, n_kernels=1, kernel=None):
    """Sample an RKHS function on a grid using a weighted combination canonical features
    
    Arguments
    xs            1d numpy array containing the evaluation points
    rkhs_norm     Positive float, RKHS norm of sampled function (exact). Default 1 
    num_kernels   Integer, number of canonical features used. Default 1
    kernel        Kernel to be used for sampling. Default is se_kernel with unit lenghtscale
    
    Returns
    1d numpy array of a random function from RKHS evaluated at all xs
    """
    indices = np.random.choice(np.arange(xs.size), size=n_kernels, replace=False)
    coeffs = np.random.normal(size=n_kernels)
    K = kernel(xs[indices].reshape([-1,1]), xs[indices].reshape([-1,1]))
    quad_form_val = coeffs.reshape([1,-1]) @ K @ coeffs
    coeffs /= np.sqrt(quad_form_val)
    ys = kernel(xs.reshape([-1,1]), xs[indices].reshape([-1,1])) @ coeffs.reshape([-1,1])
    return ys

def sample_rkhs_func_from_kernels(xs, rkhs_norm, n_max_kernels, kernel, n_min_kernels=5):
    n_kernels = np.random.randint(low=n_min_kernels, high=n_max_kernels, size=1)
    indices = np.random.choice(np.arange(xs.size), size=n_kernels, replace=False)
    coeffs = np.random.normal(size=n_kernels)
    K = kernel(xs[indices].reshape([-1,1]), xs[indices].reshape([-1,1]))
    quad_form_val = coeffs.reshape([1,-1]) @ K @ coeffs
    coeffs /= np.sqrt(quad_form_val)
    ys = kernel(xs.reshape([-1,1]), xs[indices].reshape([-1,1])) @ coeffs.reshape([-1,1])
    return ys
    
# ONB functions from Steinwart et al
def se_bfunc_1d(x, n, gamma=1):
    return np.sqrt(np.power(2,n)/(np.power(gamma, 2*n)*np.math.factorial(n))) \
        * np.power(x, n) * np.exp(-x**2/gamma**2)
    
def sample_rkhs_se_onb(xs, rkhs_norm, gamma=1, n_bfuncs_min=5):
    n_bfuncs = np.random.randint(low=n_bfuncs_min, high=50, size=1).item()
    indices_bfuncs = np.random.choice(np.arange(0, 50+1), size=n_bfuncs, replace=False)
    ys_bfuncs = np.zeros([len(xs), n_bfuncs])
    for i_bf in range(n_bfuncs):
        ys_bfuncs[:, i_bf] = se_bfunc_1d(xs.flatten(), indices_bfuncs[i_bf], gamma=gamma)
    coeffs = np.random.normal(size=n_bfuncs)
    coeffs = coeffs/la.norm(coeffs, ord=2)*rkhs_norm
    ys = ys_bfuncs @ coeffs
    return ys[:,None]

def dataset_generation_uniform_normal(xs, ys, n_samples, noise_level_data):
    """Data set generation: Uniformly distributed inputs and normal noise
    
    Select uniformely n_samples from xs_eval input grid and adds Gaussian noise
    with variance noise_level_data.
    
    Args
    xs            1d Numpy array with n_eval inputs
    ys                 1d Numpy array of length n_eval containing target function evaluated at xs_eval
    n_samples          Integer, number of samples to be selected (without replacement)
    noise_level_data   Variance of Gaussian noise added to target function
    """
    n_eval = len(xs)
    train_indices = np.random.choice(np.arange(n_eval), n_samples, replace=False)
    xs_train = xs[train_indices]
    ys_train = ys[train_indices] + np.random.normal(loc=0, scale=noise_level_data, size=n_samples)[:,None]
    
    return (xs_train, ys_train)

def check_bound_on_grid(func, upper_bound, lower_bound):
    """Check whether func is contained in tube described by upper_bound and lower_bound"""
    return np.any(((upper_bound - func) < 0) + ((func-lower_bound) < 0))

def check_bounds_on_grid(func, upper_bound, lower_bound):
    """Check whether func is contained in tubes
    
    Args
    func           1d numpy array containing function values at each grid point, length n_grid
    upper_bounds   n_bound x n_samples numpy array containing the upper bounds for all tubes
    lower_bounds   n_bound x n_samples numpy array containnig the lower bounds for all tubes
    """
    return np.any(((upper_bound - func.reshape([1,-1])) < 0) + ((func.reshape([1,-1])-lower_bound) < 0), axis=1)

def aposteriori_scaling(K, B, R, alpha, delta):
    """Calculate posterior SD scaling for the basic a-posteriori bound
    
    K     Kernel matrix
    B     Upper bound on RKHS norm of target function
    R     Subgaussian constant
    delta Confidence level
    """    
    alpha_bar = np.max([1.0, alpha]).item()
    return B + R/alpha*np.sqrt(np.log(la.det(alpha_bar/alpha*K + alpha_bar*np.identity(K.shape[0]))) - 2*np.log(delta))

def aposteriori_scalings_generator(K, deltas, B=1, R=1, alpha=1):
    alpha_bar = np.max([1, alpha]).item()
    detval =np.log(la.det(alpha_bar/alpha*K + alpha_bar*np.identity(K.shape[0])))
    return B + R/alpha*np.sqrt(detval - 2*np.log(deltas))

def aposteriori_rescalings_generator(K, n_scalings, delta, low=2, B=1, R=1, alpha=1):
    # Generate equidistant scalings between a given constant scaling and the scaling from the bound
    alpha_bar = np.max([1, alpha]).item()
    detval =np.log(la.det(alpha_bar/alpha*K + alpha_bar*np.identity(K.shape[0])))
    beta = B + R/alpha*np.sqrt(detval - 2*np.log(delta))
    return np.linspace(low, beta, n_scalings)

# Simple GPR implementation
def get_posterior_from_data(xs_train, ys_train, xs, config, return_kernel_mat=False):
    """Simple GPR function
    
    Args
    xs_train            Numpy 1d array of length n_train, containing training inputs
    ys_train            Numpy 1d array of length n_train, containing training outputs
    xs             Numpy 1d array of length n_eval, containing inputs for evaluation of posterior
    config              Dictionary with config options (at the moment noise_level and kernel)
    return_kernel_mat   Flag whether also the kernel matrix (required e.g. for CG17 bound) should be returned.
                        Default False
                        
    Returns
    (posterior_mean, posterior_var) (evaluated at xs_eval) if return_kernel_mat=False, otherwise
    (posterior_mean, posterior_var, K) where K is kernel matrix at xs_train
    """
    # Setup
    noise_level = config['noise_level']
    kernel = config['kernel']
    
    n_train = len(xs_train)
    
    # Build Kernel matrix
    K = kernel(xs_train, xs_train)

    #Inverse kernel matrix times training outputs (used for posterior mean)
    KinvY = la.solve(K + noise_level*np.identity(n_train), ys_train)

    # Posterior mean evaluated at test_inputs
    post_mean = kernel(xs,xs_train) @ KinvY

    # Post variance evaluated at test_inputs
    post_var = kernel.diag(xs)[:,None] - kernel(xs, xs_train) @ la.inv(K + noise_level*np.identity(n_train)) @ kernel(xs_train, xs)
    
    if return_kernel_mat:
        return (post_mean, post_var, K)
    else:
        return (post_mean, post_var)
