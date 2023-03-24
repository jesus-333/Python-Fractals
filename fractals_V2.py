import numpy as np
from numba import jit
from numba.typed import Dict
import time

"""
%load_ext autoreload
%autoreload 2

"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Support and settings

def config_init():
    config = dict(
        dtype = np.complex128,
        x_limit = np.asarray([-1, 1]),
        y_limit = np.asarray([-1j, 1j]),
        width_res = 800,
        height_res = 600,
        fix_c = -1,
    )

    return config

def config_computation():
    config = dict(
        scaled_output = True,
        iterations = 100,
        threshold = 4,
    )
    
    return config

@jit(nopython = True, parallel = False)
def rescaleNumpy(x, a = 0, b = 1):
    x = ((x - np.min(x)) / (np.max(x) - np.min(x))) * (b - a) + a
    return x

def convert_dict_into_numba_dict(old_dict):
    new_dict = Dict()
    
    for key, value in old_dict.items(): 
        new_dict[key] = value

    return new_dict

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Computation function

# @jit(nopython = True, parallel = False)
def init_matrix(config : dict):
    x_cor = np.linspace(config['x_limit'][0], config['x_limit'][1], config['width_res'], dtype = config['dtype'])
    y_cor = np.linspace(config['y_limit'][0], config['y_limit'][1], config['height_res'], dtype = config['dtype'])
    
    x_len = len(x_cor)
    y_len = len(y_cor)
    output = np.zeros((x_len,y_len))
    z = np.zeros((x_len, y_len), dtype = config['dtype'])
    c = np.zeros((x_len, y_len), dtype = config['dtype'])


    for i in range(x_len):
        z[i] = x_cor[i] + y_cor
        if(config['fix_c'] == -1):
            c[i] = x_cor[i] + y_cor
        else:
            c[i] = config['fix_c']
    
    return z, c, output

def compute_Mandelbrot(z, c, output, config:dict, use_jit:bool):
    for k in range(config['iterations']): 
        if use_jit:
            z, output = Mandelbrot_step_jit(z, c, output, config)
        else:
            z, output = Mandelbrot_step_nojit(z, c, output, config)

    if(config['scaled_output']): output = rescaleNumpy(output)
        
    return z, output


@jit(nopython = True, parallel = False)
def Mandelbrot_step_jit(z, c, output, config):
    z = (z * z) + c
    
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.abs(z[i, j]) < config['threshold']: output[i, j] += 1

    return z, output

def Mandelbrot_step_nojit(z, c, output, config):
    z = (z * z) + c
    
    output[np.abs(z) < config['threshold']] += 1

    return z, output


# TO DEBUG FOR JIT
@jit(nopython = True, parallel = False)
def Mandelbrot_step_scaled(z, c, output, config):
    tmp_index = np.abs(z) < config['threshold']
    
    for i in range(tmp_index.shape[0]):
        for j in range(tmp_index.shape[1]):
            tmp_value = z[i, j]

            z[i, j] = (tmp_value * tmp_value) + c[i, j]

    # z[tmp_index] = (z[tmp_index] * z[tmp_index]) + c[tmp_index]
    
    output[np.abs(z) < config['threshold']] = output[np.abs(z) < config['threshold']] * np.abs(z[np.abs(z) < config['threshold']])
    output[np.abs(z) < config['threshold']] += 1

    return z, output

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main():
    init_config =config_init()
    computation_config =  convert_dict_into_numba_dict(config_computation())

    st = time.time()
    z, c, output = init_matrix(init_config)
    print("Init time:\t{}".format(time.time() - st))
    
    st = time.time()
    z1, output1 = compute_Mandelbrot(z, c, output, computation_config)
    print("Comput time:\t{} (jit)".format(time.time() - st))

    st = time.time()
    z2, output2 = compute_Mandelbrot_nojit(z, c, output, computation_config)
    print("Comput time:\t{} (no jit)".format(time.time() - st))

if __name__ == "__main__":
    main()

