import numpy as np
from numba import jit

"""
%load_ext autoreload
%autoreload 2

"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def config_init():
    config = dict(
        dtype = np.complex128,
        x_limit = [-1, 1],
        y_limit = [-1j, 1j],
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


@jit(nopython = True, parallel = False)
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

@jit(nopython = True, parallel = False)
def compute_Mandelbrot(z, c, output, config:dict):
    for k in range(config['iterations']): 
        z, output = Mandelbrot_step(z, c, output, config)

    if(config['scaled_output']): output = rescaleNumpy(output)
        
    return output.to('cpu').T

@jit(nopython = True, parallel = False)
def Mandelbrot_step(z, c, output, config):
    if(config['scaled_output']): 
        tmp_index = np.abs(z) < config['threshold']
        z[tmp_index] = (z[tmp_index] * z[tmp_index]) + c[tmp_index]
    else:
        z = (z * z) + c
    
    if(config['scaled_output']): output[np.abs(z) < config['threshold']] = output[np.abs(z) < config['threshold']] * np.abs(z[np.abs(z) < config['threshold']])
    output[np.abs(z) < config['threshold']] += 1

    return z, output
