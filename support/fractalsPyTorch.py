# -*- coding: utf-8 -*-
"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Calculate fractals with PyTorch and tensor

"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image 

#%%

def mandelbrotTorch(width_res, height_res, iterations, device, x_limit = [-1, 1], y_limit = [-1j, 1j], scaled_output = False, print_var = True):
    x_cor = torch.linspace(x_limit[0], x_limit[1], width_res, dtype=torch.cfloat)
    y_cor = torch.linspace(y_limit[0], y_limit[1], height_res, dtype=torch.cfloat)
    
    x_len = len(x_cor)
    y_len = len(y_cor)
    output = torch.zeros((x_len,y_len)).to(device)
    z = torch.zeros((x_len,y_len), dtype=torch.cfloat).to(device)
    c = torch.zeros((x_len,y_len), dtype=torch.cfloat).to(device)
   
    for i in range(x_len):
        c[i] = x_cor[i] + y_cor

    for k in range(iterations):
        z = (z * z) + c
        
        output[torch.abs(z) < 3] += 1
        
    return output.to('cpu')

        