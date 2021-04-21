# -*- coding: utf-8 -*-
"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Calculate fractals with PyTorch and tensor

"""

#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image 

from supportFunction import createLogspacedVector

#%%

def mandelbrotTorch(width_res, height_res, iterations, device, x_limit = [-1, 1], y_limit = [-1j, 1j], scaled_output = False, print_var = True):
    x_cor = torch.linspace(x_limit[0], x_limit[1], width_res, dtype=torch.cfloat)
    y_cor = torch.linspace(y_limit[0], y_limit[1], height_res, dtype=torch.cfloat)
    
    x_len = len(x_cor)
    y_len = len(y_cor)
    output = torch.zeros((x_len,y_len)).to(device)
    z = torch.zeros((x_len,y_len), dtype=torch.cfloat).to(device)
    c = torch.zeros((x_len,y_len), dtype=torch.cfloat).to(device)
    threshold = 3
   
    with torch.no_grad():
        for i in range(x_len):
            c[i] = x_cor[i] + y_cor
    
        for k in range(iterations):
            z = (z * z) + c
            
            output[torch.abs(z) < threshold] += 1
            
            if(scaled_output): 
                output[torch.abs(z) < threshold] = output[torch.abs(z) < threshold] * torch.abs(z[torch.abs(z) < threshold])
                
    
    # if(scaled_output): 
    #     output[torch.abs(z) < threshold] = output[torch.abs(z) < threshold] * torch.abs(z[torch.abs(z) < threshold])
        
    return output.to('cpu')


#%%

def mandelbrotZoomTorch(x_zoom_limit, y_zoom_limit, device, x_start_limit = [-1, 1], y_start_limit = [-1, 1], w = 1920, h = 1080, iterations = 256, n_zoom = 200):
    
    ratio = w/h
    ratio = (abs(x_start_limit[0] - x_start_limit[1]))/(abs(y_start_limit[0] - y_start_limit[1]))
    
    list_of_outputs = []
    
    x_limit = x_start_limit
    y_limit = y_start_limit
    
    x_tick_left = (x_start_limit[0] - x_zoom_limit[0])/n_zoom
    x_tick_right = (x_start_limit[1] - x_zoom_limit[1])/n_zoom
    y_tick_down = (y_start_limit[0] - y_zoom_limit[0])/n_zoom
    y_tick_up = (y_start_limit[1] - y_zoom_limit[1])/n_zoom
    
    reverse_order = True
    # x_tick_left_vector  = createLogspacedVector(x_start_limit[0], x_zoom_limit[0], n_zoom, reverse_order = reverse_order)
    # x_tick_right_vector = createLogspacedVector(x_start_limit[1], x_zoom_limit[1], n_zoom, reverse_order = reverse_order)
    # y_tick_down_vector  = createLogspacedVector(y_start_limit[0], y_zoom_limit[0], n_zoom, reverse_order = reverse_order)
    # y_tick_up_vector    = createLogspacedVector(y_start_limit[1], y_zoom_limit[1], n_zoom, reverse_order = reverse_order)
    x_tick_left_vector  = np.linspace(x_start_limit[0], x_zoom_limit[0], n_zoom)
    x_tick_right_vector = np.linspace(x_start_limit[1], x_zoom_limit[1], n_zoom)
    y_tick_down_vector  = np.linspace(y_start_limit[0], y_zoom_limit[0], n_zoom)
    y_tick_up_vector    = np.linspace(y_start_limit[1], y_zoom_limit[1], n_zoom)
    
    
    # print(x_start_limit)
    # print(x_zoom_limit)
    # print(y_start_limit)
    # print(y_zoom_limit)
    # print(y_tick_up_vector)
    
    for i in range(n_zoom):
        output = mandelbrotTorch(w, h, iterations, device, x_limit, y_limit)
        # list_of_outputs.append(output.T.numpy())
        plt.imsave("img_zoom/{}.png".format(i), output.T, cmap = 'hot')
        
        # x_limit = [x_limit[0] - x_tick_left, x_limit[1] - x_tick_right]
        # y_limit = [y_limit[0] - y_tick_down, y_limit[1] - y_tick_up]
        x_limit = [x_tick_left_vector[i], x_tick_right_vector[i]]
        y_limit = [y_tick_down_vector[i], y_tick_up_vector[i]]
        
        # print(x_limit, y_limit)
        print("\t{}".format(i))
    
    # return list_of_outputs