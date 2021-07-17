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
import math
import torch
from PIL import Image 

from supportFunction import createLogspacedVector, rescaleTorch

#%%

def mandelbrotTorch(width_res, height_res, iterations, device, x_limit = [-1, 1], y_limit = [-1j, 1j], scaled_output = False, print_var = True, tensor_type = torch.cfloat, fix_c = -1):
    if(scaled_output): tensor_type = torch.cfloat
    
    x_cor = torch.linspace(x_limit[0], x_limit[1], width_res, dtype = tensor_type)
    y_cor = torch.linspace(y_limit[0], y_limit[1], height_res, dtype = tensor_type)
    
    x_len = len(x_cor)
    y_len = len(y_cor)
    output = torch.zeros((x_len,y_len)).to(device)
    z = torch.zeros((x_len, y_len), dtype = tensor_type).to(device)
    c = torch.zeros((x_len, y_len), dtype = tensor_type).to(device)
    threshold = 4
    

    with torch.no_grad():
       
        for i in range(x_len):
            z[i] = x_cor[i] + y_cor
            if(fix_c == -1):
                c[i] = x_cor[i] + y_cor
            else:
                c[i] = fix_c
                
        for k in range(iterations):            
            if(scaled_output): 
                tmp_index = torch.abs(z) < threshold
                z[tmp_index] = (z[tmp_index] * z[tmp_index]) + c[tmp_index]
            else:
                z = (z * z) + c
            
            if(scaled_output): output[torch.abs(z) < threshold] = output[torch.abs(z) < threshold] * torch.abs(z[torch.abs(z) < threshold])
            output[torch.abs(z) < threshold] += 1
            
    if(scaled_output):
        output = rescaleTorch(output)
        
    return output.to('cpu').T


#%%

def mandelbrotZoomTorch(x_zoom_limit, y_zoom_limit, device, x_start_limit = [-1, 1], y_start_limit = [-1, 1], w = 1920, h = 1080, iterations = 256, n_zoom = 200):
    
    # ratio = w/h
    # ratio = (abs(x_start_limit[0] - x_start_limit[1]))/(abs(y_start_limit[0] - y_start_limit[1]))
    
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
    
    
    for i in range(n_zoom):
        output = mandelbrotTorch(w, h, iterations, device, x_limit, y_limit, tensor_type=torch.cdouble)
        # list_of_outputs.append(output.T.numpy())
        plt.imsave("img_zoom/{}.png".format(i), output.T, cmap = 'hot')
        
        # x_limit = [x_limit[0] - x_tick_left, x_limit[1] - x_tick_right]
        # y_limit = [y_limit[0] - y_tick_down, y_limit[1] - y_tick_up]
        x_limit = [x_tick_left_vector[i], x_tick_right_vector[i]]
        y_limit = [y_tick_down_vector[i], y_tick_up_vector[i]]
        
        # print(x_limit, y_limit)
        print("\t{}".format(i))
    
    # return list_of_outputs
    
    
def mandelbrotZoomTorchV2(x_zoom_limit, y_zoom_limit, device, x_start_limit = [-1, 1], y_start_limit = [-1, 1], w = 1920, h = 1080, iterations = 256, n_zoom = 200):
    
    list_of_outputs = []
    
    magnitude_start = math.floor(math.log10(abs(x_start_limit[0] - x_start_limit[1])))
    magnitude_end   = math.floor(math.log10(abs(x_zoom_limit[0] - x_zoom_limit[1])))
    difference_in_magnitude = magnitude_start - magnitude_end
    
    reverse_order = True
    x_tick_left_vector  = np.linspace(x_start_limit[0], x_zoom_limit[0], n_zoom * difference_in_magnitude)
    x_tick_right_vector = np.linspace(x_start_limit[1], x_zoom_limit[1], n_zoom * difference_in_magnitude)
    y_tick_down_vector  = np.linspace(y_start_limit[0], y_zoom_limit[0], n_zoom * difference_in_magnitude)
    y_tick_up_vector    = np.linspace(y_start_limit[1], y_zoom_limit[1], n_zoom * difference_in_magnitude)
        
    x_limit = x_start_limit
    y_limit = y_start_limit
    
    for i in range(len(x_tick_left_vector)):
        output = mandelbrotTorch(w, h, iterations, device, x_limit, y_limit, tensor_type=torch.cdouble)
        # list_of_outputs.append(output.T.numpy())
        plt.imsave("img_zoom/{}.png".format(i), output.T, cmap = 'hot')
        
        # x_limit = [x_limit[0] - x_tick_left, x_limit[1] - x_tick_right]
        # y_limit = [y_limit[0] - y_tick_down, y_limit[1] - y_tick_up]
        x_limit = [x_tick_left_vector[i], x_tick_right_vector[i]]
        y_limit = [y_tick_down_vector[i], y_tick_up_vector[i]]
        
        # print(x_limit, y_limit)
        print("\t{}".format(round((i/len(x_tick_left_vector)) * 100, 2)))
    
    # return list_of_outputs
    
    
#%%

def evolvingFractal(w, h, iterations, device = torch.device("cuda"),  x_limit = [-1, 1], y_limit = [-1j, 1j], scaled_output = False, print_var = True, cmap = 'hot'):
    tensor_type = torch.cfloat
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Mouse callback function and related variables
    ix, iy = 0, 0
    def setCoordinate(event, x, y, flags, param):
        nonlocal ix, iy
        ix, iy = x, y
    
    cv2.namedWindow('Fractal', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Fractal', setCoordinate)
    
    real_c = np.linspace(x_limit[0], x_limit[1], w)
    imag_c = np.linspace(y_limit[0], y_limit[1], h)
    angle = 0
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Fractals related variable
    x_cor = torch.linspace(x_limit[0], x_limit[1], w, dtype = tensor_type)
    y_cor = torch.linspace(y_limit[0], y_limit[1], h, dtype = tensor_type)
    
    x_len = len(x_cor)
    y_len = len(y_cor)
    output = torch.zeros((x_len,y_len)).to(device)
    z_start = torch.zeros((x_len, y_len), dtype = tensor_type).to(device)
    z = torch.zeros((x_len, y_len), dtype = tensor_type).to(device)
    c = torch.zeros((x_len, y_len), dtype = tensor_type).to(device)
    threshold = 4
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    with torch.no_grad():
        
        for i in range(x_len):
            z_start[i] = x_cor[i] + y_cor
        
        while(True):
            fix_c = complex(real_c[ix], imag_c[iy])
            # fix_c = complex(float(np.sin(angle)), float(np.cos(angle)))
            # angle += 0.05
            
            # img = mandelbrotTorch(w, h, iterations, device = torch.device("cuda"), x_limit = x_limit, y_limit = y_limit, print_var = False,  scaled_output = scaled_output, tensor_type = tensor_type, fix_c = fix_c)
            
            for k in range(iterations):
                if(k == 0):
                    z = (z_start * z_start) + fix_c
                else:
                    if(scaled_output): 
                        tmp_index = torch.abs(z) < threshold
                        z[tmp_index] = (z[tmp_index] * z[tmp_index]) + fix_c
                    else:
                        z = (z * z) + fix_c
                
                if(scaled_output): output[torch.abs(z) < threshold] = output[torch.abs(z) < threshold] * torch.abs(z[torch.abs(z) < threshold])
                output[torch.abs(z) < threshold] += 1
            
            plt.imsave("tmp.png", output.T.cpu(), cmap = cmap)
            img_plot = cv2.imread("tmp.png")
            cv2.imshow("Fractal", img_plot)
            output[:] = 0
            
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
        
    return output.T.cpu().numpy(), z.T.cpu().numpy(), z_start.T.cpu().numpy()
        