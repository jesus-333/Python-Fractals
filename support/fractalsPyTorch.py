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
    # Fractals related variable
    
    x_cor = torch.linspace(x_limit[0], x_limit[1], w, dtype = tensor_type)
    y_cor = torch.linspace(y_limit[0], y_limit[1], h, dtype = tensor_type)
    
    output = torch.zeros((w, h)).to(device)
    z_start = torch.zeros((w, h), dtype = tensor_type).to(device)
    z = torch.zeros((w, h), dtype = tensor_type).to(device)
    threshold = 4
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Mouse callback function and related variables
    
    ix, iy = 0, 0
    angle = 0
    radius = 1
    scale_factor = 0.1
    use_mouse_position = True
    
    real_c = np.linspace(x_limit[0], x_limit[1], w)
    imag_c = np.linspace(y_limit[0], y_limit[1], h)
    
    def setCoordinate(event, x, y, flags, param):
        nonlocal ix, iy, real_c, imag_c, z_start, x_limit, y_limit, use_mouse_position, angle, radius
        ix, iy = x, y
        # Mouse wheel handling (use to zoom)   
        if(event == 10):   
            x_limit[0] = x_limit[0] + np.sign(flags) * scale_factor * x_limit[0] * -1
            x_limit[1] = x_limit[1] + np.sign(flags) * scale_factor * x_limit[1] * -1
            y_limit[0] = y_limit[0] + np.sign(flags) * scale_factor * y_limit[0] * -1
            y_limit[1] = y_limit[1] + np.sign(flags) * scale_factor * y_limit[1] * -1
            
            x_cor_tmp = torch.linspace(x_limit[0], x_limit[1], w, dtype = tensor_type)
            y_cor_tmp = torch.linspace(y_limit[0], y_limit[1], h, dtype = tensor_type)
            
            for i in range(w): z_start[i] = x_cor_tmp[i] + y_cor_tmp
            
            real_c = np.linspace(x_limit[0], x_limit[1], w)
            imag_c = np.linspace(y_limit[0], y_limit[1], h)
        
        # Mouse click handling (switch between using mouse position for c or change it while time pass)
        if event == cv2.EVENT_LBUTTONDOWN: 
            use_mouse_position = not use_mouse_position
            angle = np.angle(real_c[x] + imag_c[y])
            radius = abs(real_c[x] + imag_c[y])
    
    cv2.namedWindow('Fractal', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Fractal', setCoordinate)
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Trackbar callback function
    cmap_list = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
    trackbar_name = "cmap"
    
    def changeColorMap(val): 
        nonlocal cmap
        cmap = cmap_list[val]
        
    cv2.createTrackbar(trackbar_name, 'Fractal' , 0, len(cmap_list) - 1, changeColorMap)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    with torch.no_grad():
        
        for i in range(w): z_start[i] = x_cor[i] + y_cor
        
        while(True):         
            if(use_mouse_position):
                fix_c = real_c[ix] + imag_c[iy]
            else:
                fix_c = complex(radius * float(np.cos(angle)), radius * float(np.sin(angle)))
                angle += 0.05
            
            # Evaluate fractal
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
            
            # Show image
            plt.imsave("tmp.png", output.T.cpu(), cmap = cmap)
            cv2.imshow("Fractal", cv2.imread("tmp.png"))
            
            # Exit when q is pressed
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
            
            # Reset variable image
            output[:] = 0
        
    return output.T.cpu().numpy(), z.T.cpu().numpy(), z_start.T.cpu().numpy()
        
