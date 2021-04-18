import cv2
import numpy as np
import time
import torch
import matplotlib.pyplot as plt

from fractals import mandelbrot, mandelbrotTorch, showMandlbrot

#%%

# def setLimits(event,x,y,flags,param):
    
#%%

w = 1920
h = 1080

ratio = w/h

x_limit = [-1, 1]
y_limit = [-1, 1]

iterations = 100

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#%% Test execution time

t_start = time.time()
img_1 = mandelbrot(h, w, iterations, x_limit, y_limit, print_var = False)
t_end = time.time()
print("Total exection time (1): {}s (basic python)".format(t_end - t_start))

t_start = time.time()
img_2 = mandelbrotTorch(h, w, iterations, device = torch.device("cuda"), x_limit = x_limit, y_limit = [-1j, 1j], print_var = False)
t_end = time.time()
print("Total exection time (2): {}s (Pytorch CUDA)".format(t_end - t_start))

t_start = time.time()
img_3 = mandelbrotTorch(h, w, iterations, device = torch.device("cpu"), x_limit = x_limit, y_limit = [-1j, 1j], print_var = False)
t_end = time.time()
print("Total exection time (3): {}s (Pytorch CPU)".format(t_end - t_start))

showMandlbrot(img_1, x_limit, y_limit, w, h)
showMandlbrot(img_2, x_limit, y_limit, w, h)
showMandlbrot(img_3, x_limit, y_limit, w, h)


#%%
scaled_output = True

x_limit = [-1, 1]
y_limit = [-1j, 1j]

x_limit = [-0.3, -0.15]
y_limit = [0.75j, 0.85j]

img = mandelbrotTorch(w, h, iterations, device = torch.device("cuda"), x_limit = x_limit, y_limit = y_limit, print_var = False,  scaled_output = scaled_output)
showMandlbrot(img.T, x_limit, y_limit, w, h, cmap = 'hot')


# cmpa_vector = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']
# for cmap in cmpa_vector:
#     showMandlbrot(img.T, x_limit, y_limit, w, h, cmap = cmap)
