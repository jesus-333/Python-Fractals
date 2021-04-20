import sys
sys.path.insert(0, 'support/')

#%%
import cv2
import numpy as np
import time
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

from support.fractalsPython import mandelbrot
from support.fractalsPyTorch import mandelbrotTorch, mandelbrotZoomTorch
from support.supportFunction import showMandlbrot, saveSingleMatrix, saveListOfMatrix, saveListOfMatrixV2, saveListOfMatrixV3, rescale
    
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
w = 1920
h = 1080
iterations = 200
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x_limit = [-1, 1]
y_limit = [-1j, 1j]

x_limit = [-0.2240625, -0.224]
y_limit = [0.77752j, 0.77758j]

# x_limit = [0.2, 0.6]
# y_limit = [-0.6j, 0j]

t_start = time.time()
img = mandelbrotTorch(w, h, iterations, device = torch.device("cuda"), x_limit = x_limit, y_limit = y_limit, print_var = False,  scaled_output = scaled_output)
showMandlbrot(img.T, x_limit, y_limit, w, h, cmap = 'hot')
# showMandlbrot(img.T, x_limit, y_limit, w, h, cmap = 'gist_rainbow')
t_end = time.time()
print("Total exection time: {}s (Pytorch {})(Res: {}x{})".format(t_end - t_start, device.type, w, h))

# save_image(img, 'fractal_pytorch.png')
# saveSingleMatrix(img.numpy(), w = w, h= h)

# img = rescale(img.unsqueeze(0))
# R_mat = torch.ones(img.shape) * 0.5
# G_mat = torch.ones(img.shape) * 0.5
# B_mat = torch.ones(img.shape) * 0.5
# img = torch.cat((img, G_mat, B_mat), 0).numpy()

# im_PIL = Image.fromarray(img).convert('RGB') * 255 
# im_PIL.save("your_file.jpeg")

# img = cv2.merge((img, G_mat, B_mat))  # Use opencv to merge as b,g,r
# cv2.imwrite('out.png', img) 


# cmpa_vector = ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern','gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'turbo', 'nipy_spectral', 'gist_ncar']
# for cmap in cmpa_vector:
#     showMandlbrot(img.T, x_limit, y_limit, w, h, cmap = cmap)

#%% Test zoom

w = 1920
h = 1080 
iterations = 100
n_zoom = 300

x = [-1, 1]
y = [-1j, 1j]

# x_end = [1.01, 1.09]
# y_end = [0.80j, 0.88j]

x_end = [-0.2240625, -0.224]
y_end = [0.77752j, 0.77758j]

t_start = time.time()
list_of_outputs = mandelbrotZoomTorch(x_end, y_end, device = device, w = w, h = h, iterations = iterations, x_start_limit = x, y_start_limit = y, n_zoom = n_zoom)
t_end = time.time()
print("Total exection time: {}s (Pytorch {})".format(t_end - t_start, device.type))

# saveListOfMatrix(list_of_outputs, path = 'img_zoom/', w = w, h = h)
# saveListOfMatrixV2(list_of_outputs, path = 'img_zoom/')
saveListOfMatrixV3(list_of_outputs, path = 'img_zoom/', cmap = 'hot')

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video=cv2.VideoWriter('video_zoom.avi', fourcc, 10, (w, h))

for j in range(len(list_of_outputs)):
    img = cv2.imread('img_zoom/' + str(j) +'.png')
    video.write(img)

cv2.destroyAllWindows()
video.release()



#%% Test OpenCV with trackbar

# slider_max = 1
# title_window = 'title_window'
# img = mandelbrotTorch(w, h, iterations, device = torch.device("cuda"), x_limit = x_limit, y_limit = y_limit, print_var = False,  scaled_output = scaled_output)

# def on_trackbar(val):
#     x = val / slider_max
#     print(x, val)
#     cv2.imshow(title_window, img.T.numpy())

# cv2.namedWindow(title_window)
# trackbar_name = 'X x %d' % slider_max
# cv2.createTrackbar(trackbar_name, title_window , 0, slider_max, on_trackbar)
# # Show some stuff
# on_trackbar(0)
# # Wait until user press some key
# cv2.waitKey()
