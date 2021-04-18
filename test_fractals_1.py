# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:19:31 2021

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

from fractals import mandelbrot, mandelbrotTorch, mandelbrotEvolution, saveListOfMatrix, saveSingleMatrix, mandelbrotZoom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image 

#%%
w = 800
h = 600

x = [0, 2]
y = [-1, 1]

# x = [0.6, 1.1]
# y = [-1, 1]

# x = [-0.3, -0.15]
# y = [0.75, 0.85]

# x = [-0.3, -0.15]
# y = [0.75, 0.85]

output = mandelbrot(w, h, 100, x, y)

output = (output - np.min(output)) / (np.max(output) - np.min(output))

plt.imshow(output.T, cmap = "hot")
# plt.imshow(output.T)

plt.xticks([0, w/2, w], labels = [x[0], np.mean(x), x[1]])
plt.yticks([0, h/2, h], labels = [y[0], np.mean(y), y[1]])
# plt.axis("off")

plt.show()

# saveSingleMatrix(output, w = w, h = h)

# from fractals import *
# saveListOfMatrix(list_of_outputs, path = 'img_video')

#%%
w = 1920
h = 1080 
iterations = 100

# coord for standard mandelbrot
# x = [-0.3, -0.15]
# y = [0.75, 0.85]

# coord for standard mandlbrot divided by log
x = [0.2, 1.15]
y = [-1, 1]

x = [-1, 1]
y = [-1, 1]


list_of_outputs = mandelbrotEvolution(w = w, h = h, iterations = iterations, x = x, y = y)
saveListOfMatrix(list_of_outputs, path = 'img_video_1/')

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video=cv2.VideoWriter('video_1.avi', fourcc, 10, (w, h))

for j in range(len(list_of_outputs)):
    img = cv2.imread('img_video_1/' + str(j) +'.png')
    video.write(img)

cv2.destroyAllWindows()
video.release()

#%%
# w, h, zoom =  3840, 2160, 4

# julia(w, h, zoom)

#%%
w = 800
h = 600 
iterations = 100
n_zoom = 200

x = [0, 2]
y = [-1, 1]

x_end = [1.01, 1.09]
y_end = [0.80, 0.88]

list_of_outputs = mandelbrotZoom(x_end, y_end, w = w, h = h, iterations = iterations, x_start_limit = x, y_start_limit = y, n_zoom = n_zoom)
saveListOfMatrix(list_of_outputs, path = 'img_zoom/')

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video=cv2.VideoWriter('video_zoom.avi', fourcc, 10, (w, h))

for j in range(len(list_of_outputs)):
    img = cv2.imread('img_zoom/' + str(j) +'.png')
    video.write(img)

cv2.destroyAllWindows()
video.release()