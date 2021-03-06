# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:19:31 2021

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

from fractals import mandelbrot, mandelbrotEvolution, saveListOfMatrix
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image 

#%%
w = 800
h = 600

x = [-0.6, 0]
y = [-1, -0.625]

x = [-0.4, 0]
y = [0.7, 0.85]

output = mandelbrot(w, h, 100, x, y)

plt.imshow(output.T, cmap = "hot")

# plt.xticks([0, w/2, w], labels = [x[0], np.mean(x), x[1]])
# plt.yticks([0, h/2, h], labels = [y[0], np.mean(y), y[1]])
# plt.axis("off")

plt.show()

# from fractals import *
# saveListOfMatrix(list_of_outputs, path = 'img_video')

#%%
x = [-0.3, 0]
y = [0.7, 0.85]

list_of_outputs = mandelbrotEvolution(iterations = 66, x = x, y = y)
saveListOfMatrix(list_of_outputs, path = 'img_video_1/')

# # choose codec according to format needed
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# video=cv2.VideoWriter('video.avi', fourcc, 1, (1920,1080))

# for j in range(100):
#    img = cv2.imread('img_video/' + str(j) +'.png')
#    video.write(img)

# cv2.destroyAllWindows()
# video.release()

#%%
# w, h, zoom =  3840, 2160, 4

# julia(w, h, zoom)