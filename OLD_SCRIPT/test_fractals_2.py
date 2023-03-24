import cv2
import numpy as np
import time

from fractals import mandelbrot

#%%

# def setLimits(event,x,y,flags,param):
    
#%%

w = 1000
h = 1000

ratio = w/h

x_limit = [-1, 1]
y_limit = [-1, 1]

iterations = 100

#%%

img = mandelbrot(h, w, iterations, x_limit, y_limit, print_var = False)

cv2.namedWindow('img')
# cv2.setMouseCallback('img',setLimits)

while(True):
    
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    
    cv2.imshow("img", img)
    
    if cv2.waitKey(1) == ord('q'):
        break
