# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 09:19:31 2021

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

#%%

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 

#%% Internet code

def mandelbrot(n_rows, n_columns, iterations, x = [-1, 1], y = [-1, 1]):
    x_cor = np.linspace(x[0], x[1], n_rows)
    y_cor = np.linspace(y[0], y[1], n_columns)
    
    x_len = len(x_cor)
    y_len = len(y_cor)
    output = np.zeros((x_len,y_len))
    for i in range(x_len):
        for j in range(y_len):
            c = complex(x_cor[i],y_cor[j])
            z = complex(0, 0)
            count = 0
            for k in range(iterations):
                z = (z * z) + c
                # z = (z * z * z) + c
                
                count = count + 1
                if (abs(z) > 4):
                    break
            output[i,j] = count
        print("{:.2f}% completed".format((i/x_len)*100))
            
    return output


def julia(w =  3840, h = 2160, zoom = 4):
    # setting the width, height and zoom of the image to be created 
    # w, h, zoom =  3840, 2160, 4
   
    # creating the new image in RGB mode 
    bitmap = Image.new("RGB", (w, h), "white") 
  
    # Allocating the storage for the image and loading the pixel data. 
    pix = bitmap.load() 
     
    # setting up the variables according to the equation to  create the fractal 
    cX, cY = -0.7, 0.27015
    moveX, moveY = 0.0, 0.0
    maxIter = 255
   
    for x in range(w): 
        for y in range(h): 
            zx = 1.5*(x - w/2.5)/(0.5*zoom*w) + moveX 
            zy = 1.53*(y - h/2)/(0.8*zoom*h) + moveY 
            i = maxIter 
            while zx*zx + zy*zy < 4 and i > 1: 
                tmp = zx*zx - zy*zy + cX 
                zy,zx = 2.0*zx*zy + cY, tmp 
                i -= 1
  
            # convert byte to RGB (3 bytes), kinda magic to get nice colors 
            pix[x,y] = (i << 21) + (i << 10) + i*8
        
        print("{:.2f}% completed".format((x/w)*100))
  
    # to display the created fractal 
    bitmap.show() 
    
    
#%%

def mandelbrotEvolution(w = 1920, h = 1080, iterations = 256, x = [-1, 1], y = [-1, 1]):
    x_cor = np.linspace(x[0], x[1], w)
    y_cor = np.linspace(y[0], y[1], h)
    
    x_len = len(x_cor)
    y_len = len(y_cor)
    
    list_of_outputs = []
    real_parts = np.zeros((x_len, y_len))
    imag_parts = np.zeros((x_len, y_len))
    
    for k in range(iterations):
        if(k == 0): tmp_output = np.zeros((x_len, y_len))
        else: tmp_output = list_of_outputs[k - 1].copy()
        
        for i in range(x_len):
            for j in range(y_len):
                
                if(k == 0): z = complex(0, 0)
                else: z = complex(real_parts[i, j], imag_parts[i, j])
                
                # Avoid calculation if module of z is bigger than 4
                if (abs(z) > 4): continue
                
                # New value of z
                c = complex(x_cor[i],y_cor[j])
                z = (z * z) + c
                
                # Save real and imaginary part
                real_parts[i, j] = z.real
                imag_parts[i, j] = z.imag
                
                if(k > 0): tmp_output[i, j] += 1
        
        list_of_outputs.append(tmp_output)
        
        print("{:.2f}% completed".format((k/iterations)*100))
        
            
    return list_of_outputs


def saveListOfMatrix(list_of_matrix, path, w = 1920, h = 1080):
    for n in range(len(list_of_matrix)):
        matrix = list_of_matrix[n]
        
        bitmap = Image.new("RGB", (w, h), "white")
        
        pix = bitmap.load() 
        
        # Convert the matrix in PIL image
        for i in range(w):
            for j in range(h):
                pix[i,j] = (int(matrix[i, j]) << 21) + (int(matrix[i, j]) << 10) + int(matrix[i, j])*8
                
        print("{:.2f}% completed".format(n))
                
        # Save the matrix
        bitmap.save(path + str(n) + ".png", format = 'png')   