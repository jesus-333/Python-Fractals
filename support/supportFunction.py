# -*- coding: utf-8 -*-
"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Support function used to save, plot etc

"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image 
        
#%%

def saveSingleMatrix(matrix, path = '', name = 'fractal', w = 1920, h = 1080):
        
    bitmap = Image.new("RGB", (w, h), "white")
    
    pix = bitmap.load() 
    
    max_value = np.max(matrix)
    step = 255/max_value
    
    # Convert the matrix in PIL image
    for i in range(w):
        for j in range(h):
            pix[i,j] = (int(matrix[i, j]) << 21) + (int(matrix[i, j]) << 10) + int(matrix[i, j])*8
            # pix[i, j] = (0, 0, 255 - int(matrix[i, j] * step))
                        
    # Save the matrix
    bitmap.save(path + name + ".png", format = 'png')


def saveListOfMatrix(list_of_matrix, path, w = 1920, h = 1080):
    for n in range(len(list_of_matrix)):
        matrix = list_of_matrix[n]
        
        bitmap = Image.new("RGB", (w, h), "white")
        
        pix = bitmap.load() 
        
        max_value = len(list_of_matrix)
        step = 255/max_value
        
        # Convert the matrix in PIL image
        for i in range(w):
            for j in range(h):
                pix[i,j] = (int(matrix[i, j]) << 21) + (int(matrix[i, j]) << 10) + int(matrix[i, j])*8
                # pix[i, j] = (int(matrix[i, j] * step), 0, 0)
                
        print("{:.2f}% completed".format(n/len(list_of_matrix) * 100))
                
        # Save the matrix
        bitmap.save(path + str(n) + ".png", format = 'png')  
        
        
def saveListOfMatrixV2(list_of_matrix, path):
    for i in range(len(list_of_matrix)):
        img = list_of_matrix[i]
        Image.fromarray(img).convert('RGB').save(path + "{}.png".format(i))
        print("{:.2f}% completed".format(i/len(list_of_matrix) * 100))
        
        
def saveListOfMatrixV3(list_of_matrix, path, cmap = 'hot'):
    for i in range(len(list_of_matrix)):
        img = list_of_matrix[i]
        plt.imsave(path + "{}.png".format(i), img, cmap = cmap)
        print("{:.2f}% completed".format(i/len(list_of_matrix) * 100))
    

def showMandlbrot(mandlbrot, x, y, w, h, figsize = (20, 15), cmap = "hot"):
    plt.figure(figsize = figsize)
    plt.imshow(mandlbrot, cmap = cmap)
    plt.xticks([0, w/2, w], labels = [x[0], np.mean(x), x[1]])
    plt.yticks([0, h/2, h], labels = [y[0], np.mean(y), y[1]])

#%%

def rescale(x, a = 0, b = 1):
    x = (x - torch.min(x))/ (torch.max(x) - torch.min(x))
    return x
        