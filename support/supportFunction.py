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
import math
from PIL import Image 

import cv2
        
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
    
    
def showMandlbrotOpenCV(mandlbrot, cmap = 'hot'):
    plt.imsave("tmp.png", mandlbrot, cmap = cmap)
    
    img = cv2.imread("tmp.png")
    
    cv2.namedWindow('Fractal')
    
    while(True):
        cv2.imshow("Fractal", img)
        
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break
        

#%%

def rescaleTorch(x, a = 0, b = 1):
    x = ((x - torch.min(x)) / (torch.max(x) - torch.min(x))) * (b - a) + a
    return x
        
def rescaleNumpy(x, a = 0, b = 1):
    x = ((x - np.min(x)) / (np.max(x) - np.min(x))) * (b - a) + a
    return x

def createLogspacedVector(start, stop, n_elements, reverse_order = False):
    span = abs(stop - start)
    
    x = np.geomspace(1, span + 1, n_elements)
    x = rescaleNumpy(x, start, stop)
    
    if(reverse_order):
        # direction = start - stop
        reverse_x = [stop]
        for i in range(len(x) - 1):
            diff = x[i] - x[i + 1]
            print(x[i], diff)
            reverse_x.append(reverse_x[i] + diff)
        return np.asarray(reverse_x)[::-1]
    else:
        return x

def createZoomVector(start, stop, magnitude_step):
    span = abs(stop - start)
    
    x = np.zeros(0)
    
    tmp_start = 1
    for i in range(magnitude_step):
        if(i < 3): n_zoom = 50
        else: n_zoom = 100
        
        tmp_x = np.linspace(1, span + 1, 50)
        tmp_x_order = np.floor(np.log10(np.abs(tmp_x)))
        tmp_x = tmp_x[tmp_x_order < i]
        
        tmp_start = tmp_x.max()
        
        x = np.concatenate([x, tmp_x])
    
    x = rescaleNumpy(x, start, stop) 
    return x