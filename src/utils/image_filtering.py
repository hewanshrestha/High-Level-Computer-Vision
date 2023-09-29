import math
import numpy as np
from scipy import signal


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def gauss(sigma):
    
    ### Your code here
    
    # raise NotImplementedError
    x = np.arange(math.ceil(-3*sigma), math.floor(3*sigma)+1)
    Gx = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-1/2*np.square(x/sigma))
    
    return Gx, x


def gaussianfilter(img, sigma):
    
    ### Your code here

    #Slow
    # x1 = gauss(sigma)[0]
    # x1 = x1.reshape(1,len(x1))
    # kernel = x1.T@x1
    # kernel = kernel/np.sum(kernel)

    kernel = np.expand_dims(gauss(sigma)[0], 0)
    outimage = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    outimage = signal.convolve2d(outimage, kernel.T, boundary='symm', mode='same')
    
    # raise NotImplementedError
    
    return outimage


def gaussdx(sigma):
    
    ### Your code here
    x = np.arange(math.ceil(-3*sigma), math.floor(3*sigma)+1)
    D = -1*(x/(sigma**3*np.sqrt(2*np.pi)))*np.exp(-1/2*np.square(x/sigma))
    
    # raise NotImplementedError
    
    return D, x

def gaussderiv(img, sigma):
    
    ### Your code here
    kernel = gaussdx(sigma)[0]
    kernel = kernel.reshape(1,len(kernel))
    imgDx = signal.convolve2d(img, kernel, boundary='symm', mode='same')
    imgDy = signal.convolve2d(img, kernel.T, boundary='symm', mode='same')
    # raise NotImplementedError
    
    return imgDx, imgDy