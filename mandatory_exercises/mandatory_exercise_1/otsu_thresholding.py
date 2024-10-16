# -*- coding: utf-8 -*-

"""
Skeleton for first part of the blob-detection coursework as part of INF250
at NMBU (Autumn 2017).
"""

__author__ = "Christopher Ljosland Strand"
__email__ = "christopher.ljosland.strand@nmbu.no"

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.filters import threshold_otsu

def threshold(image, th=None):
    """Returns a binarised version of given image, thresholded at given value.

    Binarises the image using a global threshold `th`. Uses Otsu's method
    to find optimal thrshold value if the threshold variable is None. The
    returned image will be in the form of an 8-bit unsigned integer array
    with 255 as white and 0 as black.

    Parameters:
    -----------
    image : np.ndarray
        Image to binarise. If this image is a colour image then the last
        dimension will be the colour value (as RGB values).
    th : numeric
        Threshold value. Uses Otsu's method if this variable is None.

    Returns:
    --------
    binarised : np.ndarray(dtype=np.uint8)
        Image where all pixel values are either 0 or 255.
    """
    # Setup
    shape = np.shape(image)
    binarised = np.zeros([shape[0], shape[1]], dtype=np.uint8)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')

    if th is None:
        th = otsu(image)

    binarised[image >= th] = 255
    binarised[image < th] = 0
    


    return binarised


def histogram(image):
    """Returns the image histogram with 256 bins.
    """
    # Setup
    shape = np.shape(image)
    histogram = np.zeros(256)

    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')
    
    K = 256
    M = shape[0]*shape[1]
    histogram = np.zeros(K)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixval = int(image[i,j])
            histogram[pixval] += 1

    return histogram


def otsu(image):
    """Finds the optimal thresholdvalue of given image using Otsu's method.
    """
    hist = histogram(image)
    th = -1
    fv = -1
    pixel_number = image.shape[0] * image.shape[1]
    mean_weight = 1.0 / pixel_number
    intensity_arr = np.arange(256)
    
    for t in range(1, 255):
        pcb = np.sum(hist[:t])  
        pcf = np.sum(hist[t:])
        if pcb == 0 or pcf == 0:
            continue
        
        Wb = pcb * mean_weight  
        Wf = pcf * mean_weight
    
    
        mub = np.sum(intensity_arr[:t] * hist[:t]) / pcb  
        muf = np.sum(intensity_arr[t:] * hist[t:]) / pcf
    
        value = Wb * Wf * (mub - muf) ** 2 

        if value > fv:
            th = t
            fv = value
            
    return th



image = "gingerbreads.jpg"
gingerbreads = io.imread(image)

print(f"Optimal threshold using Otsu's method: {otsu(gingerbreads)}")
plt.figure()
plt.plot(histogram(gingerbreads))
plt.figure()
#plt.plot(threshold(gingerbreads))
io.imshow(threshold(gingerbreads))
plt.show()
