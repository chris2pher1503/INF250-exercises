from skimage import io, color, filters
from skimage.morphology import area_closing, area_opening
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import scipy.ndimage as ndi



def threshold(image, th=None):
    shape = np.shape(image)
    binarised = np.zeros(shape)

    if len(shape) == 3:
        if th is None:
            th = filters.threshold_otsu(image)
            print(f"Terskelverdi (Otsu): {th}")
        image = image.mean(axis=2)  
    elif len(shape) > 3:
        raise ValueError('Bildet må være 2D eller ha tre kanaler (RGB).')
    for i, row in enumerate(image):
        for j, value in enumerate(row):
            binarised[i][j] = 0 if value >= th else 1
    return binarised

ImgPath = 'IMG_2754_nonstop_alltogether.JPG'
original_image = io.imread(ImgPath)
plt.figure()
plt.title("Original Image")
plt.imshow(original_image)


image = io.imread(ImgPath, as_gray=True)
gray_image = image[200:3500 , 300:5500]
plt.figure()
plt.title("Original Image")
plt.imshow(gray_image)

""" gray_image = color.rgb2gray(image)
plt.figure()
plt.title("Gray-scale Image")
plt.imshow(gray_image) """


triangle_threshold = filters.threshold_triangle(gray_image)
binary_image = threshold(gray_image, triangle_threshold)
plt.figure()
plt.title("Binary Image")
plt.imshow(binary_image)


closed_image = area_closing(binary_image, 14000)
plt.figure()
plt.imshow(closed_image)

final_image = area_opening(closed_image, 12000)
plt.figure()
plt.imshow(final_image)

final_image = final_image > 0 
distance = ndi.distance_transform_edt(final_image)
plt.figure()
plt.imshow(distance)


plt.show()