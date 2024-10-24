from skimage import io, color, filters, exposure, morphology
import matplotlib.pyplot as plt
import numpy as np

"from scipy import ndimage as ndi" 

ImgPath = 'IMG_2754_nonstop_alltogether.JPG'
image = io.imread(ImgPath)
image = image[200:3500 , 300:5500]
plt.figure()
plt.title("Original Image")
plt.imshow(image)

gray_image = color.rgb2gray(image)
plt.figure()
plt.title("Gray-scale Image")
plt.imshow(gray_image, cmap='gray')

smoothed_image = filters.gaussian(gray_image, sigma=1)
plt.figure()
plt.title("smoothed image")
plt.imshow(smoothed_image, cmap='gray')

binary_image = smoothed_image > filters.threshold_triangle(smoothed_image)
plt.figure()
plt.title("Binary Image")
plt.imshow(binary_image, cmap='gray')


 
footprint1 = morphology.disk(5)
footprint2 = morphology.disk(10)

cleaned_image = morphology.closing(smoothed_image, footprint1)
plt.figure()
plt.imshow(cleaned_image, cmap='gray')
cleaned_image = morphology.opening(cleaned_image, footprint2)


plt.figure()
plt.title("Cleaned Binary Image")
plt.imshow(binary_image, cmap='gray')

plt.show()