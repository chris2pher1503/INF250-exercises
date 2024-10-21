from skimage import io, color, filters, exposure, morphology
import matplotlib.pyplot as plt
import numpy as np

ImgPath = 'IMG_2754_nonstop_alltogether.JPG'
image = io.imread(ImgPath)
plt.figure()
plt.title("Original Image")
plt.imshow(image)

gray_image = color.rgb2gray(image)
plt.figure()
plt.title("Gray-scale Image")
plt.imshow(gray_image, cmap='gray')

smoothed_image = filters.gaussian(gray_image, sigma=0.5)
plt.figure()
plt.title("smoothed image")
plt.imshow(smoothed_image, cmap='gray')

binary_image = smoothed_image > filters.threshold_otsu(smoothed_image)
plt.figure()
plt.title("Binary Image")
plt.imshow(binary_image, cmap='gray')


cleaned_image = morphology.remove_small_objects(binary_image, 5000)
cleaned_image = morphology.remove_small_holes(cleaned_image, 5000)
plt.figure()
plt.title("Cleaned Binary Image")
plt.imshow(cleaned_image, cmap='gray')

plt.show()