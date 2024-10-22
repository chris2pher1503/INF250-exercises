from skimage import io, color, filters, exposure, morphology
import matplotlib.pyplot as plt
import numpy as np

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

binary_image = gray_image > filters.threshold_otsu(gray_image)
plt.figure()
plt.title("Binary Image")
plt.imshow(binary_image, cmap='gray')

""" smoothed_image = filters.gaussian(binary_image, sigma=1)
plt.figure()
plt.title("smoothed image")
plt.imshow(smoothed_image, cmap='gray')
 """
 
footprint = morphology.disk(20)
""" binary_image = morphology.erosion(binary_image, footprint)
binary_image = morphology.dilation(binary_image, footprint) """

cleaned_image = morphology.closing(binary_image, footprint)
plt.figure()
plt.imshow(cleaned_image, cmap='gray')
cleaned_image = morphology.opening(cleaned_image, footprint)



""" cleaned_image = morphology.opening(binary_image, footprint) """

""" cleaned_image = morphology.area_opening(binary_image, area_threshold=100)
cleaned_image = morphology.remove_small_holes(cleaned_image, 5000)
cleaned_image = morphology.remove_small_objects(cleaned_image, 10000)  """
plt.figure()
plt.title("Cleaned Binary Image")
plt.imshow(binary_image, cmap='gray')

plt.show()