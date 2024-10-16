from skimage import io
import matplotlib.pyplot as plt
import skimage.filters as filters

ImgPath = 'IMG_2754_nonstop_alltogether.JPG'
image = io.imread(ImgPath)
background = filters.gaussian(image, sigma=50)

image_subtracted = image - background

plt.imshow(image_subtracted)
