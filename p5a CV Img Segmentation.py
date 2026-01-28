#p5a
#A)Implementing a computer vision project such as image segmentation
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from skimage.io import imread

# Load image
im = imread('/content/rgb.jpg')

# Reshape image to a 2D array of pixels and 3 color values (RGB)
pixels = im.reshape((-1, 3))

# Perform KMeans clustering to segment the image into 3 colors
kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)

# Replace each pixel with the center of its cluster
segmented_img = kmeans.cluster_centers_[kmeans.labels_].astype(int)
segmented_img = segmented_img.reshape(im.shape)

# Clip values to be valid pixel range
segmented_img = np.clip(segmented_img, 0, 255)

# Plot original and segmented image side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(im)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_img)
plt.title('Segmented Image (K-means)')
plt.axis('off')

plt.show()
