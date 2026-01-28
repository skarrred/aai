#p5b
#B)Implementing a computer vision project such as object detection.
from matplotlib import pyplot as plt
from skimage.feature import blob_log
from math import sqrt
from skimage.io import imread
import glob

# Load the image file path
example_file = glob.glob(r"/content/sun.jpeg")[0]

# Read image as grayscale
im = imread(example_file, as_gray=True)

# Use a grayscale colormap
cm_gray = plt.cm.gray

# Show input image
plt.imshow(im, cmap=cm_gray)
plt.title("Input Image")
plt.axis('off')
plt.show()

# Detect blobs using Laplacian of Gaussian method
blobs_log = blob_log(im, max_sigma=30, num_sigma=10, threshold=0.1)

# Compute radii in the 3rd column
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

# Print number of blobs detected
num_blobs = len(blobs_log)
print("Number of objects detected:", num_blobs)

# Plot blobs on the image
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(im, cmap=cm_gray)
for blob in blobs_log:
    y, x, r = blob
    circle = plt.Circle((x, y), r + 5, color='lime', linewidth=2, fill=False)
    ax.add_patch(circle)

plt.title("Detected Blobs (LoG)")
plt.axis('off')
plt.show()
