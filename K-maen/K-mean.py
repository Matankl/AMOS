import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.util import view_as_windows

# Load a sample CT/MRI image (Replace with your dataset loading)
image_path = "sample_ct_mri.png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
image = cv2.resize(image, (256, 256))  # Resize for easier processing

# Define parameters
patch_size = 5  # 5x5 neighborhood
stride = 1  # Move 1 pixel at a time
k_clusters = 16  # Number of classes in AMOS dataset

# Extract 5x5 patches for each pixel using sliding window
patches = view_as_windows(image, (patch_size, patch_size), step=stride)
H, W, _, _ = patches.shape  # Get new height & width

# Flatten patches and store with pixel (x, y) locations
feature_vectors = []
xy_coords = []

for i in range(H):
    for j in range(W):
        patch = patches[i, j].flatten()  # Convert 5x5 patch to a vector
        feature_vectors.append(patch)
        xy_coords.append([i, j])  # Store (x, y) coordinates

# Convert to numpy arrays
feature_vectors = np.array(feature_vectors)
xy_coords = np.array(xy_coords)

# Normalize the pixel intensity values (0-1 range)
feature_vectors = feature_vectors / 255.0

# Concatenate pixel positions (x, y) with features
features = np.hstack((feature_vectors, xy_coords))

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)  # Assign clusters to pixels

# Reshape the labels back to image size
segmented_image = labels.reshape(H, W)

# Display original vs clustered image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original CT/MRI Image")
ax[0].axis("off")

ax[1].imshow(segmented_image, cmap='tab10')
ax[1].set_title("K-Means Segmentation")
ax[1].axis("off")

plt.show()
