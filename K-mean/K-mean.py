# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from skimage.util import view_as_windows
# import os
#
# """
# We don't give K-Means the (x, y) values of the pixel
# because it gives those features too much attention and distorts clustering.
# """
# LOCATION = False
#
# # Paths to images
# image_path = r"C:\Users\matan\Desktop\Code\CS\AMOS\K-mean\amos_0001_slice61_data.jpg"
# label_path = r"C:\Users\matan\Desktop\Code\CS\AMOS\K-mean\amos_0001_slice61.jpg"
#
# # Check if paths exist
# assert os.path.exists(image_path), f"Error: Image path {image_path} not found!"
# assert os.path.exists(label_path), f"Error: Label path {label_path} not found!"
#
# # Load images
# image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale CT/MRI scan
# labels_gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Load ground truth labels
#
# # Resize for processing consistency
# image = cv2.resize(image, (256, 256))
# labels_gt = cv2.resize(labels_gt, (256, 256), interpolation=cv2.INTER_NEAREST)  # Nearest neighbor to keep discrete labels
#
# # Define parameters
# patch_size = 5  # 5x5 neighborhood
# stride = 1  # Move 1 pixel at a time
# k_clusters = 16  # Number of classes in AMOS dataset
#
# # Extract 5x5 patches for each pixel using a sliding window
# patches = view_as_windows(image, (patch_size, patch_size), step=stride)
# H, W, _, _ = patches.shape  # Get new height & width
#
# # Flatten patches and store feature vectors
# feature_vectors = []
# xy_coords = []
#
# for i in range(H):
#     for j in range(W):
#         patch = patches[i, j].flatten()  # Convert 5x5 patch to a vector
#         feature_vectors.append(patch)
#         xy_coords.append([i, j])  # Store (x, y) coordinates
#
# # Convert to numpy arrays
# feature_vectors = np.array(feature_vectors)
# xy_coords = np.array(xy_coords)
#
# # Normalize the pixel intensity values (0-1 range)
# feature_vectors = feature_vectors / 255.0
#
# # Concatenate pixel positions (x, y) with features or not
# if LOCATION:
#     features = np.hstack((feature_vectors, xy_coords))
# else:
#     features = feature_vectors
#
# # Apply K-Means Clustering
# kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
# labels = kmeans.fit_predict(features)  # Assign clusters to pixels
#
# # Reshape the labels back to image size
# segmented_image = labels.reshape(H, W)
#
# # Display original, ground truth, and clustered image
# fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title("Original CT/MRI Image")
# ax[0].axis("off")
#
# ax[1].imshow(labels_gt, cmap='tab10')
# ax[1].set_title("Ground Truth Segmentation")
# ax[1].axis("off")
#
# ax[2].imshow(segmented_image, cmap='tab10')
# ax[2].set_title("K-Means Segmentation")
# ax[2].axis("off")
#
# plt.show()


import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.util import view_as_windows
import os
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

"""
We don't give K-Means the (x, y) values of the pixel
because it gives those features too much attention and distorts clustering.
"""
LOCATION = False

# Paths to images
image_path = r"C:\Users\matan\Desktop\Code\CS\AMOS\K-mean\amos_0001_slice61_data.jpg"
label_path = r"C:\Users\matan\Desktop\Code\CS\AMOS\K-mean\amos_0001_slice61.jpg"

# Check if paths exist
assert os.path.exists(image_path), f"Error: Image path {image_path} not found!"
assert os.path.exists(label_path), f"Error: Label path {label_path} not found!"

# Load images
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale CT/MRI scan
labels_gt = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Load ground truth labels

# Resize for processing consistency
image = cv2.resize(image, (256, 256))
labels_gt = cv2.resize(labels_gt, (256, 256),
                       interpolation=cv2.INTER_NEAREST)  # Nearest neighbor to keep discrete labels

# Define parameters
patch_size = 5  # 5x5 neighborhood
stride = 1  # Move 1 pixel at a time
k_clusters = 16  # Number of classes in AMOS dataset

# Extract 5x5 patches for each pixel using a sliding window
patches = view_as_windows(image, (patch_size, patch_size), step=stride)
H, W, _, _ = patches.shape  # Get new height & width

# Crop ground truth labels to match new size (H, W)
labels_gt = labels_gt[:H, :W]

# Flatten patches and store feature vectors
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

# Concatenate pixel positions (x, y) with features or not
if LOCATION:
    features = np.hstack((feature_vectors, xy_coords))
else:
    features = feature_vectors

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(features)  # Assign clusters to pixels

# Reshape the labels back to image size
segmented_image = labels.reshape(H, W)

# Flatten arrays for evaluation
labels_gt_flat = labels_gt.flatten()
segmented_image_flat = segmented_image.flatten()

# 🔥 Ensure both arrays have the same shape before confusion matrix computation
assert labels_gt_flat.shape == segmented_image_flat.shape, \
    f"Shape mismatch: labels_gt_flat={labels_gt_flat.shape}, segmented_image_flat={segmented_image_flat.shape}"

# Compute confusion matrix (Cluster vs Ground Truth Label)
confusion = confusion_matrix(labels_gt_flat, segmented_image_flat)

# Find the best cluster-to-label mapping using the Hungarian Algorithm
row_ind, col_ind = linear_sum_assignment(-confusion)  # Negative because we want max matching

# Create mapping dictionary
cluster_to_label_map = {col: row for row, col in zip(row_ind, col_ind)}

# Relabel the clustered image based on the best mapping
mapped_segmented_image = np.vectorize(cluster_to_label_map.get)(segmented_image_flat)
mapped_segmented_image = mapped_segmented_image.reshape(H, W)

# Compute IoU for each class
iou_scores = []
for class_id in range(k_clusters):
    intersection = np.logical_and(labels_gt_flat == class_id, mapped_segmented_image.flatten() == class_id).sum()
    union = np.logical_or(labels_gt_flat == class_id, mapped_segmented_image.flatten() == class_id).sum()

    if union == 0:
        iou_scores.append(0)  # Avoid division by zero for missing classes
    else:
        iou_scores.append(intersection / union)

mean_iou = np.mean(iou_scores)  # Average IoU over all classes

# Compute pixel-wise accuracy after mapping
accuracy = np.sum(labels_gt_flat == mapped_segmented_image.flatten()) / len(labels_gt_flat)

# Print results
print(f"Mean IoU Score: {mean_iou:.4f}")
print(f"Pixel Accuracy: {accuracy:.4f}")

# Display original, ground truth, and clustered image
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original CT/MRI Image")
ax[0].axis("off")

ax[1].imshow(labels_gt, cmap='tab10')
ax[1].set_title("Ground Truth Segmentation")
ax[1].axis("off")

ax[2].imshow(mapped_segmented_image, cmap='tab10')
ax[2].set_title("Mapped K-Means Segmentation")
ax[2].axis("off")

plt.show()
