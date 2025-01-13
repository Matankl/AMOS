import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, jaccard_score, f1_score
from skimage.io import imread_collection
from skimage.color import rgba2rgb
from skimage.transform import resize
from tqdm import tqdm


def extract_3x3_patch(padded_image, x, y):
    """
    Extract a 3x3 patch around a pixel at position (x, y) in the padded image.
    """
    x += 1  # Adjust for padding
    y += 1  # Adjust for padding
    patch = padded_image[x - 1:x + 2, y - 1:y + 2, :]
    return patch.flatten()  # Flatten to a 1D array


# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess your dataset
image_collection = imread_collection('/home/or/Desktop/DataSets/AMOS/amos22/Train/mini input/*.png')
image_labels = imread_collection('/home/or/Desktop/DataSets/AMOS/amos22/Train/mini label/*.png')

images = []
labels = []

for i, (img, label) in enumerate(
        tqdm(zip(image_collection, image_labels), total=len(image_collection), desc="Processing Images")):
    # print(image_labels[0][500])
    # Convert RGBA to RGB and resize input image
    img_resized = resize(rgba2rgb(img), (768, 768))

    # Pad the image to handle edge cases
    padded_image = np.pad(img_resized, pad_width=((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)

    # Iterate through each pixel in the resized image
    for x in range(img_resized.shape[0]):
        for y in range(img_resized.shape[1]):
            # Extract a 3x3 patch around the current pixel
            patch = extract_3x3_patch(padded_image, x, y)
            images.append(patch)

            # Append the corresponding label pixel
            labels.append(image_labels[i][x][y])
            # print(label_resized[x, y])


# Convert images and labels lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print("Start to train")
# Train a logistic regression model
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

print("Start to predict")
# Predict on the test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Compute evaluation metrics
jaccard = jaccard_score(y_test, y_pred, average='macro')  # Mean IoU
f1 = f1_score(y_test, y_pred, average='macro')  # Mean Dice Coefficient

print(f"Jaccard Index (Mean IoU): {jaccard}")
print(f"Dice Coefficient (F1 Score): {f1}")

