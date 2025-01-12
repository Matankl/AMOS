import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from skimage.io import imread_collection
from skimage.color import rgba2rgb
from skimage.transform import resize

# Set random seed for reproducibility
np.random.seed(42)

# Load and preprocess your dataset (assuming image files in a folder named 'dataset')

image_collection = imread_collection('dataset/*.png')
image_labels = imread_collection('dataset/*.png')
images = []
labels = []

# Example preprocessing for images and labels
for i, img in enumerate(image_collection):
    img_resized = resize(rgba2rgb(img), (768, 768))  # Convert RGBA to RGB and resize
    labels_resized = resize(rgba2rgb(img), (768, 768))
    images.append(img_resized.flatten())  # Flatten the image
    labels.append(labels_resized.flatten())  # Flatten the labels

images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Example visualization: Display random test image with its prediction
random_idx = np.random.randint(0, len(X_test))
random_image = X_test[random_idx].reshape(768, 768, 3)
random_label = y_test[random_idx]
random_pred = y_pred[random_idx]

plt.figure(figsize=(8, 8))
plt.imshow(random_image)
plt.title(f"True Label: {random_label}, Predicted: {random_pred}", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig("sample_prediction_demo.png", dpi=150)
plt.show()

# Display confusion matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_demo.png", dpi=150)
plt.show()
