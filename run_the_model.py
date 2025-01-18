import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from UNet import Unet
from sklearn.metrics import accuracy_score
import numpy as np
from const import *


# Load model
def load_model(model_path, channels, num_classes, device):
    """
    Load the U-Net model from the given path.
    Args:
        model_path: Path to the saved model checkpoint.
        channels: List of channel sizes for the model.
        num_classes: Number of output classes.
        device: Device to use ('cuda' or 'cpu').
    Returns:
        Loaded model.
    """
    model = Unet(channels=channels, no_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint[0])
    model.eval()
    return model


# Preprocessing
def preprocess_image(image, device):
    """
    Preprocess an input image for the model.
    Args:
        image: PIL Image or ndarray.
        device: Device to use ('cuda' or 'cpu').
    Returns:
        Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Add batch dimension


# Evaluate accuracy
def evaluate_accuracy(model, dataloader, device):
    """
    Evaluate the model's accuracy on a dataset.
    Args:
        model: Trained model.
        dataloader: DataLoader for the test dataset.
        device: Device to use ('cuda' or 'cpu').
    Returns:
        Accuracy of the model.
    """
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device).float()
            labels = labels.to(device).long()

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Predicted class
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Flatten the lists
    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # Calculate accuracy
    return accuracy_score(all_labels, all_preds)


# Paths and parameters
model_path = "/home/or/PycharmProjects/AMOS/Fixed Models/model 4 epoch.pt"
test_data_dir = r"/home/or/Desktop/DataSets/AMOS/amos22/Validation/mid input"
test_label_dir = r"/home/or/Desktop/DataSets/AMOS/amos22/Validation/mid label"
batch_size = 2
channels = [1, 64, 128, 256, 512, 1024]
num_classes = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test Dataset (Assumes you have a dataset class like ImageDataset)
from train_methods import ImageDataset  # Adjust import as needed

test_dataset = ImageDataset(test_data_dir, test_label_dir, transform=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = load_model(model_path, channels, num_classes, device)

# Calculate accuracy
accuracy = evaluate_accuracy(model, test_loader, device)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
