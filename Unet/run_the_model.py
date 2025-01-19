from tkinter import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from const import *
import torch
from torchvision import transforms
from PIL import Image
from UNet import Unet
import numpy as np

def load_model(model_path, channels, no_classes, device):
    """
    Load the trained U-Net model.
    Args:
        model_path (str): Path to the saved model.
        channels (list): List of channel sizes for the model.
        no_classes (int): Number of output classes.
        device (torch.device): Device to use ('cuda' or 'cpu').
    Returns:
        torch.nn.Module: Loaded model.
    """
    model = Unet(channels=channels, no_classes=no_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def preprocess_image(image_path, device, input_size=(640, 640)):
    """
    Preprocess the input image for the model.
    Args:
        image_path (str): Path to the input image.
        device (torch.device): Device to use ('cuda' or 'cpu').
        input_size (tuple): Target size for the image.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization used during training
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale if needed
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    return image_tensor

def postprocess_prediction(prediction, output_path):
    """
    Postprocess the model's prediction and save it as an image.
    Args:
        prediction (torch.Tensor): Model's raw output tensor.
        output_path (str): Path to save the output image.
    """
    print(prediction.size())
    pred = prediction.squeeze().detach().cpu().numpy()  # Remove batch and channel dimensions
    # pred = np.argmax(pred, axis=0)  # For multi-class, get the class with max probability
    # pred_image = (pred * 255 / pred.max()).astype(np.uint8)  # Scale to [0, 255]
    Image.fromarray(pred).save(output_path)


# Paths and parameters
model_path = model_path
test_data_dir = r"/home/or/Desktop/DataSets/AMOS/amos22/Validation/mid input"
test_label_dir = r"/home/or/Desktop/DataSets/AMOS/amos22/Validation/mid label"
image_path = r"/home/or/PycharmProjects/AMOS/amos_0029_slice21.png"
output_path = "output.png"


# Test Dataset (Assumes you have a dataset class like ImageDataset)
from train_methods import ImageDataset  # Adjust import as needed

test_dataset = ImageDataset(test_data_dir, test_label_dir, transform=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = load_model(model_path, CHANNELS, NUMBER_OF_CLASSES, DEVICE)

# Preprocess the image
input_tensor = preprocess_image(image_path, DEVICE)

# Run the model on the image
with torch.no_grad():
    prediction = model(input_tensor)

# Save the prediction
postprocess_prediction(prediction, output_path)
print(f"Prediction saved to {output_path}")