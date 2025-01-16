import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from UNet import Unet
from const import *

# Load the model
AMOS_NET = Unet(channels=CHANNELS, no_classes=1).double().to(DEVICE)
checkpoint = torch.load(model_path)
AMOS_NET.load_state_dict(checkpoint['model_state_dict'])
AMOS_NET.eval()

# Preprocessing function
def preprocess_image(image_path, device):
    """
    Preprocesses the input image for the U-Net model.
    Args:
        image_path: Path to the image.
        device: Device to use ('cuda' or 'cpu').
    Returns:
        Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to the input size of the model
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize (use training normalization values)
    ])
    image = Image.open(image_path).convert('L')  # Convert to grayscale if needed
    image = transform(image)
    return image.unsqueeze(0).to(device)  # Add batch dimension and move to device

# Postprocessing function
def postprocess_prediction(pred, save_path):
    """
    Postprocesses the model's prediction and saves it as an image.
    Args:
        pred: Model's raw output tensor.
        save_path: Path to save the image.
    """
    pred = pred.squeeze().detach().cpu().numpy()  # Remove batch and channel dimensions
    pred = (pred - pred.min()) / (pred.max() - pred.min())  # Normalize to [0, 1]
    pred_image = (pred * 255).astype(np.uint8)  # Scale to [0, 255]
    im = Image.fromarray(pred_image)
    im.save(save_path)

# Path to the test image
test_image_path = "amos_0029_slice21.png"
output_image_path = "prediction_result.jpeg"

# Preprocess the image
input_tensor = preprocess_image(test_image_path, DEVICE)

# Run the model on the input image
with torch.no_grad():
    prediction = AMOS_NET(input_tensor)

# Postprocess and save the prediction
postprocess_prediction(prediction, output_image_path)

print(f"Prediction saved to {output_image_path}")
