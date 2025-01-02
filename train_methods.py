from torch.utils.data import Dataset, DataLoader
import os
import nibabel as nib
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import utils as utils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

def train(model, optimizer, batch_size, image_dir, label_dir, criterion, device):
    """
    Train the U-Net model for one epoch.

    Args:
        model: The U-Net model to train.
        optimizer: The optimizer for training.
        batch_size: Batch size for training.
        image_dir: Directory containing input images.
        label_dir: Directory containing ground truth labels.
        criterion: Loss function.
        device: Device to use ('cuda' or 'cpu').

    Returns:
        train_loss: Average training loss.
    """
    # Dataset
    train_dataset = ImageDataset(image_dir, label_dir, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.to(device)

    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(train_loader)


def validate(model, batch_size, image_dir, label_dir, criterion, device):
    """
    Validate the U-Net model for one epoch.

    Args:
        model: The U-Net model to validate.
        batch_size: Batch size for validation.
        image_dir: Directory containing validation images.
        label_dir: Directory containing ground truth labels.
        criterion: Loss function.
        device: Device to use ('cuda' or 'cpu').

    Returns:
        val_loss: Average validation loss.
    """
    # Dataset
    val_dataset = ImageDataset(image_dir, label_dir, transform=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

    return val_loss / len(val_loader)


class ImageDataset(Dataset):
    """
    Custom dataset for loading image and label pairs.
    """

    def __init__(self, image_dir, label_dir, transform=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.label_filenames = sorted(os.listdir(label_dir))


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        if self.transform:
            image = self.image_transform(image)
            label = self.label_transform(label)

        return image, label


@staticmethod
def min_max_scale(image, max_val, min_val):
    ''' Normalize an image to the range [min_val, max_val] '''
    return (image - np.min(image)) * (max_val - min_val) / (np.max(image) - np.min(image)) + min_val


class EarlyStopping(object):
    ''' Implements early stopping to prevent overfitting during training '''

    def __init__(self, patience, fname):
        '''
        Args:
            patience: Number of epochs to wait before stopping if no improvement
            fname: File name to save the best model
        '''
        self.patience = patience
        self.best_loss = np.inf  # Initialize the best loss to infinity
        self.counter = 0  # Count epochs without improvement
        self.filename = fname  # File name for saving the best model

    def __call__(self, epoch, loss, optimizer, model):
        ''' Check if early stopping condition is met '''

        if loss < self.best_loss:
            # If current loss is the best so far, reset counter and save the model
            self.counter = 0
            self.best_loss = loss

            # Save model state, optimizer state, and loss at the current epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, self.filename)

        else:
            # Increment the counter if no improvement
            self.counter += 1

        # Return True if the counter exceeds the patience threshold
        return self.counter == self.patience




#________________________________ private area ________________________________#

def old_train(model, optimizer, image_dir, label_dir, criterion, device, p_bar=None):
    ''' Training function for a single epoch using directories of CT scans '''

    # Set the model to training mode (activates dropout, batchnorm, etc.)
    model.train()

    # Initialize the optimizer gradients to zero before starting training
    optimizer.zero_grad()

    # Running loss to keep track of total loss over all batches
    running_loss = 0

    # List all files in the directories
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))

    # Ensure the directories contain the same number of files
    print("image length and lable length:",len(image_files), len(label_files))
    assert len(image_files) == len(label_files), "Mismatch between image and label files"

    for file_idx, (image_file, label_file) in enumerate(zip(image_files, label_files)):

        # Load the CT scan image and label using nibabel
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        image = nib.load(image_path).get_fdata()  # Shape: (H, W, D)
        label = nib.load(label_path).get_fdata()  # Shape: (H, W, D)
        # print shapes
        print("image shape:", image.shape, label.shape)
        print("label shape:", label.shape)

        # Ensure the CT scan and label have the same dimensions
        assert image.shape == label.shape, f"Shape mismatch in {image_file}"

        # The number of layers (slices) in the CT scan
        num_slices = image.shape[-1]  # Depth of the 3D image

        # Initialize accumulators for loss
        total_loss = 0

        # Iterate through each layer (slice)
        for slice_idx in range(num_slices):
            # Extract the 2D slice for image and label
            image_slice = image[:, :, slice_idx]
            label_slice = label[:, :, slice_idx]

            # Compute the weight map for the label slice
            weights = utils.weight_map(mask=label_slice, w0=10, sigma=5)

            # Normalize the image and label slices
            image_slice = min_max_scale(image_slice, min_val=0, max_val=1)
            label_slice = min_max_scale(label_slice, min_val=0, max_val=1)

            # Convert to tensors and move to the specified device
            image_tensor = torch.from_numpy(image_slice).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)  # (1, 1, H, W)
            label_tensor = torch.from_numpy(label_slice).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)  # (1, 1, H, W)
            weight_tensor = torch.from_numpy(weights).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)  # (1, 1, H, W)

            # Forward pass: Get predictions from the model
            y_hat = model(image_tensor)

            # Compute the loss using the provided criterion
            loss = criterion(label_tensor, y_hat, weight_tensor)

            # Accumulate loss for the current scan
            total_loss += loss

        # Backpropagation after processing all slices in the CT scan
        total_loss.backward()  # Compute gradients for the entire scan
        optimizer.step()  # Update model weights
        optimizer.zero_grad()  # Reset gradients

        # Accumulate running loss
        running_loss += total_loss.item()

        # Update the progress bar if provided
        if p_bar is not None:
            p_bar.set_postfix(loss=total_loss.item())
            p_bar.update(1)

    # Compute the average loss across all CT scans
    running_loss /= len(image_files)

    return running_loss