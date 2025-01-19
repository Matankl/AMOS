import os
import random

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from const import NUMBER_OF_CLASSES

def train(model, optimizer, batch_size, image_dir, label_dir, criterion, device, train_dataset = None):
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
        avg_iou: Average IoU score.
        avg_f1: Average F1 score.
        avg_accuracy: Average accuracy score.
    """
    if train_dataset is None:
        train_dataset = ImageDataset(image_dir, label_dir, transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.to(device)

    train_loss = 0.0
    iou_scores = []
    f1_scores = []
    accuracy_scores = []

    for images, labels in tqdm(train_loader):
        images, labels = images.to(device).float(), labels.to(device).float()

        # Forward pass
        outputs = model(images)
        outputs = outputs.squeeze(1)
        labels = labels.squeeze(1).long()
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Predictions and metrics
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        # Metrics
        iou_scores.append(calculate_iou(preds, labels_np, num_classes=model.head.out_channels))
        f1_scores.append(calculate_f1_score(preds, labels_np, num_classes=model.head.out_channels))
        accuracy_scores.append((preds == labels_np).mean())  # Compute accuracy for the batch

    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracy_scores)

    return train_loss / len(train_loader), avg_iou, avg_f1, avg_accuracy


def validate(model, batch_size, image_dir, label_dir, criterion, device, val_dataset = None):
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
        avg_iou: Average IoU score.
        avg_f1: Average F1 score.
        avg_accuracy: Average accuracy score.
    """
    if val_dataset is None:
        val_dataset = ImageDataset(image_dir, label_dir, transform=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    val_loss = 0.0
    iou_scores = []
    f1_scores = []
    accuracy_scores = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device).float(), labels.to(device).float()

            # Forward pass
            outputs = model(images)
            outputs = outputs.squeeze(1)
            labels = labels.squeeze(1).long()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Predictions and metrics
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Metrics
            iou_scores.append(calculate_iou(preds, labels_np, num_classes=model.head.out_channels))
            f1_scores.append(calculate_f1_score(preds, labels_np, num_classes=model.head.out_channels))
            accuracy_scores.append((preds == labels_np).mean())  # Compute accuracy for the batch

    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)
    avg_accuracy = np.mean(accuracy_scores)

    return val_loss / len(val_loader), avg_iou, avg_f1, avg_accuracy



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

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.label_transform = lambda x: torch.tensor(np.array(x), dtype=torch.long)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        image = Image.open(image_path).convert("L")
        label = Image.open(label_path).convert("P")


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






from sklearn.metrics import confusion_matrix
import numpy as np

import numpy as np

def calculate_iou(pred, target, num_classes):
    """
    Compute mean IoU for multi-class segmentation while avoiding fake high scores.
    """
    iou_scores = []

    for cls in range(num_classes):
        intersection = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()

        if union == 0:
            iou_scores.append(np.nan)  # Ignore empty classes
        else:
            iou_scores.append(intersection / union)

    return np.nanmean(iou_scores)  # Avoids bias from empty classes


def calculate_f1_score(pred, target, num_classes):
    """
    Calculate the F1 score.
    Args:
        pred: Predicted labels (batch_size, height, width).
        target: Ground truth labels (batch_size, height, width).
        num_classes: Number of classes.
    Returns:
        Average F1 score over all classes.
    """
    f1_scores = []
    for cls in range(num_classes):
        tp = np.logical_and(pred == cls, target == cls).sum()
        fp = np.logical_and(pred == cls, target != cls).sum()
        fn = np.logical_and(pred != cls, target == cls).sum()
        if tp + fp + fn == 0:
            f1_scores.append(1.0)  # Treat empty classes as perfect F1
        else:
            f1_scores.append(2 * tp / (2 * tp + fp + fn))
    return np.mean(f1_scores)

# weighted loss function
def compute_class_weights(dataset, num_classes, epsilon=1e-6, scaling="sqrt"):
    total_pixels = 0
    class_counts = torch.zeros(num_classes, dtype=torch.float32)

    for _, label in tqdm(dataset):
        class_counts += torch.bincount(label.flatten(), minlength=num_classes).to(torch.float32)
        total_pixels += label.numel()

    # Avoid division by zero by adding a small epsilon
    class_counts += epsilon

    # Compute weights
    if scaling == "sqrt":
        weights = 1.0 / torch.sqrt(class_counts)  # Square-root scaling
    elif scaling == "log":
        weights = 1.0 / torch.log1p(class_counts)  # Log scaling
    else:
        weights = total_pixels / (num_classes * class_counts)  # Inverse frequency

    # Normalize weights to keep them in a reasonable range
    weights /= weights.sum() * num_classes

    return weights

#________________________________ private area ________________________________#
