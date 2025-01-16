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
    iou_scores = []
    f1_scores = []
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
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Convert logits to predicted classes
        labels_np = labels.cpu().numpy()
        iou_scores.append(calculate_iou(preds, labels_np, num_classes=model.head.out_channels))
        f1_scores.append(calculate_f1_score(preds, labels_np, num_classes=model.head.out_channels))

    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)

    return train_loss / len(train_loader) ,avg_iou, avg_f1


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
    iou_scores = []
    f1_scores = []
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
            iou_scores.append(calculate_iou(preds, labels_np, num_classes=model.head.out_channels))
            f1_scores.append(calculate_f1_score(preds, labels_np, num_classes=model.head.out_channels))

    avg_iou = np.mean(iou_scores)
    avg_f1 = np.mean(f1_scores)

    return val_loss / len(val_loader), avg_iou, avg_f1


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
        # print("init the image database")

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

def calculate_iou(pred, target, num_classes):
    """
    Calculate the Intersection over Union (IoU) score.
    Args:
        pred: Predicted labels (batch_size, height, width).
        target: Ground truth labels (batch_size, height, width).
        num_classes: Number of classes.
    Returns:
        Average IoU score over all classes.
    """
    iou_scores = []
    for cls in range(num_classes):
        intersection = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()
        if union == 0:
            iou_scores.append(1.0)  # Treat empty classes as perfect IoU
        else:
            iou_scores.append(intersection / union)
    return np.mean(iou_scores)

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


#________________________________ private area ________________________________#
