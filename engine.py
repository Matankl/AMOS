import torch
from torch.utils.data import Dataset
from utils import weight_map
from torch import nn
import numpy as np


def train(model, optimizer, dataloader, criterion, effective_batch_size, p_bar=None):
    ''' Training function for a single epoch '''

    # Set the model to training mode (activates dropout, batchnorm, etc.)
    model.train()

    # Initialize the optimizer gradients to zero before starting training
    optimizer.zero_grad()

    # Running loss to keep track of total loss over all batches
    running_loss = 0

    # Iterate through each batch in the dataloader
    for batch_id, (X, y, weights) in enumerate(dataloader):

        # If a progress bar is provided, update its description
        if p_bar is not None:
            p_bar.set_description_str(f'Batch {batch_id + 1}')

        # Forward pass: Get predictions from the model
        y_hat = model(X)

        # Compute the loss using the provided criterion
        # Loss is normalized by the effective batch size
        loss = criterion(y, y_hat, weights) / effective_batch_size

        # Add this batch's loss to the running total
        running_loss += loss.item()

        # Backward pass: Compute gradients of the loss with respect to the model parameters
        loss.backward()

        # Perform optimizer step after every effective_batch_size batches or at the end of the dataloader
        if ((batch_id + 1) % effective_batch_size == 0) or ((batch_id + 1) == len(dataloader)):
            optimizer.step()  # Update model weights
            optimizer.zero_grad()  # Reset gradients to zero

        # Update the progress bar with the current loss
        if p_bar is not None:
            p_bar.set_postfix(loss=loss.item())
            p_bar.update(1)

    # Compute the average loss over all batches, scaled by the effective batch size
    running_loss = running_loss / len(dataloader) * effective_batch_size

    return running_loss


def validation(model, dataloader, criterion):
    ''' Validation function to evaluate model performance on validation data '''

    # Set the model to evaluation mode (disables dropout, batchnorm, etc.)
    model.eval()

    # Running loss to track validation loss across all batches
    running_loss = 0

    # Disable gradient calculation for validation (saves memory and speeds up computations)
    with torch.no_grad():
        for X, y, weights in dataloader:
            # Forward pass: Get predictions from the model
            y_hat = model(X)

            # Compute loss
            loss = criterion(y, y_hat, weights)

            # Accumulate the loss
            running_loss += loss.item()

    # Compute the average loss over all validation batches
    running_loss /= len(dataloader)

    return running_loss


class EarlyStopping(object):
    ''' Implements early stopping to prevent overfitting during training '''

    def __init__(self, patience, fname):
        '''
        Args:
            patience: Number of epochs to wait before stopping if no improvement
            fname: File name to save the best model
        '''
        self.patience = patience
        self.best_loss = np.Inf  # Initialize the best loss to infinity
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


class WeightedBCEWithLogitsLoss(nn.Module):
    ''' Implements a pixel-wise weighted Binary Cross Entropy with Logits Loss '''

    def __init__(self, batch_size):
        '''
        Args:
            batch_size: Number of samples in each batch
        '''
        super().__init__()
        self.batch_size = batch_size
        # BCEWithLogitsLoss combines sigmoid activation with BCE loss
        self.unw_loss = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, true, predicted, weights):
        '''
        Args:
            true: Ground truth labels
            predicted: Predicted logits
            weights: Weight map to apply to each pixel
        Returns:
            Weighted loss
        '''

        # Compute unweighted loss for each pixel
        loss = self.unw_loss(predicted, true) * weights

        # Sum the loss across all channels
        loss = loss.sum(dim=1)

        # Normalize loss by dividing by weights and reshape
        loss = loss.view(self.batch_size, -1) / weights.view(self.batch_size, -1)

        # Compute the mean loss across the batch
        loss = loss.mean()

        return loss


class SegmentationDataset(Dataset):
    ''' Dataset class for segmentation tasks '''

    def __init__(self, images, masks, wmap_w0, wmap_sigma, device, transform=None):
        '''
        Args:
            images: Input images
            masks: Ground truth segmentation masks
            wmap_w0: Weight map parameter for object importance
            wmap_sigma: Weight map parameter for boundary precision
            device: Device to load the data onto (CPU or GPU)
            transform: Data augmentation transformations
        '''
        self.images = images
        self.masks = masks
        self.transform = transform
        self.device = device

        # Parameters for weight map calculation
        self.w0 = wmap_w0
        self.sigma = wmap_sigma

    def __len__(self):
        ''' Returns the number of samples in the dataset '''
        return len(self.images)

    def __getitem__(self, idx):
        '''
        Preprocess and return an image, its mask, and its weight map
        '''
        # Get the image and mask at the given index
        image = self.images[idx, :, :]
        mask = self.masks[idx, :, :]

        if self.transform:
            # Apply data augmentations if any are provided
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        # Compute weight map for the mask
        weights = weight_map(mask=mask, w0=self.w0, sigma=self.sigma)

        # Normalize image and mask to the range [0, 1]
        image = self.min_max_scale(image, min_val=0, max_val=1)
        mask = self.min_max_scale(mask, min_val=0, max_val=1)

        # Add channel dimensions to image, mask, and weight map
        image = np.expand_dims(image, axis=0)
        weights = np.expand_dims(weights, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # Convert the data to tensors and move to the specified device
        weights = torch.from_numpy(weights).double().to(self.device)
        image = torch.from_numpy(image).double().to(self.device)
        mask = torch.from_numpy(mask).double().to(self.device)

        # Center crop mask and weights (negative padding = cropping)
        mask = nn.ZeroPad2d(-94)(mask)
        weights = nn.ZeroPad2d(-94)(weights)

        return image, mask, weights

    @staticmethod
    def min_max_scale(image, max_val, min_val):
        ''' Normalize an image to the range [min_val, max_val] '''
        image_new = (image - np.min(image)) * (max_val - min_val) / (np.max(image) - np.min(image)) + min_val
        return image_new
