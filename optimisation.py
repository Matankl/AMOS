import optuna
import train_methods as tm
from UNet import Unet
import os
import torch.nn as nn
from const import *

print("Setting up weights for the classes")
train_dataset = tm.ImageDataset(os.path.join(DATA_SET_FOLDER, train_mid_in), os.path.join(DATA_SET_FOLDER, train_mid_l), transform=True)
# class_weights = tm.compute_class_weights(train_dataset, 16)


import torch
import torch.nn.functional as F


class MultiClassLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, num_classes=3):
        """
        Multi-class loss combining Cross-Entropy Loss and Dice Loss.

        :param alpha: Weight for Cross-Entropy Loss (default 0.5).
        :param beta: Weight for Dice Loss (default 0.5).
        :param smooth: Smoothing factor for Dice Loss.
        :param num_classes: Number of classes in segmentation task.
        """
        super(MultiClassLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes

    def dice_loss(self, pred, target):
        """
        Compute multi-class Dice Loss.
        :param pred: Model predictions (softmax probabilities).
        :param target: One-hot encoded ground truth.
        """
        dice_loss = 0.0
        for c in range(self.num_classes):
            pred_c = pred[:, c]  # Extract class channel
            target_c = target[:, c]
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice_c = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice_c  # Loss = 1 - Dice coefficient

        return dice_loss / self.num_classes  # Average over classes

    def forward(self, pred, target):
        """
        Compute the combined loss.
        :param pred: Model predictions (logits).
        :param target: Ground truth labels (as class indices, not one-hot).
        """
        # Convert target to one-hot encoding
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Apply softmax to predictions (multi-class)
        pred = F.softmax(pred, dim=1)  # Ensure predictions sum to 1

        # Cross-Entropy Loss (multi-class)
        ce_loss = F.cross_entropy(pred, target)

        # Dice Loss
        dice = self.dice_loss(pred, target_one_hot)

        # Combined Loss
        loss = self.alpha * ce_loss + self.beta * dice
        return loss


# criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
criterion = MultiClassLoss(alpha=0.5, beta=0.5, num_classes=16)


class Patience:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_value = float('inf')  # Assuming minimization; use `-inf` for maximization
        self.counter = 0

    def check(self, value):
        if value < self.best_value:  # Improvement condition
            self.best_value = value
            self.counter = 0  # Reset patience
        else:
            self.counter += 1  # Increase counter if no improvement

        return self.counter >= self.patience  # Stop if patience runs out

# Initialize storage for the best model
best_min_loss = float("inf")
best_model_state = None

# Define the objective function for Optuna
def objective(trial):
    global best_min_loss, best_model_state  # Track best model globally

    torch.cuda.empty_cache()  # Releases unused memory
    torch.cuda.ipc_collect()  # Collects any unused memory from inter-process communication

    # Reset patience for each trial
    patience = Patience(patience=3)

    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('lr', 5e-5, 0.01)
    batch_size = trial.suggest_categorical('batch_size', [2, 4])

    #loss optimization
    alpha = trial.suggest_float("alpha", 0.25, 0.6)
    beta = 1 - alpha
    criterion = MultiClassLoss(alpha=alpha, beta=beta, num_classes=16)

    # Create the model
    model = Unet(channels=CHANNELS, no_classes=NUMBER_OF_CLASSES, output_size=(640, 640)).to(DEVICE).float()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay = WEIGHT_DECAY
    )

    # Data paths
    train_image_dir = os.path.join(DATA_SET_FOLDER, train_mid_in)
    train_label_dir = os.path.join(DATA_SET_FOLDER, train_mid_l)
    val_image_dir = os.path.join(DATA_SET_FOLDER, val_mid_in)
    val_label_dir = os.path.join(DATA_SET_FOLDER, val_mid_l)

    # Training and validation loop
    num_epochs = 30  # Number of epochs for each trial
    print('learning rate:', learning_rate, 'weight_decay:', WEIGHT_DECAY, 'batch_size:', batch_size)
    for epoch in range(num_epochs):
        train_loss, train_iou, train_f1, train_accuracy = tm.train(model, optimizer, batch_size, train_image_dir, train_label_dir, criterion, DEVICE)
        val_loss, avg_iou, avg_f1, avg_accuracy = tm.validate(model, batch_size, val_image_dir, val_label_dir, criterion, DEVICE)

        # log on console
        print(f"Train loss: {train_loss}, Train IoU: {train_iou}, Train F1: {train_f1}, Accuracy: {train_accuracy}")
        print(f"Val loss: {val_loss}, Val IoU: {avg_iou}, Val F1: {avg_f1}, Accuracy: {avg_accuracy}")

        # Update best model if loss is improved
        if val_loss < best_min_loss:
            best_min_loss = val_loss
            best_model_state = model.state_dict()  # Save model parameters
            torch.save(best_model_state, f"best_model_trial_{trial.number}_loss_{val_loss}.pth")
            print("best_model saved")

        # Patience stop
        if patience.check(val_loss):
            # print("Stopping early due to lack of improvement.")
            break


    # Return validation loss for Optuna to minimize
    return val_loss, avg_iou


# Create the study
study = optuna.create_study(directions=['minimize', 'maximize'])
study.optimize(objective, n_trials = 10)  # Number of trials

# Save the best model based on the minimized objective
if best_model_state:
    torch.save(best_model_state, "best_model.pth")
    print("Best model saved with loss:", best_min_loss)

# Best parameters
print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)
