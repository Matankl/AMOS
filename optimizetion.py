import optuna
import train_methods as tm
from UNet import Unet
import os
import torch.nn as nn
from const import *


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [2,6,8])

    # Create the model
    model = Unet(channels=CHANNELS, no_classes=NUMBER_OF_CLASSES, output_size=(640, 640)).to(DEVICE).float()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Data paths
    train_image_dir = os.path.join(DATA_SET_FOLDER, train_mid_in)
    train_label_dir = os.path.join(DATA_SET_FOLDER, train_mid_l)
    val_image_dir = os.path.join(DATA_SET_FOLDER, val_mid_in)
    val_label_dir = os.path.join(DATA_SET_FOLDER, val_mid_l)

    # Training and validation loop
    num_epochs = 3  # Number of epochs for each trial
    for epoch in range(num_epochs):
        print('learning rate:', learning_rate, 'weight_decay:', weight_decay, 'batch_size:', batch_size)
        train_loss = tm.train(model, optimizer, batch_size, train_image_dir, train_label_dir, criterion, DEVICE)
        val_loss = tm.validate(model, batch_size, val_image_dir, val_label_dir, criterion, DEVICE)
        print(train_loss, val_loss,)

    # Return validation loss for Optuna to minimize
    return val_loss


# Create the study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 10)  # Number of trials

# Best parameters
print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)
