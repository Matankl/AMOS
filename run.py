import torch
from torch.utils.data import DataLoader
import train_methods as tm
from torch.optim import Adam
from tqdm import tqdm
from train_methods import train, validation, EarlyStopping, WeightedBCEWithLogitsLoss, SegmentationDataset
from UNet import Unet
import json
import os
import nibabel as nib
import numpy as np

C = 16
epochs = 1000
learning_rate = 1e-2
act_batch_size = 1  # Can't fit more than one image in GPU!
eff_batch_size = 1  # Efective batch (Gradient accumulation)
momentum = 0.99
device = 'cuda'
channels = [C, 64, 128, 256, 512, 1024]
w0 = 10
sigma = 5
model_path = './model.pt'
Test_image_dir = 'imagesTr'
Test_label_dir = 'labelsTr'

# Early stopping
es = tm.EarlyStopping(patience=100, fname=model_path)

imagesTR_len = len(os.listdir("/home/or/PycharmProjects/AMOS"))
# lab = len(os.listdir(label_dir))

# Make progress bars
pbar_epoch = tqdm(total=epochs, unit='epoch', position=0, leave=False)
pbar_train = tqdm(total=len(imagesTR_len), unit='batch', position=1, leave=False)

# Make model
model = Unet(channels=channels, no_classes=1).double().to(device)

# Make adam optimiser
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,  # Learning rate
    betas=(0.9, 0.999),  # Coefficients used for computing running averages of gradient and its square
    eps=1e-08,  # Term added to denominator to improve numerical stability
    weight_decay=0  # Weight decay (L2 penalty)
)


# Make loss
criterion = tm.WeightedBCEWithLogitsLoss(batch_size=act_batch_size)

# Load checkpoint (if it exists)
cur_epoch = 0
if os.path.isfile(model_path):
    checkpoint = torch.load(model_path)
    cur_epoch = checkpoint['epoch']
    es.best_loss = checkpoint['loss']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Hold stats for training process
stats = {'epoch': [], 'train_loss': [], 'val_loss': []}

# Training  / validation loop
for epoch in range(cur_epoch, epochs):

    # Train / validate
    pbar_epoch.set_description_str(f'Epoch {epoch + 1}')
    train_loss = tm.train(model, optimizer, Test_image_dir, Test_label_dir, criterion, device, pbar_train)
    val_loss = tm.validation(model, Test_image_dir, Test_label_dir, criterion, device)


    # Append stats
    stats['epoch'].append(epoch)
    stats['train_loss'].append(train_loss)
    stats['val_loss'].append(val_loss)

    # Early stopping (just saves model if validation loss decreases when: pass)
    if es(epoch, val_loss, optimizer, model): pass

    # Update progress bars
    pbar_epoch.set_postfix(train_loss=train_loss, val_loss=val_loss)
    pbar_epoch.update(1)
    pbar_train.reset()