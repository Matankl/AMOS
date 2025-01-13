import train_methods as tm
from tqdm import tqdm
from UNet import Unet
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from const import *

# i dont thing there is a use for them but lets keep it for now
# act_batch_size = 1
# momentum = 0.99
# w0 = 10
# sigma = 5


# Early stopping
early_stopping = tm.EarlyStopping(patience=100, fname=model_path)


# debug
if DEBUGMODE:
    imagesTR_len = len(os.listdir("/amos22/imagesTr"))
    print(f"Number of files in imagesTr: {imagesTR_len}")


# Make progress bars
# pbar_epoch = tqdm(total=epochs, unit='epoch', position=0, leave=False)
# pbar_train = tqdm(total=imagesTR_len, unit='batch', position=1, leave=False)

# Make model
AMOS_NET = Unet(channels=CHANNELS, no_classes=16).double().to(DEVICE)
AMOS_NET = AMOS_NET.to(DEVICE).float()


# Make adam optimiser
optimizer = torch.optim.Adam(
    AMOS_NET.parameters(),
    lr = LEARNING_RATE,  # Learning rate
    betas=(0.9, 0.999),  # Coefficients used for computing running averages of gradient and its square
    eps=1e-08,  # Term added to denominator to improve numerical stability
    weight_decay=0  # Weight decay (L2 penalty)
)

# Make loss
criterion = torch.nn.CrossEntropyLoss()


# Load checkpoint (if it exists)
cur_epoch = 0
if LOAD_CHECKPOINT:
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        cur_epoch = checkpoint['epoch']
        early_stopping.best_loss = checkpoint['loss']
        AMOS_NET.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Hold stats for training process
stats = {'epoch': [], 'train_loss': [], 'val_loss': []}

# Training  / validation loop
for epoch in range(cur_epoch, EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    # Train + validate
    train_loss = tm.train(AMOS_NET, optimizer, batch_size, os.path.join(DATA_SET_FOLDER, Train_image_dir), os.path.join(DATA_SET_FOLDER, Train_label_dir), criterion, DEVICE)
    print(f"Train loss: {train_loss}")
    print(f"end of train epoch{epoch}")

    val_loss = tm.validate(AMOS_NET, batch_size, TEMP_FOLSER, TEMP_FOLSER, criterion, DEVICE)
    print(f"end of val epoch{epoch}")


    # Append stats
    stats['epoch'].append(epoch)
    stats['train_loss'].append(train_loss)
    stats['val_loss'].append(val_loss)

    # Early stopping (just saves model if validation loss decreases when: pass)
    if early_stopping(epoch, val_loss, optimizer, AMOS_NET): pass




# load model for the testing part
AMOS_NET      = Unet(channels = CHANNELS, no_classes = 1).double().to(DEVICE)
checkpoint = torch.load(model_path)
AMOS_NET.load_state_dict(checkpoint['model_state_dict'])
AMOS_NET.eval()

# # Make loss
# criterion = torch.CrossEntropyLoss()
# with torch.no_grad():
#     for  (X, y) in enumerate(test_loader):
#
#         # Forward
#         y_hat = AMOS_NET(X)
#         y_hat = torch.sigmoid(y_hat)
#
#
#         # Convert to numpy
#         X = np.squeeze(X.cpu().numpy())
#         y = np.squeeze(y.cpu().numpy())
#         y_hat = np.squeeze(y_hat.detach().cpu().numpy())
#
#         # Make mask
#         y_hat2 = y_hat > 0.5
