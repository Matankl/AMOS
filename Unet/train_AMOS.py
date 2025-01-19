import train_methods as tm
from tqdm import tqdm
from UNet import Unet
import os
import torch.nn as nn
from const import *

# Early stopping
early_stopping = tm.EarlyStopping(patience = 3, fname=model_path)


# debug
if DEBUGMODE:
    imagesTR_len = len(os.listdir("/amos22/imagesTr"))
    print(f"Number of files in imagesTr: {imagesTR_len}")


# Make model
AMOS_NET = Unet(channels=CHANNELS, no_classes=NUMBER_OF_CLASSES, output_size= (640, 640)).double().to(DEVICE)
AMOS_NET = AMOS_NET.to(DEVICE).float()


# Make adam optimiser
optimizer = torch.optim.Adam(
    AMOS_NET.parameters(),
    lr = LEARNING_RATE,  # Learning rate
    betas = (0.9, 0.999),  # Coefficients used for computing running averages of gradient and its square
    eps = 1e-08,  # Term added to denominator to improve numerical stability
    weight_decay = WEIGHT_DECAY # Weight decay (L2 penalty)
    )




# Load checkpoint (if it exists)
cur_epoch = 0
if LOAD_CHECKPOINT:
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        cur_epoch = checkpoint['epoch']
        early_stopping.best_loss = checkpoint['best_loss']
        # AMOS_NET.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Hold stats for training process
stats = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'train_f1': [], 'val_f1': [], 'train_acc': [], 'val_acc': []}


# train_dataset = tm.ImageDataset(os.path.join(DATA_SET_FOLDER, train_mid_in), os.path.join(DATA_SET_FOLDER, train_mid_l), transform=True)

# Make loss
# class_weights = tm.compute_class_weights(train_dataset)
# criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
criterion = nn.CrossEntropyLoss()

# Training  / validation loop
for epoch in tqdm(range(cur_epoch, EPOCHS)):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    # Train + validate
    train_loss, train_iou, train_f1, train_accuracy = tm.train(
        AMOS_NET, optimizer, BATCH_SIZE,
        os.path.join(DATA_SET_FOLDER, train_mid_in),
        os.path.join(DATA_SET_FOLDER, train_mid_l),
        criterion, DEVICE
    )
    print(f"Train loss: {train_loss}, Train IoU: {train_iou}, Train F1: {train_f1}, Accuracy: {train_accuracy}")

    val_loss, val_iou, val_f1, val_accuracy = tm.validate(
        AMOS_NET, BATCH_SIZE,
        os.path.join(DATA_SET_FOLDER, val_mid_in),
        os.path.join(DATA_SET_FOLDER, val_mid_l),
        criterion, DEVICE
    )
    print(f"Validation loss: {val_loss}, Validation IoU: {val_iou}, Validation F1: {val_f1}, Accuracy: {val_accuracy}")

    if epoch > 0:
        if val_loss < stats["val_loss"][-1]:
            # Save only the state_dict
            torch.save({'model_state_dict': AMOS_NET.state_dict(), 'epoch': epoch }, model_path)
    else:
        torch.save({'model_state_dict': AMOS_NET.state_dict(), 'epoch': epoch}, model_path)

    # Append stats
    stats['epoch'].append(epoch)
    stats['train_loss'].append(train_loss)
    stats['val_loss'].append(val_loss)
    stats['train_iou'].append(train_iou)
    stats['val_iou'].append(val_iou)
    stats['train_f1'].append(train_f1)
    stats['val_f1'].append(val_f1)
    stats['train_acc'].append(train_accuracy)
    stats['val_acc'].append(val_accuracy)

    # Early stopping
    if early_stopping(epoch, val_loss, optimizer, AMOS_NET):
        pass



# # load model for the testing part
# AMOS_NET = Unet(channels = CHANNELS, no_classes = 1).double().to(DEVICE)
# checkpoint = torch.load(model_path)
# AMOS_NET.load_state_dict(checkpoint['model_state_dict'])
# AMOS_NET.eval()
# # test_loss = tm.validate(AMOS_NET, batch_size, os.path.join(DATA_SET_FOLDER, ), os.path.join(DATA_SET_FOLDER, val_mid_l), criterion, DEVICE)
# out = AMOS_NET()

