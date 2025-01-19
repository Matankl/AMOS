import torch

# global var
BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 7.890813261743468e-05
WEIGHT_DECAY = 0.0011571908305315678
CHANNELS = [1, 64, 128, 256, 512, 1024]
NUMBER_OF_CLASSES = 16

LOAD_CHECKPOINT = False
DEBUGMODE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# saved model path
model_path = '../Fixed Models/model_with_weighted_lossbatch2_lr_7.890813261743468e-05_10epoch.pth'

DATA_SET_FOLDER = '/home/or/Desktop/DataSets/AMOS'

Train_image_dir = 'amos22/Train/input'
Train_label_dir = 'amos22/Train/label'

Val_image_dir = 'amos22/Validation/input'
Val_label_dir = 'amos22/Validation/label'

Test_image_dir = 'amos22/Test/input'
Test_label_dir = 'amos22/Test/label'

train_mid_in = 'amos22/Train/mid input'
train_mid_l = 'amos22/Train/mid label'
val_mid_in = 'amos22/Validation/mid input'
val_mid_l = 'amos22/Validation/mid label'

train_mini_input = 'amos22/Train/mini input'
train_mini_label = 'amos22/Train/mini label'
val_mini_input = 'amos22/Validation/mini input'
val_mini_label = 'amos22/Validation/mini label'