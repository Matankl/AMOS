import torch

# global var
batch_size = 8
EPOCHS = 3
LEARNING_RATE = 1e-2
CHANNELS = [1, 64, 128, 256, 512, 1024]
NUMBER_OF_CLASSES = 16

LOAD_CHECKPOINT = False
DEBUGMODE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# saved model path
model_path = './model.pt'

DATA_SET_FOLDER = '/home/or/Desktop/DataSets/AMOS/'

Train_image_dir = 'amos22/Train/input'
Train_label_dir = 'amos22/Train/label'

Val_image_dir = 'amos22/Validation/input'
Val_label_dir = 'amos22/Validation/label'



train_mid_in = 'amos22/Train/mid input'
train_mid_l = 'amos22/Train/mid label'
val_mid_in = 'amos22/Validation/mid input'
val_mid_l = 'amos22/Validation/mid label'