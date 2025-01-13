import torch

# global var
C = 3                  # number of channels to give the model
batch_size = 16
EPOCHS = 5
LEARNING_RATE = 1e-2
CHANNELS = [C, 64, 128, 256, 512, 1024]

LOAD_CHECKPOINT = False
DEBUGMODE = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_path = './model.pt'
DATA_SET_FOLDER = '/home/or/Desktop/DataSets/AMOS/'
Train_image_dir = 'amos22/Train/input'
Train_label_dir = 'amos22/Train/label'
Val_image_dir = 'amos22/Validation/input'
Val_label_dir = 'amos22/Validation/label'
TEMP_FOLSER = "TEMP"