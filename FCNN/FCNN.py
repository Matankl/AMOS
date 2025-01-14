import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Define the dataset class
class CTScanDataset(Dataset):
    def __init__(self, input_dir, label_dir, transform=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.transform = transform
        self.file_names = os.listdir(input_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.file_names[idx])
        label_path = os.path.join(self.label_dir, self.file_names[idx])

        input_image = Image.open(input_path).convert("L")  # Grayscale input
        label_image = Image.open(label_path)  # Label image

        if self.transform:
            input_image = self.transform(input_image)
            label_image = torch.tensor(np.array(label_image), dtype=torch.long)

        return input_image, label_image

# Fully Connected Neural Network for segmentation
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FullyConnectedNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, input_size * input_size * num_classes)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.num_classes, self.input_size, self.input_size)
        return x

# Hyperparameters
input_size = 640
num_classes = 16  # Adjust according to your dataset
batch_size = 2
epochs = 10
learning_rate = 0.001

# Transforms for the input images
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load datasets
train_dataset = CTScanDataset("../amos22/Train/input", "../amos22/Train/label", transform=transform)
test_dataset = CTScanDataset("../amos22/Test/input", "../amos22/Test/label", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FullyConnectedNN(input_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)


        # Forward pass
        outputs = model(inputs)
        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)  # Reshape for loss calculation
        labels = torch.where(labels < num_classes, labels, torch.tensor(-1, device=labels.device))

        print(f"{outputs.shape=}")
        print(f"{labels.shape=}")

        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Evaluation
model.eval()
total, correct = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        total += labels.numel()
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# import os
# import torch
# from torchvision import transforms
# from PIL import Image
# import numpy as np
#
#
# # Load images, classify, and save output labels
#
# def process_images(input_folder, output_folder, model, device, input_size):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
#
#     file_names = sorted(os.listdir(input_folder))
#
#     model.eval()
#
#     with torch.no_grad():
#         for idx, file_name in enumerate(file_names):
#             input_path = os.path.join(input_folder, file_name)
#             output_path = os.path.join(output_folder, f"{idx:03d}.png")
#
#             input_image = Image.open(input_path).convert("L")
#             input_tensor = transform(input_image).unsqueeze(0).to(device)
#
#             # Run the classification
#             output = model(input_tensor)
#             predicted = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
#
#             # Save the label as an image
#             label_image = Image.fromarray(predicted.astype(np.uint8))
#             label_image.save(output_path)
#
#
# # Define paths
# input_folder = "path/to/input/folder"
# output_folder = "path/to/output/folder"
#
# # Load the trained model
# model = FullyConnectedNN(input_size=640, num_classes=3)  # Adjust num_classes
# model.load_state_dict(torch.load("path/to/model.pth"))  # Replace with your model path
# model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#
# # Process images
# process_images(input_folder, output_folder, model, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#                input_size=640)
