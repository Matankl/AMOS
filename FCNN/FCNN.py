import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.transforms.functional import pil_to_tensor
import matplotlib.pyplot as plt
from FCNN_Architecture import FullyConnectedNN
import seaborn as sns

TRAIN = False

# Define the dataset class
class CTScanDataset(Dataset):
    def __init__(self, input_dir, label_dir, input_transform=None, label_transform=None):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.file_names = os.listdir(input_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.file_names[idx])
        label_path = os.path.join(self.label_dir, self.file_names[idx])

        input_image = Image.open(input_path).convert("L")  # Grayscale input
        label_image = Image.open(label_path)  # Label image

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.label_transform:
            label_image = self.label_transform(label_image)

        return input_image, label_image

def iou_metric(predictions, labels, num_classes, ignore_background=True):
    ious = []
    for cls in range(1 if ignore_background else 0, num_classes):  # Skip background class 0
        intersection = ((predictions == cls) & (labels == cls)).sum().item()
        union = ((predictions == cls) | (labels == cls)).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def visualize_predictions(inputs, labels, predictions, num_images=4, save=False):
    """
    Visualizes input images, ground truth labels, and predicted labels side by side.
    """
    inputs = inputs.cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()

    for i in range(min(num_images, inputs.shape[0])):
        input_img = inputs[i, 0, :, :]  # Assumes grayscale input
        label_img = labels[i]
        pred_img = predictions[i]

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(input_img, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Label")
        plt.imshow(label_img, cmap="jet")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Label")
        plt.imshow(pred_img, cmap="jet")
        plt.axis("off")

        if save:
            os.makedirs("results", exist_ok=True)
            j = 0
            while True:
                if os.path.exists(f"results/plot_{j}.png"):
                    j += 1
                else:
                    break
            plt.savefig(f"results/plot_{j}.png")
        else:
            plt.show()


if __name__ == '__main__':
    # Hyperparameters
    input_size = 220
    num_classes = 16  # Adjust according to your dataset
    batch_size = 16
    epochs = 10
    learning_rate = 0.001

    # Transforms for the input images
    input_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # Resize to the input size
        transforms.ToTensor(),  # Convert to tensor
    ])

    label_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=Image.NEAREST),
        lambda img: pil_to_tensor(img).squeeze(0).long(),  # Convert to tensor without scaling
    ])


    amos_path = r"D:\Database\Images\amos22 (1)\amos22"

    # Load datasets
    train_dataset = CTScanDataset(
        f"{amos_path}/Train/input",
        f"{amos_path}/Train/label",
        input_transform=input_transform,
        label_transform=label_transform,
        )

    test_dataset = CTScanDataset(
        f"{amos_path}/Test/input",
        f"{amos_path}/Test/label",
        input_transform=input_transform,
        label_transform=label_transform,
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print(f"Running on {device.type}. BEWARE !!!")

    model = FullyConnectedNN(input_size, num_classes).to(device)

    # weighted loss function
    def compute_class_weights(dataset):
        total_pixels = 0
        class_counts = torch.zeros(num_classes)
        for _, label in tqdm(dataset):
            class_counts += torch.bincount(label.flatten(), minlength=num_classes)
            total_pixels += label.numel()
        return total_pixels / (num_classes * class_counts)

    if TRAIN:
        print(f"Computing class occurrence for weighted loss.")
        class_weights = compute_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


        best_val_loss = float('inf')
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):
                inputs, labels = batch  # Load batch by batch
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)  # outputs: [batch_size, num_classes, input_size, input_size]

                # Validate output and label shapes
                if outputs.size(2) != labels.size(1) or outputs.size(3) != labels.size(2):
                    raise ValueError(
                        f"Mismatch in output and label shapes: outputs.shape={outputs.shape}, labels.shape={labels.shape}")

                # Compute loss
                loss = criterion(outputs, labels)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {train_loss:.4f}")

            # Validation Loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Validating"):
                    inputs, labels = batch
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_loss /= len(test_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_model.pth")
                print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss Improved: {val_loss:.4f}. Model Saved!")

            print(f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}")

            #saving losses for plotting
            train_losses.append(train_loss)
            val_losses.append(val_loss)


    # Recreate the model architecture
    model = FullyConnectedNN(input_size=input_size, num_classes=num_classes)

    # Load the state dictionary
    model.load_state_dict(torch.load("best_model.pth"))
    model.to(device)

    ### visualizing predictions and computing metrics ###
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    total_foreground, correct_foreground = 0, 0
    iou_scores = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, labels = batch  # Load batch by batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Ignore background pixels (class 0)
            foreground_mask = labels > 0  # Mask for foreground classes

            correct_foreground += ((predicted == labels) & foreground_mask).sum().item()
            total_foreground += foreground_mask.sum().item()

            # Compute IoU only for foreground classes
            iou_scores.append(iou_metric(predicted, labels, num_classes=num_classes, ignore_background=True))

            # Update confusion matrix
            for true_class in range(num_classes):
                for pred_class in range(num_classes):
                    conf_matrix[true_class, pred_class] += (
                                (labels == true_class) & (predicted == pred_class)).sum().item()

            # visualize_predictions(inputs, labels, predicted, num_images=4, save=True)
            # break  # Only visualize one batch

    # Compute foreground accuracy
    foreground_accuracy = 100 * correct_foreground / total_foreground
    print(f"Foreground Test Accuracy: {foreground_accuracy:.2f}%")

    # Compute mean IoU without background
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU (Foreground Classes): {mean_iou:.4f}")

    conf_matrix_no_bg = conf_matrix[1:, 1:]  # Exclude row and column for class 0

    # Normalize per class (excluding background)
    conf_matrix_normalized = conf_matrix_no_bg.astype("float") / conf_matrix_no_bg.sum(axis=1, keepdims=True)

    # Plot confusion matrix without background class
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=range(1, num_classes), yticklabels=range(1, num_classes))

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Excluding Background)")
    plt.show()