import torch.nn as nn


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