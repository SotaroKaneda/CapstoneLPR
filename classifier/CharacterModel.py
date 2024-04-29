import torch.nn as nn
import torchvision


class CharacterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        self.num_filters = self.model.fc.in_features  # 2048 for resnet18
        self.fc = nn.Sequential(
            nn.Linear(self.num_filters, 512),        # New fully connected layer with 512 neurons
            nn.ReLU(),
            nn.BatchNorm1d(512),             # Batch normalization
            nn.Dropout(0.5),                  # Dropout with a dropout rate of 0.5
            nn.Linear(512, 35)      # Output layer
        )
        # self.fc = nn.Linear(self.num_filters, 35)
        self.model.fc = self.fc
    
    def forward(self, x):
        return self.model(x)