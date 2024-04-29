import torch
import torch.nn as nn
from torchvision.models import resnet18 as resnet

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

class CombinedModel(nn.Module):
    def __init__(self, mlp, n_classes, train_resnet=False):
        super(CombinedModel, self).__init__()
        self.resnet18 = resnet(weights='IMAGENET1K_V1')
        self.mlp = mlp
        
        # Configure trainability of ResNet layers
        for param in self.resnet18.parameters():
            param.requires_grad = train_resnet
        
        num_features_resnet = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()  # Remove the final fully connected layer
        
        mlp_output_size = mlp.layers[-2].out_features
        combined_input_size = num_features_resnet + mlp_output_size
        
        self.combined = nn.Sequential(
            nn.Linear(combined_input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, image, csv_data):
        x1 = self.resnet18(image)
        x2 = self.mlp(csv_data)
        x = torch.cat((x1, x2), dim=1)
        return self.combined(x)
