import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights
from torchvision.models import EfficientNet_B0_Weights

# Define Pytorch CNN Model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        
        # Convolutional Blocks
        self.features=nn.Sequential(
            # Block 1: Input 3 channels, Output 32 channels
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # Block 2: Input 32 channels, Ouput 64 channels
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            # Block 3: Input 64 channels, Ouput 128 channels
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            
            nn.Dropout2d(p=0.2)
        )
        
        # Dense Layer
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )
        
    def forward(self,x):
        x=self.features(x)
        x=self.classifier(x)
        return x

class TransferVGG(nn.Module):
    def __init__(self, num_classes):
        super(TransferVGG, self).__init__()
        
        weights = VGG16_Weights.IMAGENET1K_V1
        self.backbone = models.vgg16(weights=weights) 
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        num_ftrs = self.backbone.classifier[0].in_features 
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 4096),   
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    
    # Frozen Training - Unfreeze the model
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=True

class TransferEfficient(nn.Module):
    def __init__(self, num_classes):
        super(TransferEfficient, self).__init__()
        
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.backbone = models.efficientnet_b0(weights=weights) 
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        num_ftrs = self.backbone.classifier[1].in_features 
        
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),   
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
    
    # Frozen Training - Unfreeze the model
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad=True