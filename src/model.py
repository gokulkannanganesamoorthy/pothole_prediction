import torch
import torch.nn as nn
from torchvision import models

class PotholeRiskModel(nn.Module):
    def __init__(self, metadata_dim=4):
        super(PotholeRiskModel, self).__init__()
        
        # 1. Visual Branch (CNN)
        # Using ResNet18 for efficiency
        # Fix deprecation warning: use weights instead of pretrained
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Remove the classification head
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_dim = 512 # ResNet18 output dim
        
        # 2. Metadata Branch (MLP)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.metadata_out_dim = 32
        
        # 3. Fusion Head
        self.fusion_head = nn.Sequential(
            nn.Linear(self.visual_dim + self.metadata_out_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output probability (0-1)
        )
        
    def forward(self, images, metadata):
        """
        Args:
            images: Tensor (B, 3, 224, 224)
            metadata: Tensor (B, metadata_dim)
        Returns:
            risk_score: Tensor (B, 1) in [0, 1]
        """
        # Visual Path
        img_features = self.visual_encoder(images)
        img_features = img_features.view(img_features.size(0), -1) # Flatten (B, 512)
        
        # Metadata Path
        meta_features = self.metadata_encoder(metadata) # (B, 32)
        
        # Fusion
        combined = torch.cat((img_features, meta_features), dim=1)
        risk = self.fusion_head(combined)
        
        return risk
