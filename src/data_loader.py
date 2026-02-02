import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from .data_generator import SyntheticDataGenerator
from torchvision import transforms

class RoadDamageDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, seed=42):
        """
        Args:
            root_dir (str): Path to 'China_MotorBike' folder.
            mode (str): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.generator = SyntheticDataGenerator(seed=seed)
        
        # Path setup
        self.image_dir = os.path.join(root_dir, mode, 'images')
        self.annotation_dir = os.path.join(root_dir, mode, 'annotations')
        
        # List all images
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        
        # Define basic transform if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Annotation (XML)
        xml_name = img_name.replace('.jpg', '.xml')
        xml_path = os.path.join(self.annotation_dir, xml_name)
        
        # Generate Metadata & Risk
        # 1. Get ground truth damages from XML
        damages = self.generator.parse_annotation(xml_path) if os.path.exists(xml_path) else []
        
        # 2. Generate random weather/traffic (This simulates "future" or "current" conditions for prediction)
        metadata = self.generator.generate_metadata()
        
        # 3. Calculate Risk Label (The target we want to predict)
        risk_score = self.generator.calculate_risk_score(damages, metadata)
        
        # Prepare Metadata Vector [Temp, Rain, Freeze, Traffic_Low, Traffic_Med, Traffic_High, Weather_Sunny, ...]
        # For simplicity: [Temp_Norm, Rain_Index, Freeze_Index, Traffic_Index]
        # Traffic Index: Low=0, Med=0.5, High=1.0
        
        traffic_map = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}
        traffic_val = traffic_map[metadata['traffic_load']]
        
        # Normalize Temp (assume -10 to 40)
        temp_norm = (metadata['temperature'] + 10) / 50.0 
        
        meta_vector = torch.tensor([
            temp_norm,
            metadata['rain_index'],
            metadata['freeze_index'],
            traffic_val
        ], dtype=torch.float32)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, meta_vector, torch.tensor([risk_score], dtype=torch.float32)
