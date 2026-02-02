import torch
import cv2
import numpy as np
from src.model import PotholeRiskModel
from src.data_generator import SyntheticDataGenerator
from torchvision import transforms
from PIL import Image

class PotholePredictor:
    def __init__(self, model_path='models/pothole_risk_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Model
        self.model = PotholeRiskModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.generator = SyntheticDataGenerator()

    def predict(self, image_path, weather='Sunny', traffic='Low', temperature=25):
        """
        Predicts pothole formation risk.
        Args:
            image_path (str): Path to image.
            weather (str): 'Sunny', 'Rainy', 'Snowy'.
            traffic (str): 'Low', 'Medium', 'High'.
            temperature (float): Celsius.
        """
        # 1. Process Image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device) # Add batch dim
        
        # 2. Process Metadata
        # Normalize inputs
        rain_index = 1.0 if weather in ['Rainy', 'Snowy'] else 0.0
        freeze_index = 1.0 if temperature < 0 else 0.0
        
        traffic_map = {'Low': 0.0, 'Medium': 0.5, 'High': 1.0}
        traffic_val = traffic_map.get(traffic, 0.0)
        
        temp_norm = (temperature + 10) / 50.0
        
        meta_vector = torch.tensor([[
            temp_norm,
            rain_index,
            freeze_index,
            traffic_val
        ]], dtype=torch.float32).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            risk_score = self.model(img_tensor, meta_vector).item()
            
        return risk_score

    def estimate_weather(self, image_path):
        """
        Estimates weather condition from image statistics.
        """
        img = cv2.imread(image_path)
        if img is None:
            return 'Sunny'
            
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness = hsv[:,:,2].mean()
        saturation = hsv[:,:,1].mean()
        
        # Heuristics
        # High Brightness + Low Saturation -> Snowy
        if brightness > 180 and saturation < 40:
            return 'Snowy'
        # Low Brightness -> Rainy/Overcast
        elif brightness < 100:
            return 'Rainy'
        else:
            return 'Sunny'

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Pothole Formation Risk")
    parser.add_argument('--image', type=str, help='Path to the road image')
    parser.add_argument('--weather', type=str, default=None, choices=['Sunny', 'Rainy', 'Snowy'], help='Weather condition (detected if not provided)')
    parser.add_argument('--traffic', type=str, default='Low', choices=['Low', 'Medium', 'High'], help='Traffic load')
    parser.add_argument('--temp', type=float, default=25.0, help='Temperature in Celsius')
    
    args = parser.parse_args()
    
    predictor = PotholePredictor()
    
    # If image provided, use it
    if args.image:
        try:
            # Auto-detect weather if not provided
            weather = args.weather
            if weather is None:
                weather = predictor.estimate_weather(args.image)
                print(f"Auto-Detected Weather: {weather}")
            
            risk = predictor.predict(args.image, weather=weather, traffic=args.traffic, temperature=args.temp)
            print(f"\n--- Prediction Results ---")
            print(f"Image: {args.image}")
            print(f"Conditions: {weather}, {args.traffic} Traffic, {args.temp}°C")
            print(f"Predicted Risk Score: {risk:.4f}")
            # Calibrated Thresholds: Model output is conservative, typically < 0.25 even for bad roads.
            # > 0.4 is Critical, > 0.15 is High, > 0.05 is Moderate.
            print(f"Risk Level: {'CRITICAL' if risk > 0.1 else 'HIGH' if risk > 0.07 else 'MODERATE' if risk > 0.05 else 'LOW'}")
            print("--------------------------\n")
        except Exception as e:
            print(f"Error: {e}")
            
    # Default behavior (demo) if no image provided
    else:
        print("No image provided using --image. Running demo on random test image...")
        # Pick a random test image
        test_dir = 'data/raw/China_MotorBike/test/images' if os.path.exists('data/raw/China_MotorBike/test/images') else 'data/raw/China_MotorBike/train/images'
        
        if not os.path.exists(test_dir):
             test_dir = 'data/raw/China_MotorBike/test'

        test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        if test_images:
            sample_img = os.path.join(test_dir, test_images[0])
            print(f"Testing on: {sample_img}")
            
            risk = predictor.predict(sample_img, weather='Rainy', traffic='High', temperature=5)
            print(f"Scenario (Rainy, High, 5°C): Risk = {risk:.4f}")
        else:
            print("No test images found for demo.")
