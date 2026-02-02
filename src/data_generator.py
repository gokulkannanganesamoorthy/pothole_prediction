import random
import numpy as np
import xml.etree.ElementTree as ET
import os

class SyntheticDataGenerator:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        self.weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy']
        self.traffic_conditions = ['Low', 'Medium', 'High']
        
        # Risk weights
        self.weather_weights = {'Sunny': 0.0, 'Cloudy': 0.1, 'Rainy': 0.3, 'Snowy': 0.4}
        self.traffic_weights = {'Low': 0.0, 'Medium': 0.2, 'High': 0.4}
        
    def generate_metadata(self):
        """
        Generates random environmental metadata.
        Returns:
            dict: {
                'temperature': float (Celsius),
                'weater_condition': str,
                'traffic_load': str,
                'rain_index': float (0-1),
                'freeze_index': float (0-1)
            }
        """
        weather = random.choice(self.weather_conditions)
        traffic = random.choice(self.traffic_conditions)
        
        # Correlate temp with weather
        if weather == 'Snowy':
            temp = random.uniform(-10, 2)
        elif weather == 'Sunny':
            temp = random.uniform(20, 40)
        else:
            temp = random.uniform(5, 25)
            
        return {
            'temperature': round(temp, 1),
            'weather_condition': weather,
            'traffic_load': traffic,
            'rain_index': 1.0 if weather in ['Rainy', 'Snowy'] else 0.0,
            'freeze_index': 1.0 if temp < 0 else 0.0
        }

    def parse_annotation(self, xml_path):
        """
        Parses Pascal VOC XML to extract damage types.
        Returns:
            list: List of class names found in the image.
        """
        if not os.path.exists(xml_path):
            return []
            
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        damages = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            damages.append(name)
        return damages

    def calculate_risk_score(self, current_damages, metadata):
        """
        Calculates Risk.
        Aggressive Tuning:
        - Any damage detected -> Base Risk starts at 0.4 (Medium).
        - Specific severe damage -> Base Risk 0.7 (High).
        - Multipliers can easily push it to 0.9+.
        """
        # Default: Perfect road
        base_risk = 0.1 
        
        # Check for ANY damage in the list
        if len(current_damages) > 0:
            # Baseline for "Some Damage"
            base_risk = 0.4 
            
            has_pothole = 'D40' in current_damages
            has_alligator = 'D20' in current_damages
            has_linear = 'D00' in current_damages or 'D10' in current_damages
            
            if has_pothole:
                return 1.0 # Failed
            
            if has_alligator:
                base_risk = 0.7 # High structural failure
            elif has_linear:
                base_risk = 0.5 # Moderate cracking
        
        # Factors (Amplified)
        # Weather: Sunny=0, Cloudy=0.1, Rainy=0.4, Snowy=0.6
        w_map = {'Sunny': 0.0, 'Cloudy': 0.1, 'Rainy': 0.4, 'Snowy': 0.6}
        weather_factor = w_map.get(metadata['weather_condition'], 0.0)
        
        # Traffic: Low=0, Medium=0.3, High=0.6
        t_map = {'Low': 0.0, 'Medium': 0.3, 'High': 0.6}
        traffic_factor = t_map.get(metadata['traffic_load'], 0.0)
        
        freeze_bonus = 0.5 if metadata['freeze_index'] > 0 and metadata['rain_index'] > 0 else 0.0
        
        # Calculation
        # Example: Linear Crack (0.5) + Rain (0.4) + Med Traffic (0.3)
        # Multiplier = 1 + 0.4 + 0.3 = 1.7
        # Total = 0.5 * 1.7 = 0.85 (Critical)
        
        # Example: No Damage (0.1) + Rain (0.4) + Med Traffic (0.3)
        # Total = 0.1 * 1.7 = 0.17 (Low) -> Correct!
        
        multiplier = 1.0 + weather_factor + traffic_factor + freeze_bonus
        
        total_risk = base_risk * multiplier
        return min(max(total_risk, 0.0), 1.0)
