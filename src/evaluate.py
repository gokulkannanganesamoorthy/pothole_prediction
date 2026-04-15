import torch
from torch.utils.data import DataLoader
from src.data_loader import RoadDamageDataset
from src.model import PotholeRiskModel
from src.utils import plot_confusion_matrix_sns, get_risk_level
import os
from tqdm import tqdm

def evaluate_model(model_path='models/pothole_risk_model.pth', batch_size=16):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Dataset (Annotated Validation Split)
    data_path = 'data/raw/China_MotorBike'
    print("Loading annotated training dataset for validation split...")
    
    full_dataset = RoadDamageDataset(root_dir=data_path, mode='train')
    if len(full_dataset) == 0:
        print(f"ERROR: No annotated images found in {data_path}/train/images")
        return
        
    # Consistency in split matters: we use the same 80/20 split as in train.py
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Evaluating on {len(val_dataset)} annotated samples.")
    
    # 2. Initialize and Load Model
    model = PotholeRiskModel().to(device)
    if not os.path.exists(model_path):
        print(f"ERROR: Model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Collect Predictions
    y_true = []
    y_pred = []
    
    print("Running inference...")
    with torch.no_grad():
        for images, metadata, labels in tqdm(loader):
            images = images.to(device)
            metadata = metadata.to(device)
            
            outputs = model(images, metadata)
            
            # Convert to CPU for processing
            outputs = outputs.cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()
            
            for o, l in zip(outputs, labels):
                y_pred.append(get_risk_level(o))
                y_true.append(get_risk_level(l))
    
    # 4. Generate Confusion Matrix
    categories = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL']
    plot_confusion_matrix_sns(y_true, y_pred, categories)
    
    print("\nEvaluation complete. Check 'metrics/' for results.")

if __name__ == "__main__":
    evaluate_model()
