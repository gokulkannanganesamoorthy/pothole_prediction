import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data_loader import RoadDamageDataset
from src.model import PotholeRiskModel
import os

def train_model(num_epochs=5, batch_size=16, learning_rate=0.001):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Dataset
    print("Loading Dataset...")
    full_dataset = RoadDamageDataset(root_dir='data/raw/China_MotorBike', mode='train')
    
    # Split Train/Val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 2. Initialize Model
    model = PotholeRiskModel().to(device)
    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 3. Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, metadata, labels) in enumerate(train_loader):
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Log every 10 batches
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)
                
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
    # 4. Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/pothole_risk_model.pth')
    print("Model saved to models/pothole_risk_model.pth")

if __name__ == "__main__":
    # Run for just 2 epochs for demonstration
    train_model(num_epochs=2, batch_size=16)
