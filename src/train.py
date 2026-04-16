import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data_loader import RoadDamageDataset
from src.model import PotholeRiskModel
import os
import json
from src.utils import plot_training_history

def train_model(num_epochs=5, batch_size=16, learning_rate=0.001):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Prepare Dataset
    print("Loading Dataset...")
    data_path = 'data/raw/China_MotorBike'
    if not os.path.exists(os.path.join(data_path, 'train', 'images')):
        print(f"\nERROR: Training data not found at {data_path}")
        print("Please run 'python -m src.setup_data' for instructions on how to acquire the dataset.")
        return

    full_dataset = RoadDamageDataset(root_dir=data_path, mode='train')
    if len(full_dataset) == 0:
        print(f"\nERROR: No images found in {data_path}/train/images")
        print("Please follow the instructions from 'python -m src.setup_data' to acquire the dataset.")
        return
    
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
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_loss_steps': [],
        'train_acc_steps': []
    }
    
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
            
            # Log and track every 10 batches
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
                # Step-wise metrics
                history['train_loss_steps'].append(loss.item())
                predicted = (outputs > 0.5).float()
                acc = (predicted == (labels > 0.5).float()).sum().item() / labels.size(0)
                history['train_acc_steps'].append(acc)
            
        avg_train_loss = running_loss / len(train_loader)
        
        # Calculate Training Accuracy (using 0.5 threshold for binary)
        model.eval()
        train_correct = 0
        train_total = 0
        with torch.no_grad():
            for images, metadata, labels in train_loader:
                images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
                outputs = model(images, metadata)
                predicted = (outputs > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == (labels > 0.5).float()).sum().item()
        avg_train_acc = train_correct / train_total
        
        # Validation
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, metadata, labels in val_loader:
                images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
                outputs = model(images, metadata)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == (labels > 0.5).float()).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
        
    # 4. Save Model and Metrics
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/history.json', 'w') as f:
        json.dump(history, f)
    
    plot_training_history(history)
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/pothole_risk_model.pth')
    print("Model saved to models/pothole_risk_model.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    
    train_model(num_epochs=args.epochs, batch_size=args.batch_size)
