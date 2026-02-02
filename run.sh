#!/bin/bash
echo "Activating virtual environment..."
source venv/bin/activate

echo "Checking if model exists..."
# Always force retrain for now since we changed logic
rm -f models/pothole_risk_model.pth

if [ ! -f "models/pothole_risk_model.pth" ]; then
    echo "Model not found. Starting training..."
    python -m src.train
else
    echo "Model found. Skipping training."
fi

echo "Running Inference on Test Set..."
python -m src.inference
