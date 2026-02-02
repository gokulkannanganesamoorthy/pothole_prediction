# Pothole Risk Prediction System

A Multi-Modal Deep Learning system that predicts the **risk of pothole formation** by combining visual road analysis with environmental conditions (Weather, Traffic).

## 🚀 Features

- **Visual Analysis**: Uses a ResNet18 CNN to detect road damage (cracks, existing potholes).
- **Environmental Context**: Incorporates Weather (Rain/Snow) and Traffic load into the risk calculation.
- **Auto-Detect Weather**: Automatically estimates whether it's Sunny, Rainy, or Snowy from the image itself.
- **Risk Prediction**: Outputs a "Risk Score" and "Severity Level" (Low, Moderate, High, Critical).

## 🛠️ Setup

1.  **Create a Virtual Environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Data (Optional for Inference)**:
    If you want to retrain the model, you need the RDD2022 dataset in `data/raw/`.
    _(The inference script works with the pre-trained model provided)._

## 🏃 Usage

### Quick Start

Run the helper script to auto-train (if needed) and test on a sample image:

```bash
./run.sh
```

### Run on Your Own Image

To test a specific road image:

```bash
# Basic (Auto-detect weather)
python -m src.inference --image path/to/image.jpg

# Simulation (Force specific conditions)
python -m src.inference --image path/to/image.jpg --weather Rainy --traffic High
```

### Options

- `--image`: Path to the image file.
- `--weather`: `Sunny`, `Rainy`, `Snowy` (Auto-detected if omitted).
- `--traffic`: `Low`, `Medium`, `High` (Default: Low).
- `--temp`: Temperature in Celsius (Default: 25).

## 🧠 How it Works

1.  **Visual Encoder**: The CNN extracts features from the road image (e.g., detecting alligator cracks).
2.  **Metadata Encoder**: A separate network processes the environmental factors.
3.  **Fusion Layer**: Combines both inputs to calculate a final **Risk Score**.
    - _Example_: A road with minor cracks (Medium Risk) + Heavy Rain (Multiplier) = **Critical Risk**.

## 📂 Project Structure

- `src/model.py`: Neural Network Architecture.
- `src/inference.py`: Prediction script with Auto-Weather detection.
- `src/train.py`: Training loop.
- `src/data_generator.py`: Synthetic data simulation logic.
