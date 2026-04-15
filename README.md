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

3.  **Download Data**:
    If you want to retrain the model, you need the **RDD2022 (China_MotorBike)** subset.

    #### Automated Setup

    Run the setup script to create the directory structure and see detailed instructions:

    ```bash
    python3 -m src.setup_data
    ```

    #### Manual Download
    1.  Download the subset zip: [RDD2022_China_MotorBike.zip](https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/Country_Specific_Data_CRDDC2022/RDD2022_China_MotorBike.zip)
    2.  Extract the contents into `data/raw/China_MotorBike/`.
    3.  Ensure your structure looks like this:
        ```text
        data/raw/China_MotorBike/
        ├── train/
        │   ├── images/
        │   └── annotations/
        └── test/
            ├── images/
            └── annotations/
        ```
    4.  **Note**: If the zip contains an `xmls` folder inside `annotations`, move the `.xml` files directly into the `annotations` folder using:
        ```bash
        mv data/raw/China_MotorBike/train/annotations/xmls/*.xml data/raw/China_MotorBike/train/annotations/
        ```

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

## 🌐 REST API

You can also run the system as a REST API:

```bash
# Start the server
python3 -m src.api
```

### Endpoints
- **POST `/predict`**: Upload an image to get risk assessment.
  - Parameters: `image` (file), `weather` (optional), `traffic` (optional), `temp` (optional).
- **GET `/health`**: Check server status.

## 📂 Project Structure


- `src/model.py`: Neural Network Architecture.
- `src/inference.py`: Prediction script with Auto-Weather detection.
- `src/train.py`: Training loop.
- `src/data_generator.py`: Synthetic data simulation logic.
