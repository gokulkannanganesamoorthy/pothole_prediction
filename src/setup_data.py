import os
import sys

def setup_directories():
    """Creates the required directory structure for the dataset."""
    dirs = [
        'data/raw/China_MotorBike/train/images',
        'data/raw/China_MotorBike/train/annotations',
        'data/raw/China_MotorBike/test/images',
        'data/raw/China_MotorBike/test/annotations',
        'models'
    ]
    
    print("Setting up directory structure...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")

def provide_instructions():
    """Provides download instructions for the user."""
    print("\n" + "="*50)
    print("DATA RECOVERY INSTRUCTIONS")
    print("="*50)
    print("The RDD2022 China_MotorBike subset is required for training.")
    print("\nOption 1: Manual Download")
    print("1. Visit: https://github.com/sekilab/RoadDamageDetector")
    print("2. Download the 'China_MotorBike' subset.")
    print("3. Place .jpg files in: data/raw/China_MotorBike/train/images")
    print("4. Place .xml files in: data/raw/China_MotorBike/train/annotations")
    
    print("\nOption 2: Programmatic (Experimental)")
    print("You can try using dataset-tools:")
    print("pip install dataset-tools")
    print("python -c \"import dataset_tools as dtools; dtools.download(dataset='RDD2022', dst_dir='data/external/')\"")
    print("Note: You will then need to move/symlink China_MotorBike to data/raw/")
    print("="*50 + "\n")

if __name__ == "__main__":
    setup_directories()
    provide_instructions()
