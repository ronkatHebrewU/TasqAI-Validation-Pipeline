import os
from dotenv import load_dotenv
from roboflow import Roboflow
from pathlib import Path

# Load the .env file
load_dotenv()

# Access variables safely
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
PROJECT = os.getenv("ROBOFLOW_PROJECT")
VERSION_NUMBER = 1

def setup_roboflow():
    if not all([API_KEY, WORKSPACE, PROJECT]):
        print("Error: Please set ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, and ROBOFLOW_PROJECT in your .env file.")
        return

    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    version = project.version(VERSION_NUMBER)

    # Download the dataset in yolov8 format and extract to 'data' folder
    dataset = version.download("yolov8", location="data")

    # Print the first 5 image file paths in 'data/train/images'
    train_images_path = Path("data/train/images")

    if train_images_path.exists():
        # Get all files in the directory
        images = [f for f in train_images_path.iterdir() if f.is_file()]
        
        print(f"Found {len(images)} images. Listing the first 5:")
        for img in images[:5]:
            print(img)
    else:
        print(f"Directory {train_images_path} does not exist.")

if __name__ == "__main__":
    setup_roboflow()
