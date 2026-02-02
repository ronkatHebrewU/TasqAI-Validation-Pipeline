from ultralytics import YOLO
from pathlib import Path
from typing import List, Dict, Any

class VisionWorker:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def process_folder(self, folder_path: str, limit: int = None) -> List[Dict[str, Any]]:
        folder = Path(folder_path)
        if not folder.exists():
            print(f"Folder {folder_path} does not exist.")
            return []

        # Define common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        results_data = []
        
        # Find all images
        images = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]

        if not images:
            return []

        # Apply limit if specified
        if limit:
            images = images[:limit]
            print(f"Limiting processing to first {limit} images.")

        # Run inference on the list of images.
        # stream=True returns a generator, which is memory efficient for large folders.
        results = self.model(images, stream=True)

        for result in results:
            # result.path contains the path to the image file processed
            current_path = result.path
            
            for box in result.boxes:
                # Extract data
                cls_id = int(box.cls[0].item())
                label = result.names[cls_id]
                confidence = round(box.conf[0].item(), 2)
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                detection = {
                    'image_path': current_path,
                    'label': label,
                    'confidence': confidence,
                    'bbox': bbox
                }
                results_data.append(detection)
                    
        return results_data
