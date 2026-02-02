from pathlib import Path
import yaml
from PIL import Image

def load_class_names(yaml_path='data/data.yaml'):
    """Loads class names from the data.yaml file."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            return data.get('names', [])
    except Exception as e:
        print(f"Error loading class names: {e}")
        return []

CLASS_NAMES = load_class_names()

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1 (list): [x1, y1, x2, y2]
        box2 (list): [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0
    
    return intersection_area / union_area

def yolo_to_xyxy(x_center, y_center, width, height, img_width, img_height):
    """
    Convert YOLO format (normalized center_x, center_y, width, height) to [x1, y1, x2, y2].
    """
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)

    return [x1, y1, x2, y2]

def compare_to_gt(detection, labels_dir='data/train/labels', iou_threshold=0.5):
    """
    Compares a detection dictionary against the ground truth label file.
    
    Args:
        detection (dict): Detection dictionary from YOLO.
        labels_dir (str): Path to the directory containing label files.
        iou_threshold (float): Minimum IoU required to consider a match valid.

    Returns:
        dict or None: Returns the detection dict with 'flag_reason' if it fails validation, else None.
    """
    image_path = Path(detection['image_path'])
    label_file_path = Path(labels_dir) / image_path.with_suffix('.txt').name
    
    if not label_file_path.exists():
        detection['flag_reason'] = 'Missing Ground Truth File'
        return detection

    # Get image dimensions
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        detection['flag_reason'] = f'Error reading image dimensions: {e}'
        return detection

    # Read GT labels
    gt_boxes = []
    with open(label_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                bbox = yolo_to_xyxy(float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), img_width, img_height)
                
                # Get class name from ID
                cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
                
                gt_boxes.append({'cls_name': cls_name, 'bbox': bbox})

    # Check for match
    det_bbox = detection['bbox']
    det_label = detection['label']
    
    best_iou = 0
    matched_gt = None

    for gt in gt_boxes:
        iou = calculate_iou(det_bbox, gt['bbox'])
        if iou > best_iou:
            best_iou = iou
            matched_gt = gt

    if best_iou < iou_threshold:
        detection['flag_reason'] = f'Low IoU with GT ({best_iou:.2f})'
        return detection
    
    if matched_gt and matched_gt['cls_name'] != det_label:
        detection['flag_reason'] = f"Class Mismatch (Det: {det_label}, GT: {matched_gt['cls_name']})"
        return detection
    
    return None
