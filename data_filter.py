CONFIDENCE_THRESHOLD = 0.7

def filter_detections(detections):
    """
    Filters a list of detection dictionaries into confident and audit-required categories.
    
    Args:
        detections (list): A list of dictionaries, each containing detection info.
                           Expects keys like 'image_path', 'label', 'confidence', 'bbox'.
                           If an image has no detections, it's expected to be in the list
                           with 'label' as None.

    Returns:
        dict: A dictionary with 'confident_detections' and 'audit_required' lists.
    """
    filtered_results = {
        "confident_detections": [],
        "audit_required": []
    }

    for detection in detections:
        # Handle cases where an image has no detections
        # Assuming such cases are represented by a dictionary with label=None
        if detection.get('label') is None:
            detection['flag_reason'] = 'No Objects Found'
            filtered_results['audit_required'].append(detection)
            continue

        confidence = detection.get('confidence', 0.0)

        if confidence >= CONFIDENCE_THRESHOLD:
            filtered_results['confident_detections'].append(detection)
        else:
            detection['flag_reason'] = 'Low Confidence'
            filtered_results['audit_required'].append(detection)

    return filtered_results
