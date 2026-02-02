import json
from pathlib import Path
from vision_worker import VisionWorker
from data_filter import filter_detections
from validator import compare_to_gt
from visual_report import generate_report_graph, generate_pipeline_story_graph
from vlm_auditor import run_vlm_audit

def main():
    # Define paths
    images_path = Path("data/train/images")
    audit_file = "to_audit.json"
    human_intervention_file = "human_intervention_required.json"

    # Initialize VisionWorker
    worker = VisionWorker()

    # 1. Run inference
    print(f"Processing images in {images_path}...")
    # Limit to 10 images for testing
    detections = worker.process_folder(str(images_path), limit=10)
    
    # 2. Post-YOLO Validation (Raw Stats)
    print("Performing initial Raw Validation...")
    raw_total = len(detections)
    raw_correct = 0
    
    for detection in detections:
        if compare_to_gt(detection) is None:
            raw_correct += 1

    initial_accuracy = (raw_correct / raw_total * 100) if raw_total > 0 else 0
    print(f"Initial Raw Accuracy: {initial_accuracy:.2f}%")

    # 3. Filter results (Confidence Check)
    filtered_results = filter_detections(detections)
    
    confident_detections = filtered_results["confident_detections"]
    audit_required = filtered_results["audit_required"]

    # 4. Post-Filter Validation (Confident Stats)
    print("Validating confident detections against Ground Truth...")
    filtered_total = len(confident_detections)
    filtered_correct = 0
    validated_confident = []
    
    for detection in confident_detections:
        validation_result = compare_to_gt(detection)
        if validation_result:
            audit_required.append(validation_result)
        else:
            filtered_correct += 1
            validated_confident.append(detection)
            
    confident_detections = validated_confident
    clean_accuracy = (filtered_correct / filtered_total * 100) if filtered_total > 0 else 0

    # Save audit list
    with open(audit_file, "w") as f:
        json.dump(audit_required, f, indent=4)
    
    # 5. Run VLM Audit
    print("\n--- Running VLM Audit ---")
    vlm_results = run_vlm_audit(audit_file)
    
    human_intervention_required = []
    vlm_passed_list = []

    for item in vlm_results:
        vlm_verification = item.get('vlm_verification')
        if vlm_verification not in ["YES", "NO"]:
            item['human_flag_reason'] = 'Uncertain VLM Response'
            human_intervention_required.append(item)
        else:
            vlm_passed_list.append(item)

    vlm_total = len(vlm_results)
    vlm_passed = sum(1 for item in vlm_passed_list if item.get('vlm_verification') == 'YES')
    human_total = len(human_intervention_required)

    # Save human intervention list
    with open(human_intervention_file, "w") as f:
        json.dump(human_intervention_required, f, indent=4)

    # Calculate file statistics
    total_images = 0
    if images_path.exists():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        # Just counting files here, but we only processed 10
        total_images = sum(1 for f in images_path.iterdir() if f.suffix.lower() in image_extensions and f.is_file())
    
    # Print summary
    print("\n--- Processing Summary ---")
    print(f"Total Images in Folder: {total_images}")
    print(f"Images Processed: {min(10, total_images)}")
    print(f"Total Raw Detections: {raw_total}")
    print(f"Correct Raw Detections: {raw_correct}")
    print(f"Total Filtered (Confident) Detections: {filtered_total}")
    print(f"Correct Filtered Detections: {filtered_correct}")
    print(f"VLM Audit Total: {vlm_total}")
    print(f"VLM Audit Passed (YES): {vlm_passed}")
    print(f"Human Intervention Required: {human_total}")
    print(f"Clean Data Accuracy: {clean_accuracy:.2f}%")

    # Generate Graphs
    generate_report_graph(raw_total, raw_correct, filtered_total, filtered_correct, vlm_total, vlm_passed, human_total)
    generate_pipeline_story_graph()

if __name__ == "__main__":
    main()
