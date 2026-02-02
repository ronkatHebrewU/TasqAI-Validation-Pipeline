import json
import os
import time
from pathlib import Path
from google import genai
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Configure Gemini API using GOOGLE_API_KEY
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("Warning: GOOGLE_API_KEY not found in .env file.")
    client = None
else:
    client = genai.Client(api_key=API_KEY)

def get_vlm_opinion(image_path, detected_label):
    """
    Sends an image to Gemini 2.0 Flash to check for the detected label.
    """
    if not client:
        return "ERROR: API KEY NOT CONFIGURED"
        
    try:
        img = Image.open(image_path)
        
        prompt = (f"You are a Quality Control expert. A YOLO model detected a {detected_label}. "
                  f"Is there actually a {detected_label} in this image? Answer only YES or NO.")
        
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[prompt, img]
        )
        
        # Clean up response text
        answer = response.text.strip().upper()
        
        # Basic validation of the answer
        if "YES" in answer:
            return "YES"
        elif "NO" in answer:
            return "NO"
        else:
            return f"UNCERTAIN ({answer})"
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "ERROR"

def run_vlm_audit(audit_file_path="to_audit.json", output_file_path="final_report.json"):
    audit_file = Path(audit_file_path)
    output_file = Path(output_file_path)

    if not audit_file.exists():
        print(f"Error: {audit_file} not found.")
        return []

    try:
        with open(audit_file, 'r') as f:
            audit_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode {audit_file}.")
        return []

    print(f"Loaded {len(audit_data)} items for VLM audit using Gemini 1.5 Flash.")

    verified_results = []

    for item in audit_data:
        image_path = item.get('image_path')
        detected_label = item.get('label', 'object')
        
        if not image_path or not Path(image_path).exists():
            print(f"Skipping missing image: {image_path}")
            item['vlm_verification'] = "IMAGE_NOT_FOUND"
            verified_results.append(item)
            continue

        print(f"Auditing {image_path} for {detected_label}...")
        
        # Call Gemini
        vlm_result = get_vlm_opinion(image_path, detected_label)
        
        # Update the item with VLM result
        item['vlm_verification'] = vlm_result
        
        if vlm_result == "NO":
            item['vlm_suggested_action'] = "REJECT"
        elif vlm_result == "YES":
            item['vlm_suggested_action'] = "APPROVE"
        else:
             item['vlm_suggested_action'] = "MANUAL_REVIEW"
        
        verified_results.append(item)
        
        # Respect rate limits (simple pause)
        time.sleep(1)

    # Save final report
    with open(output_file, 'w') as f:
        json.dump(verified_results, f, indent=4)

    print(f"Audit complete. Final report saved to {output_file}")
    return verified_results

if __name__ == "__main__":
    run_vlm_audit()
