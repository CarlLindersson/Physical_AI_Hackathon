#!/usr/bin/env python3
"""
Quick Test: Roboflow Live Detection on Webcam
Run this FIRST to verify detection works before connecting robot
No dependencies on QArmMini - just camera + Roboflow API
"""

import cv2
import base64
import numpy as np
from inference_sdk import InferenceHTTPClient
import os

# Try to load env vars, but use defaults if not found
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuration
API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
API_KEY = os.getenv("ROBOFLOW_API_KEY", "6fuEkn2CDg3leS8F0O8j")
WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE_NAME", "eshitas-workspace-gas5f")
WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "qarm-trash-ensemble-detection-1779037824337")

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

print("="*60)
print("Roboflow Webcam Test")
print("="*60)
print(f"API URL: {API_URL}")
print(f"Workspace: {WORKSPACE}")
print(f"Workflow: {WORKFLOW_ID}")
print(f"Camera index: {CAMERA_INDEX}")
print("="*60)
print("\nPress 'q' to quit, 's' for stats\n")

# Initialize Roboflow client
try:
    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=API_KEY
    )
    print("✓ Connected to Roboflow API")
except Exception as e:
    print(f"✗ Failed to connect to Roboflow: {e}")
    exit(1)

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"✗ Failed to open camera {CAMERA_INDEX}")
    print(f"  Try: python -c \"import cv2; print(cv2.VideoCapture(0).read())\"")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("✓ Camera opened (640x480)")

frame_count = 0
detection_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        frame_count += 1
        
        # Send to Roboflow
        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": frame},
                use_cache=True
            )
        except Exception as e:
            print(f"Workflow error: {e}")
            cv2.putText(frame, f"ERROR: {str(e)[:50]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Roboflow Webcam Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        
        # Parse results
        try:
            output = result[0]
            predictions = output.get("predictions", {}).get("predictions", [])
            object_count = output.get("object_count", 0)
        except (IndexError, KeyError, TypeError):
            predictions = []
            object_count = 0
        
        if predictions:
            detection_count += len(predictions)
            print(f"[Frame {frame_count}] Detected {len(predictions)} object(s):")
            for pred in predictions:
                cls = pred.get("class", "?")
                conf = pred.get("confidence", 0)
                print(f"  - {cls} ({conf:.2f})")
        
        # Display annotated image from Roboflow if available
        display_frame = frame
        if output.get("annotated_image"):
            try:
                img_data = base64.b64decode(output["annotated_image"])
                img_arr = np.frombuffer(img_data, dtype=np.uint8)
                annotated = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if annotated is not None:
                    display_frame = annotated
            except Exception as e:
                print(f"Could not decode annotated image: {e}")
        
        # Draw status
        h, w = display_frame.shape[:2]
        cv2.rectangle(display_frame, (0, 0), (w, 50), (0, 0, 0), -1)
        cv2.putText(display_frame, f"Frame: {frame_count} | Objects: {object_count}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Total Detections: {detection_count}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Roboflow Webcam Test", display_frame)
        
        # Keyboard control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\n=== STATS ===")
            print(f"Total frames: {frame_count}")
            print(f"Total detections: {detection_count}")
            print(f"Average objects per frame: {detection_count/frame_count:.2f}")
            print()

except KeyboardInterrupt:
    print("\nInterrupted")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nTest complete!")
