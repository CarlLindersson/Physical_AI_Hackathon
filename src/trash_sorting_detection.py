#!/usr/bin/env python3
"""
QArmMini Trash Sorting - Live Object Detection + Classification
Uses Roboflow workflow to detect and classify objects in real-time
Sends results to robot arm for picking and sorting

Controls:
  q - Quit
  s - Show stats
  p - Pick and place (demo)
  h - Home position
"""

import cv2
import base64
import numpy as np
from inference_sdk import InferenceHTTPClient
import time
import os
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from qarm_interface import QArmMiniRobot

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠ python-dotenv not installed, using hardcoded values")

# ============ CONFIGURATION ============
ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "6fuEkn2CDg3leS8F0O8j")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE_NAME", "eshitas-workspace-gas5f")
ROBOFLOW_WORKFLOW_ID = os.getenv("ROBOFLOW_WORKFLOW_ID", "qarm-trash-ensemble-detection-1779037824337")

QARM_SERIAL_PORT = os.getenv("QARM_SERIAL_PORT", "/dev/ttyUSB0")
QARM_BAUD = int(os.getenv("QARM_BAUD_RATE", "115200"))
USE_ROBOT = os.getenv("USE_ROBOT", "true").lower() == "true"
USE_SDK = os.getenv("USE_SDK", "true").lower() == "true"

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
FRAME_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))

# Calibration: map image pixels to robot workspace coordinates
CAMERA_X_OFFSET = float(os.getenv("CAMERA_X_OFFSET", "0.0"))
CAMERA_Y_OFFSET = float(os.getenv("CAMERA_Y_OFFSET", "0.0"))
PIXEL_TO_METER_X = float(os.getenv("PIXEL_TO_METER_X", "0.001"))
PIXEL_TO_METER_Y = float(os.getenv("PIXEL_TO_METER_Y", "0.001"))
PICK_Z = float(os.getenv("PICK_Z", "0.05"))
APPROACH_Z = float(os.getenv("APPROACH_Z", "0.10"))
LIFT_Z = float(os.getenv("LIFT_Z", "0.15"))

# Zone mapping: object class -> drop zone
ZONE_MAP = {
    "plastic bottles": "ZONE_A",
    "paper cup": "ZONE_B",
    "metal cans": "ZONE_C",
    "paper crumble": "ZONE_B",
    "paper box": "ZONE_B",
    "marker": "ZONE_D",
    "pen": "ZONE_D",
}

# Zone coordinates (update with your actual robot positions)
ZONE_COORDS = {
    "ZONE_A": {"x": 0.2, "y": 0.3, "z": 0.1},   # Plastic
    "ZONE_B": {"x": -0.2, "y": 0.3, "z": 0.1},  # Paper
    "ZONE_C": {"x": 0.0, "y": -0.3, "z": 0.1},  # Metal
    "ZONE_D": {"x": 0.0, "y": 0.0, "z": -0.1},  # Discard
}

OBJECT_PICK_PARAMS = {
    "paper cup": {"grip_strength": 0.35, "pick_z": 0.05, "priority": 0.4},
    "plastic bottles": {"grip_strength": 0.45, "pick_z": 0.06, "priority": 0.5},
    "metal cans": {"grip_strength": 0.80, "pick_z": 0.05, "priority": 0.9},
    "paper crumble": {"grip_strength": 0.30, "pick_z": 0.05, "priority": 0.3},
    "paper box": {"grip_strength": 0.40, "pick_z": 0.05, "priority": 0.6},
    "marker": {"grip_strength": 0.30, "pick_z": 0.04, "priority": 0.2},
    "pen": {"grip_strength": 0.30, "pick_z": 0.04, "priority": 0.2},
}

DEFAULT_OBJECT_PICK_PARAMS = {"grip_strength": 0.5, "pick_z": 0.055, "priority": 0.7}


class TrashSortingDemo:
    """Live trash detection and sorting with Roboflow + QArmMini"""
    
    def __init__(self):
        # Initialize Roboflow
        self.client = InferenceHTTPClient(
            api_url=ROBOFLOW_API_URL,
            api_key=ROBOFLOW_API_KEY
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        # Initialize robot
        self.robot = None
        if USE_ROBOT:
            self.robot = QArmMiniRobot(port=QARM_SERIAL_PORT, baud=QARM_BAUD, use_sdk=USE_SDK)
        
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()
        self.last_detection = None
        self.detected_objects = []
        
        print("="*60)
        print("QArmMini Trash Sorting - Live Detection")
        print("="*60)
        print("✓ Roboflow client initialized")
        print(f"✓ Camera opened: {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print(f"✓ Workflow: {ROBOFLOW_WORKFLOW_ID}")
        if self.robot:
            print(f"✓ Robot connected: {QARM_SERIAL_PORT}")
        else:
            print("⚠ Running in detection-only mode (no robot)")
        print("\nControls:")
        print("  q - Quit")
        print("  s - Show stats")
        print("  p - Pick current object")
        print("  a - Auto pick all visible objects")
        print("  h - Home position")
        print("="*60 + "\n")
    
    def run_detection(self, frame):
        """Send frame to Roboflow workflow"""
        try:
            result = self.client.run_workflow(
                workspace_name=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_WORKFLOW_ID,
                images={"image": frame},
                use_cache=True
            )
            return result
        except Exception as e:
            print(f"Workflow error: {e}")
            return None
    
    def draw_predictions(self, frame, predictions):
        """Draw bounding boxes and labels on frame"""
        h, w = frame.shape[:2]
        
        for pred in predictions:
            cls = pred.get("class", "unknown")
            conf = pred.get("confidence", 0)
            x = pred.get("x", 0)
            y = pred.get("y", 0)
            width = pred.get("width", 0)
            height = pred.get("height", 0)
            
            # Convert to pixel coordinates
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)
            
            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Get zone for this object
            zone = ZONE_MAP.get(cls, "ZONE_D")
            zone_color = (0, 255, 0) if conf > 0.7 else (255, 165, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), zone_color, 2)
            
            # Draw label
            label = f"{cls} ({conf:.2f}) -> {zone}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)
            
            # Draw center point
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        
        return frame
    
    def process_frame(self, frame):
        """Process single frame through detection pipeline"""
        self.frame_count += 1
        
        # Run detection
        result = self.run_detection(frame)
        if not result:
            return frame, None
        
        # Parse results
        try:
            output = result[0]
            predictions = output.get("predictions", {}).get("predictions", [])
            object_count = output.get("object_count", 0)
        except (IndexError, KeyError, TypeError) as e:
            print(f"Parse error: {e}")
            self.detected_objects = []
            return frame, None
        
        self.detected_objects = []
        if predictions:
            self.detection_count += len(predictions)
            
            # Sort by confidence
            sorted_preds = sorted(predictions, key=lambda p: p.get("confidence", 0), reverse=True)
            self.detected_objects = sorted_preds
            
            # Print detections
            print(f"\n[Frame {self.frame_count}] {len(predictions)} object(s) detected:")
            for i, pred in enumerate(sorted_preds):
                cls = pred.get("class", "unknown")
                conf = pred.get("confidence", 0)
                x = pred.get("x", 0)
                y = pred.get("y", 0)
                zone = ZONE_MAP.get(cls, "ZONE_D")
                print(f"  {i+1}. {cls} ({conf:.2f}) at ({x:.0f},{y:.0f}) → {zone}")
            
            # Get best prediction for picking
            best = sorted_preds[0]
            best_class = best.get("class", "unknown")
            best_zone = ZONE_MAP.get(best_class, "ZONE_D")
            best_x = best.get("x", FRAME_WIDTH/2)
            best_y = best.get("y", FRAME_HEIGHT/2)
            best_conf = best.get("confidence", 0)
            
            print(f"\n  → NEXT PICK: {best_class} ({best_conf:.2f}) at ({best_x:.0f},{best_y:.0f}) → {best_zone}")
            
            # Store for later pick/place
            self.last_detection = {
                "class": best_class,
                "zone": best_zone,
                "x": best_x,
                "y": best_y,
                "confidence": best_conf
            }
            
            # Draw on frame
            frame = self.draw_predictions(frame, sorted_preds)
            
            # Return annotated image from Roboflow if available
            if output.get("annotated_image"):
                img_data = base64.b64decode(output["annotated_image"])
                img_arr = np.frombuffer(img_data, dtype=np.uint8)
                annotated = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if annotated is not None:
                    frame = annotated
            
            return frame, best
        
        return frame, None
    
    def pixel_to_world(self, x: float, y: float, frame_width: int, frame_height: int):
        """Convert pixel coordinates to robot workspace coordinates."""
        dx = x - frame_width / 2.0
        dy = y - frame_height / 2.0
        world_x = CAMERA_X_OFFSET + dx * PIXEL_TO_METER_X
        world_y = CAMERA_Y_OFFSET + dy * PIXEL_TO_METER_Y
        return world_x, world_y

    def get_pick_params(self, obj_class: str):
        params = OBJECT_PICK_PARAMS.get(obj_class, DEFAULT_OBJECT_PICK_PARAMS)
        return params["grip_strength"], params["pick_z"], params["priority"]

    def sort_objects_for_pick(self, detections):
        """Sort detections by a simple ease+distance score."""
        world_centers = []
        for det in detections:
            x = det.get("x", FRAME_WIDTH / 2)
            y = det.get("y", FRAME_HEIGHT / 2)
            wx, wy = self.pixel_to_world(x, y, FRAME_WIDTH, FRAME_HEIGHT)
            _, _, priority = self.get_pick_params(det.get("class", "unknown"))
            distance = (wx ** 2 + wy ** 2) ** 0.5
            score = distance + priority
            world_centers.append((score, det, wx, wy))

        world_centers.sort(key=lambda item: item[0])
        return world_centers

    def do_robot_pick(self, det: dict) -> bool:
        """Pick a single detected object using robot motion sequence."""
        if not self.robot:
            print("⚠ No robot attached")
            return False

        object_x, object_y = self.pixel_to_world(det["x"], det["y"], FRAME_WIDTH, FRAME_HEIGHT)
        grip_strength, pick_z, _ = self.get_pick_params(det.get("class", "unknown"))
        zone_coords = ZONE_COORDS.get(det["zone"], {"x": 0.0, "y": 0.0, "z": PICK_Z})

        print(f"\n[ROBOT] object class={det['class']} pixel=({det['x']:.0f},{det['y']:.0f}) world=({object_x:.3f},{object_y:.3f},{pick_z:.3f}) grip={grip_strength:.2f}")
        print(f"[ROBOT] placing into {det['zone']} at ({zone_coords['x']:.3f},{zone_coords['y']:.3f},{zone_coords['z']:.3f})")

        success = self.robot.pick_and_place(
            x=object_x,
            y=object_y,
            z=pick_z,
            zone=det["zone"],
            zone_coords=zone_coords,
            grip_strength=grip_strength,
            return_home=True
        )
        return success

    def execute_pick_sequence(self):
        """Auto pick visible objects one by one."""
        if not self.robot:
            print("⚠ Robot not connected")
            return

        print("\n→ STARTING AUTO PICK SEQUENCE")
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to read frame for auto sequence")
            return

        frame, _ = self.process_frame(frame)
        if not self.detected_objects:
            print("No objects detected in frame. Auto sequence complete.")
            return

        objects = self.sort_objects_for_pick(self.detected_objects)
        for round_idx, (score, det, wx, wy) in enumerate(objects, start=1):
            print(f"\nRound {round_idx}: picking {det['class']} ({det['confidence']:.2f}) → {det['zone']} score={score:.2f}")
            if not self.do_robot_pick(det):
                print("Failed to complete pick. Aborting auto sequence.")
                break

            print("Pick complete. Re-acquiring scene for next object...")
            time.sleep(1.0)
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame after pick")
                break
            frame, _ = self.process_frame(frame)
            if not self.detected_objects:
                print("No remaining objects detected. Sequence finished.")
                break
        print("Auto pick sequence finished")

    def draw_ui(self, frame):
        """Draw status UI on frame"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        h, w = frame.shape[:2]
        
        # Status bar background
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        
        # Text
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Detections: {self.detection_count}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frames: {self.frame_count}", (w-200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self):
        """Main loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Process frame
                frame, best = self.process_frame(frame)
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Show frame
                cv2.imshow("QArmMini Trash Sorting - Roboflow Detection", frame)
                
                # Keyboard control
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    self.print_stats()
                elif key == ord('p'):
                    self.execute_pick()
                elif key == ord('a'):
                    self.execute_pick_sequence()
                elif key == ord('h'):
                    self.execute_home()
        
        except KeyboardInterrupt:
            print("\nInterrupted")
        
        finally:
            self.cleanup()
    
    def execute_pick(self):
        """Execute pick and place for last detected object"""
        if not self.last_detection:
            print("⚠ No detection yet - move object in front of camera first")
            return
        
        if not self.robot:
            print("⚠ Robot not connected")
            return
        
        det = self.last_detection
        print(f"\n→ EXECUTING PICK: {det['class']} to {det['zone']}")
        self.do_robot_pick(det)
    
    def execute_home(self):
        """Move robot to home position"""
        if not self.robot:
            print("⚠ Robot not connected")
            return
        
        print("→ MOVING TO HOME POSITION")
        self.robot.home()
    
    def print_stats(self):
        """Print session statistics"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"\n=== STATISTICS ===")
        print(f"Frames: {self.frame_count}")
        print(f"Detections: {self.detection_count}")
        print(f"FPS: {fps:.1f}")
        print(f"Time: {elapsed:.1f}s")
        if self.last_detection:
            print(f"Last detected: {self.last_detection['class']} → {self.last_detection['zone']}")
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        if self.robot:
            self.robot.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Done!")


def main():
    demo = TrashSortingDemo()
    demo.run()


if __name__ == "__main__":
    main()
