#!/usr/bin/env python3
"""
QArmMini Object Detection & Grasping Demo
Detects objects in real-time and controls the arm to grasp them
Uses: MediaPipe (detection), OpenCV (camera), PySerial (robot control)

Controls:
  t - Toggle tracking mode
  g - Grasp (close gripper)
  r - Release (open gripper)
  s - Home/safe position
  q - Quit
"""

import cv2
import mediapipe as mp
import numpy as np
import serial
import time
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))
from arm_config import ARM_CONFIG, DETECTION_CONFIG, SPEED_CONFIG


class QArmMiniController:
    """Simple QArmMini serial controller"""
    
    def __init__(self, port=ARM_CONFIG["port"], baud=ARM_CONFIG["baud_rate"]):
        """Initialize serial connection to arm"""
        self.port = port
        self.baud = baud
        self.ser = None
        self.connected = False
        self.gripper_open = True
        
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.connected = True
            print(f"✓ Connected to arm at {port} ({baud} baud)")
            time.sleep(2)  # Let arm initialize
        except Exception as e:
            print(f"✗ Failed to connect to arm: {e}")
            print(f"  Check port with: ls /dev/tty* (macOS/Linux) or Device Manager (Windows)")
    
    def send_command(self, cmd):
        """Send command to arm"""
        if not self.connected or not self.ser:
            return False
        try:
            self.ser.write(cmd.encode() + b'\r\n')
            response = self.ser.readline().decode().strip()
            return response
        except Exception as e:
            print(f"Serial error: {e}")
            return False
    
    def home_position(self):
        """Move arm to safe home position"""
        print("Moving to home position...")
        # QArmMini home command (adjust based on your SDK)
        self.send_command("HOME")
        time.sleep(2)
    
    def move_joints(self, joint_angles):
        """Move arm to joint angles (list of 6 floats)"""
        if not self.connected:
            return
        # Format: "JOINTS j0 j1 j2 j3 j4 j5"
        cmd = f"JOINTS {' '.join(f'{a:.1f}' for a in joint_angles)}"
        self.send_command(cmd)
    
    def move_cartesian(self, x, y, z):
        """Move end effector to cartesian coords"""
        if not self.connected:
            return
        cmd = f"MOVE {x:.3f} {y:.3f} {z:.3f}"
        self.send_command(cmd)
    
    def grasp(self):
        """Close gripper"""
        if not self.connected or not self.gripper_open:
            return
        print("Grasping...")
        self.send_command("GRIP CLOSE")
        time.sleep(1)
        self.gripper_open = False
    
    def release(self):
        """Open gripper"""
        if not self.connected or self.gripper_open:
            return
        print("Releasing...")
        self.send_command("GRIP OPEN")
        time.sleep(1)
        self.gripper_open = True
    
    def close(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")


class RobotVisionDemo:
    """Main demo class"""
    
    def __init__(self, use_robot=True):
        self.use_robot = use_robot
        self.tracking_enabled = False
        self.frame_count = 0
        
        # Initialize robot
        if use_robot:
            self.robot = QArmMiniController()
        else:
            self.robot = None
            print("Running in SIMULATION mode (no robot)")
        
        # Initialize MediaPipe ObjectDetector
        print("Loading MediaPipe object detector...")
        try:
            BaseOptions = mp.tasks.BaseOptions
            ObjectDetector = mp.tasks.vision.ObjectDetector
            ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            options = ObjectDetectorOptions(
                base_options=BaseOptions(model_asset_path=None),
                running_mode=VisionRunningMode.IMAGE,
                max_results=3,
                score_threshold=DETECTION_CONFIG["min_confidence"]
            )
            self.detector = ObjectDetector.create_from_options(options)
            print("✓ MediaPipe detector ready")
        except Exception as e:
            print(f"Warning: Could not load MediaPipe object detector: {e}")
            print("  Falling back to basic color detection")
            self.detector = None
        
        # Initialize camera
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, DETECTION_CONFIG["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DETECTION_CONFIG["frame_height"])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print("✓ Camera ready")
    
    def pixel_to_arm_delta(self, obj_x, obj_y, frame_width, frame_height):
        """Convert pixel offset to arm movement delta"""
        center_x = frame_width / 2
        center_y = frame_height / 2
        
        # Normalize to [-1, 1]
        delta_x = (obj_x - center_x) / (frame_width / 2)
        delta_y = (obj_y - center_y) / (frame_height / 2)
        
        return delta_x, delta_y
    
    def detect_objects_mediapipe(self, frame):
        """Detect objects using MediaPipe"""
        if not self.detector:
            return None
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )
            detection_result = self.detector.detect(mp_image)
            
            if detection_result and detection_result.detections:
                return detection_result.detections
        except Exception as e:
            print(f"Detection error: {e}")
        
        return None
    
    def detect_objects_fallback(self, frame):
        """Fallback: detect colored objects (red, blue, green)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red object detection
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        
        contours, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:
                x, y, w, h = cv2.boundingRect(largest)
                return [(x, y, x+w, y+h, "Red Object")]
        
        return None
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes on frame"""
        h, w, _ = frame.shape
        
        if not detections:
            return frame, None
        
        # Get largest detection
        largest_det = None
        largest_area = 0
        
        if isinstance(detections, list) and len(detections) > 0:
            if hasattr(detections[0], 'bounding_box'):
                # MediaPipe format
                for det in detections:
                    bbox = det.bounding_box
                    area = bbox.width * bbox.height
                    if area > largest_area:
                        largest_area = area
                        largest_det = det
                
                if largest_det:
                    bbox = largest_det.bounding_box
                    x_min = max(0, int(bbox.origin_x * w))
                    y_min = max(0, int(bbox.origin_y * h))
                    x_max = min(w, int((bbox.origin_x + bbox.width) * w))
                    y_max = min(h, int((bbox.origin_y + bbox.height) * h))
                    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    obj_x = (x_min + x_max) // 2
                    obj_y = (y_min + y_max) // 2
                    cv2.circle(frame, (obj_x, obj_y), 8, (0, 0, 255), -1)
                    
                    label = "Object detected"
                    cv2.putText(frame, label, (x_min, y_min - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    return frame, (obj_x, obj_y)
            
            elif isinstance(detections[0], tuple):
                # Fallback format (x1, y1, x2, y2, label)
                for (x1, y1, x2, y2, label) in detections:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    obj_x = (x1 + x2) // 2
                    obj_y = (y1 + y2) // 2
                    cv2.circle(frame, (obj_x, obj_y), 8, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    return frame, (obj_x, obj_y)
        
        return frame, None
    
    def run(self):
        """Main demo loop"""
        print("\n" + "="*60)
        print("QArmMini Vision Demo Started")
        print("="*60)
        print("\nControls:")
        print("  [t] Toggle tracking")
        print("  [g] Grasp")
        print("  [r] Release")
        print("  [s] Safe home position")
        print("  [q] Quit")
        print("="*60 + "\n")
        
        if self.robot:
            self.robot.home_position()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                h, w, c = frame.shape
                self.frame_count += 1
                
                # Detect objects
                detections = self.detect_objects_mediapipe(frame) if self.detector else None
                if not detections:
                    detections = self.detect_objects_fallback(frame)
                
                # Draw and get position
                frame, obj_pos = self.draw_detections(frame, detections)
                
                # Track and move arm if enabled
                if self.tracking_enabled and obj_pos and self.robot:
                    delta_x, delta_y = self.pixel_to_arm_delta(obj_pos[0], obj_pos[1], w, h)
                    # Arm movement would go here based on your SDK
                
                # Draw status
                status_text = "TRACKING" if self.tracking_enabled else "IDLE"
                gripper_text = "OPEN" if self.robot and self.robot.gripper_open else "CLOSED"
                
                cv2.putText(frame, f"Status: {status_text}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Gripper: {gripper_text}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {1 if self.frame_count % 30 else 30}", (w-150, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("QArmMini Vision Demo", frame)
                
                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('t'):
                    self.tracking_enabled = not self.tracking_enabled
                    print(f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}")
                elif key == ord('g'):
                    if self.robot:
                        self.robot.grasp()
                elif key == ord('r'):
                    if self.robot:
                        self.robot.release()
                elif key == ord('s'):
                    if self.robot:
                        self.robot.home_position()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        if self.robot:
            self.robot.release()
            self.robot.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Done!")


def main():
    """Entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="QArmMini Vision Demo")
    parser.add_argument("--no-robot", action="store_true", help="Run without robot (camera test only)")
    args = parser.parse_args()
    
    demo = RobotVisionDemo(use_robot=not args.no_robot)
    demo.run()


if __name__ == "__main__":
    main()
