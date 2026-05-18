# QArmMini Computer Vision Demo

Real-time object detection + robotic arm control using MediaPipe, OpenCV, and PySerial.

## Setup (Step-by-Step)

### 1. **Clone/Download Project**
```bash
cd /path/to/Physical_AI_Hackathon
```

### 2. **Create Python Virtual Environment**
```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**What you're installing:**
- `numpy` - Numerical computations
- `opencv-python` - Camera & image processing
- `mediapipe` - AI object detection
- `pyserial` - Serial communication with robot

### 4. **Find Your Robot's Serial Port**

**macOS/Linux:**
```bash
ls /dev/tty* | grep -i usb
```
Look for `/dev/ttyUSB0` or `/dev/ttyUSB1` (Linux) or `/dev/tty.usbserial*` (macOS)

**Windows:**
- Device Manager → Ports (COM & LPT)
- Look for `COM3`, `COM4`, etc. (or whatever your device shows)

### 5. **Configure for Your Robot**

Edit `src/arm_config.py`:

```python
ARM_CONFIG = {
    "port": "/dev/ttyUSB0",  # ← UPDATE THIS to your port
    "baud_rate": 115200,      # Keep default unless your arm uses different
    # ... rest of config
}
```

### 6. **Test Camera Only (No Robot)**

This verifies everything works before connecting the arm:
```bash
python src/robot_vision_demo.py --no-robot
```

You should see your webcam with a window titled "QArmMini Vision Demo"

**Controls while running:**
- `t` - Toggle tracking mode
- `q` - Quit

### 7. **Run Full Demo (With Robot)**

Connect QArmMini to USB, then:
```bash
python src/robot_vision_demo.py
```

**Controls:**
- `t` - Toggle object tracking (arm follows detected objects)
- `g` - Grasp (close gripper)
- `r` - Release (open gripper)  
- `s` - Move to safe home position
- `q` - Quit

### Manual Keyboard Control

For direct arm control without the vision demo:
```bash
python run/run_model.py
```

This opens the robot keyboard controls with a live camera feed from OpenCV camera index `1`.
If your camera is on another index, use:
```bash
python run/run_model.py --camera-index 0
```

Roboflow inference is supported through the Serverless Hosted API workflow endpoint.
Set your workflow credentials first:
```powershell
$env:ROBOFLOW_API_KEY="your_api_key"
$env:ROBOFLOW_WORKSPACE_NAME="eshitas-workspace-gas5f"
$env:ROBOFLOW_WORKFLOW_ID="qarm-trash-ensemble-detection-1779037824337"
python run/run_model.py --vision-backend roboflow
```

Or put them in a local `.env.local` file. That file is ignored by Git and is loaded automatically:
```text
ROBOFLOW_API_KEY=your_api_key
ROBOFLOW_WORKSPACE_NAME=eshitas-workspace-gas5f
ROBOFLOW_WORKFLOW_ID=qarm-trash-ensemble-detection-1779037824337
ROBOFLOW_FALLBACK_WORKFLOW_ID=qarm-trash-detection-1779035433585
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_IMAGE_KEY=image
ROBOFLOW_CONFIDENCE=0.0
ROBOFLOW_EVERY_N_FRAMES=1
```

You can also pass them directly:
```bash
python run/run_model.py --vision-backend roboflow --roboflow-workspace-name your-workspace-name --roboflow-workflow-id your-workflow-id --roboflow-api-key YOUR_KEY
```

Useful Roboflow options:
```bash
python run/run_model.py --roboflow-confidence 0.45
python run/run_model.py --roboflow-classes plastic,metal,paper
python run/run_model.py --roboflow-every-n-frames 10
python run/run_model.py --no-roboflow-cache
python run/run_model.py --no-roboflow-annotated-image
```

The ensemble workflow is tried first. If Roboflow returns a server-side workflow error,
the script can fall back to `ROBOFLOW_FALLBACK_WORKFLOW_ID`.

Camera centering uses the numbered detected target list and moves the arm until the
selected target center is near the screen-center target reticle. Press `n` to move
centering to the next detected target:
```bash
python run/run_model.py --center-target-y-ratio 0.5
python run/run_model.py --center-base-sign -1 --center-shoulder-sign 1
python run/run_model.py --center-max-speed 0.3 --center-deadband-px 30
```
If centering moves away from the object, flip the matching sign.

Press `g` to run a selected-target grab. The arm will center the selected target,
approach in small inverse-kinematics steps, lower a little extra, close the
gripper, then lift. Press `x` to cancel a grab in progress:
```bash
python run/run_model.py --grab-approach-step 0.015
python run/run_model.py --grab-close-area-ratio 0.18
python run/run_model.py --grab-approach-sign 1
python run/run_model.py --grab-approach-axis tool
python run/run_model.py --grab-max-approach-steps 8
python run/run_model.py --grab-lower-m 0.035 --grab-lower-steps 3
```
If the arm moves away during the grab approach, flip `--grab-approach-sign`.
If IK still fails, try a smaller step such as `--grab-approach-step 0.005`.
The app now retries the approach using the opposite direction, smaller steps, and fallback axes before stopping.
If the gripper still closes too high, increase `--grab-lower-m` slightly.

The script also keeps the local PIT YOLO backend available:
```bash
python run/run_model.py --vision-backend pit-yolo
python run/run_model.py --yolo-model C:\path\to\model.pt
python run/run_model.py --yolo-confidence 0.45
python run/run_model.py --yolo-classes 39,41,46
python run/run_model.py --vision-backend none
```

If OpenCV reports that NumPy arrays are "not a numpy array", reinstall the dependencies.
This project pins NumPy below 2.0 because older OpenCV builds can reject NumPy 2 arrays:
```bash
pip install -r requirements.txt
```

Click/focus the "QArm Mini Keyboard Control + Camera" window, then use:
- Arrow keys - Move base left/right and shoulder up/down
- `w` / `s` - Move wrist up/down
- `p` - Close gripper
- `o` - Open gripper
- `h` - Move to home position
- `c` - Toggle camera centering on target #1 from the detected target list
- `n` - Move centering to the next detected target
- `g` - Center, approach, lower, close gripper, and lift the selected target
- `x` - Cancel a grab in progress
- `t` - Print Roboflow stats
- `q` or `Esc` - Quit

To run the same controls without camera access:
```bash
python run/run_model.py --no-camera
```

## What It Does

1. **Captures** video from your webcam in real-time
2. **Detects** objects using the configured vision backend
3. **Builds** a numbered target list from boxed detections
4. **Controls** the QArmMini robotic arm to center, approach, and grab the selected target

## Troubleshooting

### Camera not showing
- Check webcam permission (macOS/Linux): System Preferences → Security & Privacy → Camera
- Try: `python -c "import cv2; print(cv2.VideoCapture(0).read())"`

### "Cannot connect to arm"
- Verify port: `ls /dev/tty*`
- Check USB cable connection
- Try different baud rate in `arm_config.py` (usually 115200 or 9600)
- Restart arm and try again

### Object detection not working
- Falls back to red color detection automatically
- Make sure room has adequate lighting
- Try waving a bright red object in front of camera

## File Structure

```
Physical_AI_Hackathon/
├── src/
│   ├── robot_vision_demo.py    # Main demo script
│   └── arm_config.py           # Robot configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Next Steps

**To customize:**

1. **Different objects** - Edit `detect_objects_fallback()` in `robot_vision_demo.py` to detect other colors
2. **Different movements** - Modify `move_joints()` and `move_cartesian()` calls
3. **Different behaviors** - Add logic to `tracking_enabled` section
4. **Add more features** - Grasp specific object types, create waypoints, add depth sensing, etc.

## Quick Tips for Your Friend

1. **Don't forget to activate venv** before running:
   ```bash
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

2. **Test camera first** with `--no-robot` flag to make sure everything works

3. **Keep the arm in view** while it moves - always have emergency stop ready

4. **Red object test** - Use a red cup or object to test detection

## Support

If issues arise, check:
- Robot manual for correct serial port & baud rate
- Camera works with other apps (Zoom, Photo Booth, etc.)
- All dependencies installed: `pip list | grep -E "opencv|mediapipe|serial"`
