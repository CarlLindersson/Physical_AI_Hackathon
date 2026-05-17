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

## What It Does

1. **Captures** video from your webcam in real-time
2. **Detects** objects using MediaPipe AI model
3. **Tracks** the largest object in frame
4. **Controls** the QArmMini robotic arm to follow/grasp objects

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
