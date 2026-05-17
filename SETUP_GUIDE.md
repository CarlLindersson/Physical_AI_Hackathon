# QArmMini Trash Sorting - Complete Setup Guide

## What This Does

✓ **Live object detection** using Roboflow AI  
✓ **Real-time classification** of plastic/paper/other  
✓ **Zone mapping** - each object class → pickup zone  
✓ **Robot integration** - sends pick/place commands to QArmMini  
✓ **Live visualization** - annotated camera feed with bounding boxes  

---

## Quick Start (For Your Friend)

### 1. **Clone/Pull the Latest Code**
```bash
cd Physical_AI_Hackathon
git pull origin main
```

### 2. **Activate Virtual Environment**
```bash
source .venv/bin/activate
# (or .venv\Scripts\activate on Windows)
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

Should install:
- `opencv-python` - camera
- `inference-sdk` - Roboflow API
- `pyserial` - robot communication
- Others

### 4. **Set Up Environment Configuration**
```bash
cp .env.example .env
```

Edit `.env` and update:
```
ROBOFLOW_API_KEY=YOUR_KEY_HERE
ROBOFLOW_WORKSPACE_NAME=YOUR_WORKSPACE
ROBOFLOW_WORKFLOW_ID=YOUR_WORKFLOW_ID
QARM_SERIAL_PORT=/dev/ttyUSB0  # Find your actual port with: ls /dev/tty*
USE_ROBOT=true  # Set to false to test detection without robot
```

### 5. **Find Your Robot's Serial Port**

**macOS/Linux:**
```bash
ls /dev/tty* | grep -E "USB|usbserial"
```
Look for `/dev/ttyUSB0`, `/dev/ttyUSB1`, or `/dev/tty.usbserial-*`

**Windows:**
- Device Manager → Ports (COM & LPT)
- Look for your device (usually COM3, COM4, etc.)

Update in `.env`:
```
QARM_SERIAL_PORT=/dev/ttyUSB0
```

### 6. **Run the Demo**

**Test detection first (no robot):**
```bash
USE_ROBOT=false python src/trash_sorting_detection.py
```

You should see:
- Live camera feed in a window
- Detected objects with bounding boxes
- Classification labels

**With robot (if connected):**
```bash
python src/trash_sorting_detection.py
```

---

## Controls During Demo

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Show stats (FPS, frame count, etc.) |
| `p` | Pick and place last detected object |
| `h` | Move robot to home position |

---

## How It Works

### Detection Pipeline

1. **Camera captures frame** (640x480, 30 FPS)
2. **Roboflow workflow processes** frame
   - Uses your trained model
   - Returns bounding boxes + class labels
3. **Classification** → maps class to zone:
   - `plastic` → ZONE_A
   - `paper` → ZONE_B
   - `metal` → ZONE_C
   - `other` → ZONE_D (discard)
4. **Visualization** - draws boxes and labels on live feed
5. **Robot control** - optional, press `p` to execute pick/place

### Frame Flow
```
Camera → Roboflow API → Detection Result → Classification → Robot Cmd
  ↓                           ↓                 ↓              ↓
30 FPS                   Annotated Image   Zone Map         Pick/Place
```

---

## Troubleshooting

### No camera appearing
- Check camera is not in use by other apps
- Try: `python -c "import cv2; print(cv2.VideoCapture(0).read())"`
- macOS: Settings → Security & Privacy → Camera (grant permission)

### Roboflow connection failed
```
ERROR: Could not connect to Roboflow API
```
- Check internet connection
- Verify API key in `.env` is correct
- Check workspace name and workflow ID

### Robot not responding
```
ERROR: Failed to connect to robot
```
- Verify serial port: `ls /dev/tty*`
- Check USB cable is connected
- Try different baud rate in `.env` (usually 115200 or 9600)
- Power cycle robot

### Poor detection accuracy
- Check lighting is adequate
- Ensure objects are clearly visible in camera
- Check object is within frame
- If needed, retrain Roboflow model with more examples

### No annotated images showing
- Some workflow versions don't return `annotated_image`
- Script falls back to drawing its own boxes
- This is expected behavior

---

## File Structure

```
Physical_AI_Hackathon/
├── src/
│   ├── trash_sorting_detection.py  ← MAIN SCRIPT
│   ├── qarm_interface.py           ← Robot interface
│   └── arm_config.py               ← Arm configuration
├── run/
│   └── run_model.py                ← Your friend's code
├── .env.example                    ← Config template
├── .env                            ← Your config (create this)
├── requirements.txt                ← Dependencies
└── README.md
```

---

## Customization

### Change detection classes
Edit `ZONE_MAP` in `trash_sorting_detection.py`:
```python
ZONE_MAP = {
    "plastic bottles": "ZONE_A",
    "paper cup": "ZONE_B",
    "metal cans": "ZONE_C",
    # Add more classes...
}
```

### Change robot drop zones
Edit `ZONE_COORDS` in `trash_sorting_detection.py`:
```python
ZONE_COORDS = {
    "ZONE_A": {"x": 0.2, "y": 0.3, "z": 0.1},   # Adjust x,y,z
    "ZONE_B": {"x": -0.2, "y": 0.3, "z": 0.1},
    # ...
}
```

### Use different camera
Change in `.env`:
```
CAMERA_INDEX=1  # Try 0, 1, 2, etc.
```

---

## Next Steps

1. **Verify detection works first** → run with `USE_ROBOT=false`
2. **Calibrate zone coordinates** → test `p` key to pick/place
3. **Retrain model if needed** → use Roboflow to add more training data
4. **Automate picking** → modify code to automatically pick best detections

---

## Support

Check these first:
- Is `.env` file properly configured?
- Is robot connected and port is correct?
- Is Roboflow API key valid?
- Are dependencies installed? → `pip list`
- Is camera working? → try opening in Photo Booth or Zoom first
