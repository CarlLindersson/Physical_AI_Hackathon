# Quick Test: Roboflow Detection on Webcam

Test your Roboflow workflow on your webcam **before connecting the robot**.

## Quick Start

```bash
cd /Users/eshnanigans/Physical_AI_Hackathon
source .venv/bin/activate

# Install inference SDK if not already done
pip install inference-sdk

# Run webcam test
python src/test_roboflow_webcam.py
```

## What You'll See

- Live webcam feed
- Roboflow-annotated image with bounding boxes
- Object detection results printed to terminal
- FPS counter

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Show stats (total frames, detections, average objects) |

## Expected Output

```
============================================================
Roboflow Webcam Test
============================================================
API URL: https://serverless.roboflow.com
Workspace: eshitas-workspace-gas5f
Workflow: qarm-trash-ensemble-detection-1779037824337
Camera index: 0
============================================================

✓ Connected to Roboflow API
✓ Camera opened (640x480)

[Frame 1] Detected 1 object(s):
  - plastic bottles (0.85)
[Frame 2] Detected 2 object(s):
  - paper cup (0.92)
  - marker (0.71)
...
```

## Troubleshooting

### Camera not opening
```
✗ Failed to open camera 0
```
**Solutions:**
- Try different camera index:
  ```bash
  CAMERA_INDEX=1 python src/test_roboflow_webcam.py
  ```
- macOS: Settings → Security & Privacy → Camera (grant permission)
- Check camera is not in use by Zoom/Photo Booth

### Roboflow connection failed
```
✗ Failed to connect to Roboflow: ...
```
**Solutions:**
- Check internet connection
- Verify API key in `.env` is correct
- Make sure workspace name is right

### No detections showing
- Move object closer to camera
- Ensure good lighting
- Check object is in frame and visible
- If Roboflow returns `annotated_image`, bounding boxes will show

### Running slow / high latency
- This is normal — Roboflow API calls take ~300-500ms
- For real-time, consider local YOLO instead
- Or increase frame skip in main script

## Next Steps

Once detection works:
1. You can integrate with QArmMini in `trash_sorting_detection.py`
2. Or use this test to collect training data for custom model
3. Or just run detection without robot for analysis

## Configuration

Edit `.env` to change:
```
ROBOFLOW_API_KEY=your_key
ROBOFLOW_WORKSPACE_NAME=your_workspace
ROBOFLOW_WORKFLOW_ID=your_workflow_id
CAMERA_INDEX=0  # Try 0, 1, 2, etc.
```
