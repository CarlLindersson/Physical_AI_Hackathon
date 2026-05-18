# QArm Mini Trash Sorting Demo

Computer vision guided robotic sorting with a Quanser QArm Mini, Roboflow-hosted object detection, and calibrated pick-and-place control.

Built during the Oxford Physical AI Hackathon, this project demonstrates how a small desktop robot arm can identify common trash items, classify them as paper, plastic, or metal, and place them into separate bins.

## Overview

The main demo script is [`run/run_model_w_objclass_demo.py`](run/run_model_w_objclass_demo.py). It combines:

- A live OpenCV camera feed for object localization.
- A Roboflow workflow using a fine-tuned SAM3-based trash detection model.
- Class-to-zone routing for paper, plastic, and metal waste streams.
- Camera-to-robot calibration for mapping image detections into QArm workspace coordinates.
- A QArm Mini control loop for centering, grasping, lifting, rotating, and dropping objects into bins.
- Manual keyboard controls for safe intervention and debugging.

## Demo Flow

```text
Camera frame
  -> Roboflow trash detection
  -> class + bounding box
  -> paper/plastic/metal zone mapping
  -> calibrated pixel-to-robot coordinates
  -> QArm Mini pick, lift, rotate, and drop
```

Default sorting map:

| Detection class | Bin zone | Category |
| --- | --- | --- |
| `plastic bottles` | `ZONE_A` | Plastic |
| `paper cup`, `paper crumble`, `paper box` | `ZONE_B` | Paper |
| `metal cans` | `ZONE_C` | Metal |

Additional demo classes such as `marker` and `pen` are mapped into existing zones in the script for testing.

## Repository Layout

```text
Physical_AI_Hackathon/
|-- run/
|   |-- run_model_w_objclass_demo.py   # Main hackathon demo
|   |-- run_model_w_objclass.py        # Development version of the object-class demo
|   |-- run_model.py                   # Manual QArm keyboard/camera control
|   |-- calibrate_qarm_camera.py       # Camera-to-robot workspace calibration
|   |-- calibrate_fisheye_intrinsics.py
|   |-- calibration_map.json
|   `-- intrinsic_calibration.json
|-- src/
|   |-- trash_sorting_detection.py     # Roboflow trash-sorting prototype
|   |-- qarm_interface.py
|   `-- camera_robot_calibration.py
|-- Atech x Quanser Qarm Mini Integration/
|   `-- src/main.cpp                   # Embedded integration workspace
|-- requirements.txt
|-- SETUP_GUIDE.md
`-- README.md
```

## Requirements

- Python 3 with `pip`
- Quanser QArm Mini Python stack (`hal`, `pal`, and QArm Mini drivers)
- Webcam or USB camera
- Roboflow API key and workflow access
- Python packages from [`requirements.txt`](requirements.txt)

Install the Python dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Configuration

Create a local `.env.local` file for Roboflow credentials. This file should stay out of Git.

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

The demo loads `.env.local` automatically when present. You can also pass the same values as command-line flags.

## Run The Demo

Connect the QArm Mini, make sure the camera is visible to OpenCV, then run:

```bash
python run/run_model_w_objclass_demo.py --vision-backend roboflow
```

If your camera is not OpenCV index `1`, choose another index:

```bash
python run/run_model_w_objclass_demo.py --camera-index 0 --vision-backend roboflow
```

Useful startup options:

```bash
python run/run_model_w_objclass_demo.py --roboflow-confidence 0.45
python run/run_model_w_objclass_demo.py --roboflow-classes "plastic bottles,paper cup,metal cans"
python run/run_model_w_objclass_demo.py --no-auto-repeat-grab
python run/run_model_w_objclass_demo.py --vision-backend none
```

## Controls

Click the Pygame control window before using the keyboard.

| Key | Action |
| --- | --- |
| Arrow keys | Move base and shoulder |
| `w` / `s` | Move wrist up/down |
| `p` / `o` | Close/open gripper |
| `h` | Return to wrist-down home pose |
| `c` | Toggle camera/object centering |
| `n` | Select next detected target |
| `g` | Run calibrated pick-and-place for the selected target |
| `x` | Cancel the current grab |
| `t` | Print vision backend statistics |
| `q` or `Esc` | Quit |

## Calibration

The automated pick-and-place path depends on camera-to-robot calibration:

- [`run/calibration_map.json`](run/calibration_map.json) maps image pixels to robot workspace coordinates.
- [`run/intrinsic_calibration.json`](run/intrinsic_calibration.json) stores optional fisheye camera intrinsics.
- [`run/calibrate_qarm_camera.py`](run/calibrate_qarm_camera.py) can be used to rebuild the workspace calibration.
- [`run/calibrate_fisheye_intrinsics.py`](run/calibrate_fisheye_intrinsics.py) can be used to rebuild camera intrinsics.

Zone drop positions can be tuned with flags such as:

```bash
python run/run_model_w_objclass_demo.py --zone-a-xyz 0.20,0.12,0.08
python run/run_model_w_objclass_demo.py --zone-b-xyz 0.20,0.00,0.08
python run/run_model_w_objclass_demo.py --zone-c-xyz 0.20,-0.12,0.08
```

## Development Notes

- `run/run_model_w_objclass_demo.py` is the polished demo entry point.
- `run/run_model_w_objclass.py` and `run/run_model_w_objclass_copy.py` contain development variants.
- `run/run_model.py` provides lower-level manual keyboard control with camera support.
- `src/trash_sorting_detection.py` is a Roboflow-first prototype for detection and sorting logic.
- The vision backend can fall back to a local PIT YOLO path when Roboflow is not configured.

## Safety

Keep the QArm workspace clear, start with low speeds and known poses, and be ready to cancel with `x` or quit with `q`/`Esc`. Re-run calibration whenever the camera, robot, bins, or table layout moves.
