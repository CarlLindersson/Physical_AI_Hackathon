#-----------------------------------------------------------------------------#
#---------------- Keyboard + Camera Control - QArm Mini -----------------------#
#-----------------------------------------------------------------------------#

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import pygame
from hal.content.qarm_mini import QArmMiniFunctions
from pal.products.qarm_mini import QArmMini
from pal.utilities.timing import QTimer
from pathlib import Path


SAMPLE_RATE_HZ = 10.0
RUN_TIME_SECONDS = 300.0
JOINT_SPEED_RAD_PER_SEC = np.pi / 4
CENTER_MAX_SPEED_RAD_PER_SEC = np.pi / 10

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PANEL_WIDTH = 320
WINDOW_WIDTH = CAMERA_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = CAMERA_HEIGHT

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0

DEFAULT_ROBOFLOW_WORKSPACE = "eshitas-workspace-gas5f"
DEFAULT_ROBOFLOW_WORKFLOW_ID = "qarm-trash-ensemble-detection-1779037824337"
DEFAULT_ROBOFLOW_FALLBACK_WORKFLOW_ID = "qarm-trash-detection-1779035433585"

ZONE_MAP = {
    # Map detected class names to zones: paper -> ZONE_A, plastic -> ZONE_B, metal -> ZONE_C
    "paper": "ZONE_A",
    "paper cup": "ZONE_A",
    "paper crumble": "ZONE_A",
    "paper box": "ZONE_A",
    "plastic": "ZONE_B",
    "plastic bottles": "ZONE_B",
    "bottle": "ZONE_B",
    "metal": "ZONE_C",
    "metal cans": "ZONE_C",
    "can": "ZONE_C",
    "marker": "ZONE_D",
    "pen": "ZONE_D",
}

CENTER_TARGET_X_RATIO = 0.5
DEFAULT_CENTER_TARGET_Y_RATIO = 0.5
DEFAULT_CENTER_DEADBAND_PX = 30
DEFAULT_CENTER_BASE_SIGN = -1.0
DEFAULT_CENTER_SHOULDER_SIGN = 1.0

GRAB_APPROACH_STEP_M = 0.015
GRAB_MAX_APPROACH_STEPS = 8
GRAB_CLOSE_AREA_RATIO = 0.18
GRAB_CENTERED_FRAMES = 2
GRAB_CLOSE_TICKS = 12
GRIP_SETTLE_TICKS = 10
GRIP_POST_CLOSE_TICKS = 5
GRAB_LOWER_M = 0.035
GRAB_LOWER_STEPS = 3
GRAB_LIFT_M = 0.045
GRAB_LIFT_STEPS = 4
DEFAULT_GRAB_APPROACH_SIGN = 1.0
DEFAULT_GRAB_APPROACH_AXIS = "tool"
HOME_WRIST_DOWN_DEG = 0.0
DEFAULT_CALIBRATION_MAP = "run/calibration_map.json"
DEFAULT_PICK_APPROACH_M = 0.060
DEFAULT_PICK_Z_OFFSET_M = 0.000
DEFAULT_PICK_JOINT_SPEED_RAD_PER_SEC = np.pi / 5
DEFAULT_PICK_JOINT_TOLERANCE_RAD = np.deg2rad(1.0)
DEFAULT_PICK_HOME_SETTLE_TICKS = 5
DEFAULT_ZONE_APPROACH_M = 0.060
DEFAULT_ZONE_RELEASE_Z_OFFSET_M = 0.080
DEFAULT_ZONE_SIDE_OFFSET_M = 0.120
DEFAULT_ZONE_FURTHER_RIGHT_OFFSET_M = 0.200
DEFAULT_RELEASE_OPEN_TICKS = 3


def wrist_down_home_pose():
    pose = QArmMini.HOME_POSE.copy()
    pose[3] = np.deg2rad(HOME_WRIST_DOWN_DEG)
    np.clip(pose, QArmMini.LIMITS_MIN, QArmMini.LIMITS_MAX, out=pose)
    return pose


def parse_args():
    load_local_env()
    parser = argparse.ArgumentParser(description="QArm Mini keyboard control with live camera feed.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=int(os.environ.get("CAMERA_INDEX", "1")),
        help="OpenCV camera index to use. Defaults to CAMERA_INDEX or 1.",
    )
    parser.add_argument("--no-camera", action="store_true", help="Run keyboard control without opening a camera.")
    parser.add_argument(
        "--calibration-map",
        default=os.environ.get("CALIBRATION_MAP", DEFAULT_CALIBRATION_MAP),
        help="Camera-to-robot calibration JSON from run/calibrate_qarm_camera.py.",
    )
    parser.add_argument(
        "--calibration-transform",
        choices=("auto", "affine", "homography"),
        default=os.environ.get("CALIBRATION_TRANSFORM", "auto"),
        help="Transform to use from the calibration map. Auto prefers homography then affine.",
    )
    parser.add_argument(
        "--vision-backend",
        choices=("auto", "roboflow", "pit-yolo", "none"),
        default="auto",
        help="Object classification backend. Auto uses Roboflow when workflow/key are configured, otherwise PIT YOLO.",
    )
    parser.add_argument("--no-yolo", action="store_true", help="Deprecated alias for --vision-backend none.")
    parser.add_argument("--yolo-model", default=None, help="Optional path to a YOLO .pt or .engine model.")
    parser.add_argument("--yolo-confidence", type=float, default=0.3, help="YOLO confidence threshold.")
    parser.add_argument(
        "--yolo-classes",
        type=parse_yolo_classes,
        default=None,
        help="Comma-separated COCO class IDs to detect, or 'all'. Default: all.",
    )
    parser.add_argument(
        "--yolo-every-n-frames",
        type=int,
        default=3,
        help="Run YOLO every N camera frames and reuse the last detections between runs.",
    )
    parser.add_argument("--yolo-half", action="store_true", help="Use half precision for YOLO inference.")
    parser.add_argument(
        "--roboflow-api-url",
        default=os.environ.get("ROBOFLOW_API_URL", "https://serverless.roboflow.com"),
        help="Roboflow Inference API URL. Defaults to ROBOFLOW_API_URL or Roboflow serverless.",
    )
    parser.add_argument(
        "--roboflow-api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key. Defaults to ROBOFLOW_API_KEY.",
    )
    parser.add_argument(
        "--roboflow-model-id",
        default=os.environ.get("ROBOFLOW_MODEL_ID"),
        help="Deprecated alias for --roboflow-workflow-id. Defaults to ROBOFLOW_MODEL_ID.",
    )
    parser.add_argument(
        "--roboflow-workspace-name",
        default=os.environ.get("ROBOFLOW_WORKSPACE_NAME", DEFAULT_ROBOFLOW_WORKSPACE),
        help="Roboflow workspace name for run_workflow. Defaults to ROBOFLOW_WORKSPACE_NAME.",
    )
    parser.add_argument(
        "--roboflow-workflow-id",
        default=os.environ.get("ROBOFLOW_WORKFLOW_ID", DEFAULT_ROBOFLOW_WORKFLOW_ID),
        help="Roboflow workflow ID for run_workflow. Defaults to ROBOFLOW_WORKFLOW_ID.",
    )
    parser.add_argument(
        "--roboflow-fallback-workflow-id",
        default=os.environ.get("ROBOFLOW_FALLBACK_WORKFLOW_ID", DEFAULT_ROBOFLOW_FALLBACK_WORKFLOW_ID),
        help="Optional fallback workflow ID if the primary workflow request fails.",
    )
    parser.add_argument(
        "--roboflow-image-key",
        default=os.environ.get("ROBOFLOW_IMAGE_KEY", "image"),
        help="Workflow image input name. Defaults to ROBOFLOW_IMAGE_KEY or 'image'.",
    )
    parser.add_argument(
        "--roboflow-confidence",
        type=float,
        default=float(os.environ.get("ROBOFLOW_CONFIDENCE", "0.0")),
        help="Client-side Roboflow confidence threshold from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--roboflow-every-n-frames",
        type=int,
        default=int(os.environ.get("ROBOFLOW_EVERY_N_FRAMES", "1")),
        help="Submit every Nth camera frame to Roboflow and reuse the last detections between requests.",
    )
    parser.add_argument(
        "--roboflow-classes",
        type=parse_string_list,
        default=None,
        help="Optional comma-separated Roboflow class names to keep client-side.",
    )
    parser.add_argument(
        "--no-roboflow-cache",
        action="store_true",
        default=not parse_bool_env(os.environ.get("ROBOFLOW_USE_CACHE"), True),
        help="Disable Roboflow workflow caching.",
    )
    parser.add_argument(
        "--no-roboflow-annotated-image",
        action="store_true",
        help="Do not display the annotated image returned by the Roboflow workflow.",
    )
    parser.add_argument(
        "--center-target-y-ratio",
        type=float,
        default=float(os.environ.get("CENTER_TARGET_Y_RATIO", str(DEFAULT_CENTER_TARGET_Y_RATIO))),
        help="Auto-center target height in the image, 0.0 top to 1.0 bottom. Default: screen center.",
    )
    parser.add_argument(
        "--center-deadband-px",
        type=float,
        default=float(os.environ.get("CENTER_DEADBAND_PX", str(DEFAULT_CENTER_DEADBAND_PX))),
        help="Pixel error tolerated before auto-center moves the arm.",
    )
    parser.add_argument(
        "--center-max-speed",
        type=float,
        default=float(os.environ.get("CENTER_MAX_SPEED_RAD_PER_SEC", str(CENTER_MAX_SPEED_RAD_PER_SEC))),
        help="Maximum auto-center joint speed in rad/s.",
    )
    parser.add_argument(
        "--center-base-sign",
        type=float,
        default=float(os.environ.get("CENTER_BASE_SIGN", str(DEFAULT_CENTER_BASE_SIGN))),
        help="Flip this between 1 and -1 if horizontal auto-centering moves the wrong way.",
    )
    parser.add_argument(
        "--center-shoulder-sign",
        type=float,
        default=float(os.environ.get("CENTER_SHOULDER_SIGN", str(DEFAULT_CENTER_SHOULDER_SIGN))),
        help="Flip this between 1 and -1 if vertical auto-centering moves the wrong way.",
    )
    parser.add_argument(
        "--grab-approach-step",
        type=float,
        default=float(os.environ.get("GRAB_APPROACH_STEP_M", str(GRAB_APPROACH_STEP_M))),
        help="Cartesian approach step in meters for each grab update.",
    )
    parser.add_argument(
        "--grab-max-approach-steps",
        type=int,
        default=int(os.environ.get("GRAB_MAX_APPROACH_STEPS", str(GRAB_MAX_APPROACH_STEPS))),
        help="Maximum number of small approach steps before closing the gripper.",
    )
    parser.add_argument(
        "--grab-close-area-ratio",
        type=float,
        default=float(os.environ.get("GRAB_CLOSE_AREA_RATIO", str(GRAB_CLOSE_AREA_RATIO))),
        help="Close the gripper when the selected target box covers this frame area ratio.",
    )
    parser.add_argument(
        "--grab-centered-frames",
        type=int,
        default=int(os.environ.get("GRAB_CENTERED_FRAMES", str(GRAB_CENTERED_FRAMES))),
        help="Number of consecutive centered frames required before approaching.",
    )
    parser.add_argument(
        "--grab-center-deadband-px",
        type=float,
        default=parse_optional_float(os.environ.get("GRAB_CENTER_DEADBAND_PX")),
        help="Optional grab-specific centering deadband in pixels. Defaults to --center-deadband-px.",
    )
    parser.add_argument(
        "--grab-approach-sign",
        type=float,
        default=float(os.environ.get("GRAB_APPROACH_SIGN", str(DEFAULT_GRAB_APPROACH_SIGN))),
        help="Flip between 1 and -1 if grab approach moves away from the target.",
    )
    parser.add_argument(
        "--grab-approach-axis",
        choices=("tool", "radial", "z"),
        default=os.environ.get("GRAB_APPROACH_AXIS", DEFAULT_GRAB_APPROACH_AXIS),
        help="Primary direction for grab approach steps. 'tool' follows the gripper/camera forward axis.",
    )
    parser.add_argument(
        "--grab-lower-m",
        type=float,
        default=float(os.environ.get("GRAB_LOWER_M", str(GRAB_LOWER_M))),
        help="Extra downward distance in meters before closing the gripper.",
    )
    parser.add_argument(
        "--grab-lower-steps",
        type=int,
        default=int(os.environ.get("GRAB_LOWER_STEPS", str(GRAB_LOWER_STEPS))),
        help="Number of small IK steps used for the extra pre-close lowering.",
    )
    parser.add_argument(
        "--grab-lift-m",
        type=float,
        default=float(os.environ.get("GRAB_LIFT_M", str(GRAB_LIFT_M))),
        help="Lift distance in meters after closing the gripper.",
    )
    parser.add_argument(
        "--grab-lift-steps",
        type=int,
        default=int(os.environ.get("GRAB_LIFT_STEPS", str(GRAB_LIFT_STEPS))),
        help="Number of small IK steps used for the post-grab lift.",
    )
    parser.add_argument(
        "--grab-close-ticks",
        type=int,
        default=int(os.environ.get("GRAB_CLOSE_TICKS", str(GRAB_CLOSE_TICKS))),
        help="Control-loop ticks to hold the close command before lifting.",
    )
    parser.add_argument(
        "--grip-settle-ticks",
        type=int,
        default=int(os.environ.get("GRIP_SETTLE_TICKS", str(GRIP_SETTLE_TICKS))),
        help="Control-loop ticks to hold still at the pick pose before closing the gripper.",
    )
    parser.add_argument(
        "--grip-post-close-ticks",
        type=int,
        default=int(os.environ.get("GRIP_POST_CLOSE_TICKS", str(GRIP_POST_CLOSE_TICKS))),
        help="Control-loop ticks to hold still after closing before lifting.",
    )
    parser.add_argument(
        "--pick-approach-m",
        type=float,
        default=float(os.environ.get("PICK_APPROACH_M", str(DEFAULT_PICK_APPROACH_M))),
        help="Height above calibrated object/table point before lowering to grip.",
    )
    parser.add_argument(
        "--pick-z-offset-m",
        type=float,
        default=float(os.environ.get("PICK_Z_OFFSET_M", str(DEFAULT_PICK_Z_OFFSET_M))),
        help="Z offset added to calibrated table Z for the final grip pose.",
    )
    parser.add_argument(
        "--pick-joint-speed",
        type=float,
        default=float(os.environ.get("PICK_JOINT_SPEED_RAD_PER_SEC", str(DEFAULT_PICK_JOINT_SPEED_RAD_PER_SEC))),
        help="Joint-space speed limit for calibrated pick/place motions.",
    )
    parser.add_argument(
        "--pick-joint-tolerance",
        type=float,
        default=float(os.environ.get("PICK_JOINT_TOLERANCE_RAD", str(DEFAULT_PICK_JOINT_TOLERANCE_RAD))),
        help="Joint-space tolerance used to consider a pick/place waypoint reached.",
    )
    parser.add_argument(
        "--pick-home-settle-ticks",
        type=int,
        default=int(os.environ.get("PICK_HOME_SETTLE_TICKS", str(DEFAULT_PICK_HOME_SETTLE_TICKS))),
        help="Control-loop ticks to wait at wrist-down home before reading the first detection.",
    )
    parser.add_argument(
        "--zone-approach-m",
        type=float,
        default=float(os.environ.get("ZONE_APPROACH_M", str(DEFAULT_ZONE_APPROACH_M))),
        help="Height above a zone release point before lowering to release.",
    )
    parser.add_argument(
        "--zone-release-z-offset-m",
        type=float,
        default=float(os.environ.get("ZONE_RELEASE_Z_OFFSET_M", str(DEFAULT_ZONE_RELEASE_Z_OFFSET_M))),
        help="Z offset above calibrated table Z for zone release poses.",
    )
    parser.add_argument(
        "--zone-side-offset-m",
        type=float,
        default=float(os.environ.get("ZONE_SIDE_OFFSET_M", str(DEFAULT_ZONE_SIDE_OFFSET_M))),
        help="Default left/right zone Y offset from the calibrated workspace center.",
    )
    parser.add_argument(
        "--zone-further-right-offset-m",
        type=float,
        default=float(os.environ.get("ZONE_FURTHER_RIGHT_OFFSET_M", str(DEFAULT_ZONE_FURTHER_RIGHT_OFFSET_M))),
        help="Default Zone C Y offset from the calibrated workspace center.",
    )
    parser.add_argument(
        "--zone-a-xyz",
        type=parse_optional_xyz,
        default=parse_optional_xyz(os.environ.get("ZONE_A_XYZ")),
        help="Override Zone A release XYZ in meters, e.g. 0.20,0.12,0.08.",
    )
    parser.add_argument(
        "--zones-file",
        default=os.environ.get("ZONES_FILE", "run/zones.json"),
        help="Path to read/write taught zone poses (JSON).",
    )
    parser.add_argument(
        "--auto-pick",
        action="store_true",
        help="Enable auto pick loop: automatically pick first detected object and place into its zone.",
    )
    parser.add_argument(
        "--zone-b-xyz",
        type=parse_optional_xyz,
        default=parse_optional_xyz(os.environ.get("ZONE_B_XYZ")),
        help="Override Zone B release XYZ in meters.",
    )
    parser.add_argument(
        "--zone-c-xyz",
        type=parse_optional_xyz,
        default=parse_optional_xyz(os.environ.get("ZONE_C_XYZ")),
        help="Override Zone C release XYZ in meters.",
    )
    parser.add_argument(
        "--release-open-ticks",
        type=int,
        default=int(os.environ.get("RELEASE_OPEN_TICKS", str(DEFAULT_RELEASE_OPEN_TICKS))),
        help="Control-loop ticks to hold the gripper open after reaching a zone.",
    )

    args = parser.parse_args()
    if not args.roboflow_workflow_id and args.roboflow_model_id:
        args.roboflow_workflow_id = args.roboflow_model_id
    return args


def parse_bool_env(value, default=False):
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def parse_optional_float(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def parse_optional_xyz(value):
    if value is None or str(value).strip() == "":
        return None

    parts = [part.strip() for part in str(value).replace(";", ",").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected XYZ as three comma-separated numbers.")

    try:
        return np.array([float(part) for part in parts], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected XYZ as three comma-separated numbers.") from exc


def load_local_env():
    project_root = Path(__file__).resolve().parents[1]
    candidates = (
        Path.cwd() / ".env.local",
        Path.cwd() / ".env",
        project_root / ".env.local",
        project_root / ".env",
    )

    seen = set()
    for env_file in candidates:
        env_file = env_file.resolve()
        if env_file in seen:
            continue
        seen.add(env_file)

        if not env_file.exists():
            continue

        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def parse_string_list(value):
    value = value.strip()
    if value.lower() in ("", "all", "none"):
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_yolo_classes(value):
    value = value.strip().lower()
    if value in ("", "all", "none"):
        return None

    try:
        return [int(class_id.strip()) for class_id in value.split(",") if class_id.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--yolo-classes must be 'all' or comma-separated integer class IDs."
        ) from exc


def print_controls(camera_index, camera_enabled):
    print("\nQArm Mini keyboard control")
    print("=" * 32)
    if camera_enabled:
        print(f"  Camera      OpenCV index {camera_index}")
    else:
        print("  Camera      disabled")
    print("  Up/Down     shoulder up/down")
    print("  Left/Right  base left/right")
    print("  A/D         elbow in/out")
    print("  w/s         wrist up/down")
    print("  p           close gripper")
    print("  o           open gripper")
    print("  h           home position with wrist down")
    print("  1/2/3       record zone A/B/C (saves run/zones.json)")
    print("  Shift+1/2/3 overwrite a saved zone")
    print("  c           toggle camera/object centering")
    print("  n           center next detected target")
    print("  g           calibrated pick/place first target")
    print("  x           cancel grab")
    print("  t           show vision stats")
    print("  q or Esc    quit")
    print("\nClick/focus the pygame keyboard window before driving the arm.\n")


def load_vision_backend(args):
    if args.no_yolo or args.no_camera or args.vision_backend == "none":
        return None

    backend = args.vision_backend
    if backend == "auto":
        if args.roboflow_api_key and args.roboflow_workspace_name and args.roboflow_workflow_id:
            backend = "roboflow"
            print("Vision auto selected Roboflow because API key and workflow are configured.")
        else:
            backend = "pit-yolo"
            print("Vision auto selected PIT YOLO because Roboflow is not fully configured.")
            print("         Use --vision-backend roboflow after setting ROBOFLOW_API_KEY,")
            print("         ROBOFLOW_WORKSPACE_NAME, and ROBOFLOW_WORKFLOW_ID.")

    if backend == "roboflow":
        return load_roboflow(args)

    return load_pit_yolo(args)


def load_zones_file(path):
    try:
        p = Path(path)
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_zones_file(path, data):
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return True
    except Exception as exc:
        print(f"Warning: could not save zones file: {exc}")
        return False


def load_roboflow(args):
    missing = []
    if not args.roboflow_workspace_name:
        missing.append("Set ROBOFLOW_WORKSPACE_NAME")

    if not args.roboflow_workflow_id:
        missing.append("Set ROBOFLOW_WORKFLOW_ID")

    if not args.roboflow_api_key:
        missing.append("Set ROBOFLOW_API_KEY")

    if missing:
        print("Warning: Roboflow backend selected but it is not fully configured.")
        for item in missing:
            print(f"         {item}")
        print("         Or pass --roboflow-workspace-name, --roboflow-workflow-id, and --roboflow-api-key.")
        return InactiveVision("roboflow not configured", missing)

    try:
        from inference_sdk import InferenceHTTPClient
    except Exception as exc:
        print(f"Warning: could not import Roboflow inference-sdk: {exc}")
        print("         Install it with: python -m pip install inference-sdk")
        return InactiveVision(
            "roboflow sdk missing",
            ["Install inference-sdk", "python -m pip install inference-sdk"],
        )

    try:
        client = InferenceHTTPClient(
            api_url=args.roboflow_api_url,
            api_key=args.roboflow_api_key,
        )
    except Exception as exc:
        print(f"Warning: could not configure Roboflow client: {exc}")
        return InactiveVision("roboflow config error", [short_text(exc)])

    print(
        "Roboflow workflow configured: "
        f"{args.roboflow_workspace_name}/{args.roboflow_workflow_id}"
    )
    print(f"Roboflow API URL: {args.roboflow_api_url}")
    return RoboflowClassifier(
        client=client,
        workspace_name=args.roboflow_workspace_name,
        workflow_id=args.roboflow_workflow_id,
        fallback_workflow_id=args.roboflow_fallback_workflow_id,
        image_key=args.roboflow_image_key,
        use_cache=not args.no_roboflow_cache,
        use_annotated_image=not args.no_roboflow_annotated_image,
        class_filter=args.roboflow_classes,
        confidence=args.roboflow_confidence,
        every_n_frames=max(1, args.roboflow_every_n_frames),
    )


def load_pit_yolo(args):
    try:
        patch_ultralytics_letterbox_if_needed()
        from pit.YOLO.nets import YOLOv8
    except Exception as exc:
        print(f"Warning: could not import PIT YOLO support: {exc}")
        return InactiveVision("pit-yolo unavailable", [short_text(exc)])

    try:
        model = YOLOv8(
            imageWidth=CAMERA_WIDTH,
            imageHeight=CAMERA_HEIGHT,
            modelPath=args.yolo_model,
            convert_tensorrt=False,
        )
    except Exception as exc:
        print(f"Warning: could not load YOLO model: {exc}")
        return InactiveVision("pit-yolo load error", [short_text(exc)])

    return YoloClassifier(
        model=model,
        classes=args.yolo_classes,
        confidence=args.yolo_confidence,
        every_n_frames=max(1, args.yolo_every_n_frames),
        half=args.yolo_half,
    )


def short_text(value, max_length=42):
    text = str(value).replace("\n", " ")
    if len(text) <= max_length:
        return text
    return f"{text[:max_length - 3]}..."


def patch_ultralytics_letterbox_if_needed():
    test_image = np.zeros((8, 8, 3), dtype=np.uint8)
    try:
        cv2.copyMakeBorder(test_image, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
        return
    except Exception:
        pass

    from ultralytics.data.augment import LetterBox

    def apply_image_without_cv2(self, labels, params):
        img = np.asarray(labels["img"])
        shape = img.shape[:2]
        new_unpad = params["new_unpad"]

        if shape[::-1] != new_unpad:
            img = resize_image(img, new_unpad)
            if img.ndim == 2:
                img = img[..., None]

        top, bottom = params["top"], params["bottom"]
        left, right = params["left"], params["right"]
        if img.ndim == 2:
            pad_width = ((top, bottom), (left, right))
        else:
            pad_width = ((top, bottom), (left, right), (0, 0))

        labels["img"] = np.ascontiguousarray(
            np.pad(img, pad_width, mode="constant", constant_values=self.padding_value)
        )
        labels["resized_shape"] = params["new_shape"]
        return labels

    LetterBox.apply_image = apply_image_without_cv2
    print("Warning: OpenCV cannot process NumPy arrays in this environment.")
    print("         Patched Ultralytics letterbox with a Pillow/NumPy fallback.")


def resize_image(frame, size):
    image = Image.fromarray(np.asarray(frame))
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized))


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        print(f"Warning: could not open camera index {camera_index}. Continuing without video.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, SAMPLE_RATE_HZ)
    print(f"Camera opened on index {camera_index}.")
    return cap


def read_camera_frame(cap):
    if cap is None:
        return None

    ok, frame = cap.read()
    if not ok:
        return None

    frame = np.ascontiguousarray(np.asarray(frame))
    if frame.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
        frame = resize_image(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

    return frame


def frame_to_surface(frame):
    if frame is None:
        return None

    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    return pygame.image.frombuffer(rgb_frame.tobytes(), (CAMERA_WIDTH, CAMERA_HEIGHT), "RGB")


def draw_text(surface, font, text, pos, color=(190, 196, 205)):
    surface.blit(font.render(text, True, color), pos)


def draw_camera_feed(screen, font, camera_surface, vision, center_target=None, selected_target_index=0):
    camera_rect = pygame.Rect(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)

    if camera_surface is None:
        pygame.draw.rect(screen, (12, 14, 18), camera_rect)
        pygame.draw.rect(screen, (70, 76, 86), camera_rect, 1)
        draw_text(
            screen,
            font,
            "No camera feed",
            (CAMERA_WIDTH // 2 - 70, CAMERA_HEIGHT // 2 - 12),
            (220, 220, 220),
        )
        return

    screen.blit(camera_surface, camera_rect)
    pygame.draw.rect(screen, (70, 76, 86), camera_rect, 1)
    draw_text(screen, font, "LIVE CAMERA", (12, 12), (255, 255, 255))
    if center_target is not None:
        draw_center_target(screen, center_target)
    if vision:
        draw_detection_overlays(screen, font, vision.detections, selected_target_index)


def draw_center_target(screen, center_target):
    x, y = center_target
    color = (255, 255, 255)
    pygame.draw.circle(screen, color, (int(x), int(y)), 16, 1)
    pygame.draw.line(screen, color, (int(x) - 24, int(y)), (int(x) + 24, int(y)), 1)
    pygame.draw.line(screen, color, (int(x), int(y) - 24), (int(x), int(y) + 24), 1)


def draw_keyboard_window(
    font,
    small_font,
    gripper_cmd,
    camera_surface,
    vision,
    auto_center_enabled=False,
    center_status="center off",
    center_target=None,
    selected_target_index=0,
    grab_status="grab idle",
):
    screen = pygame.display.get_surface()
    screen.fill((20, 22, 26))
    draw_camera_feed(
        screen,
        small_font,
        camera_surface,
        vision,
        center_target if auto_center_enabled else None,
        selected_target_index,
    )

    panel_x = CAMERA_WIDTH + 22
    y = 18

    draw_text(screen, font, "QArm Mini Keyboard Control", (panel_x, y), (240, 240, 240))
    y += 34
    draw_text(
        screen,
        small_font,
        f"Gripper: {'closed' if gripper_cmd == GRIPPER_CLOSED else 'open'}",
        (panel_x, y),
        (240, 240, 240),
    )
    y += 38

    draw_text(screen, font, "Controls", (panel_x, y), (240, 240, 240))
    y += 26
    for item in (
        "Arrows move base/shoulder",
        "w/s wrist, p/o gripper",
        "h home wrist down, c center",
        "n next target, g pick/place",
        "x cancel grab",
        "t stats, q/Esc quits",
    ):
        draw_text(screen, small_font, item, (panel_x, y))
        y += 21
    y += 8

    draw_text(screen, font, "Vision", (panel_x, y), (240, 240, 240))
    y += 26
    draw_text(screen, small_font, vision.status if vision else "off", (panel_x, y))
    y += 22

    if vision:
        for line in vision.summary_lines(max_lines=1):
            draw_text(screen, small_font, line, (panel_x, y))
            y += 22

    y += 8
    draw_text(screen, font, "Center", (panel_x, y), (240, 240, 240))
    y += 26
    center_color = (120, 220, 160) if auto_center_enabled else (190, 196, 205)
    draw_text(screen, small_font, "on" if auto_center_enabled else "off", (panel_x, y), center_color)
    y += 22
    draw_text(screen, small_font, center_status, (panel_x, y))
    y += 22
    draw_text(screen, small_font, f"Grab: {grab_status}", (panel_x, y), (190, 196, 205))
    y += 26

    draw_text(screen, font, "Targets", (panel_x, y), (240, 240, 240))
    y += 26
    target_lines = target_list_lines(vision, selected_target_index, max_lines=2)
    if not target_lines:
        draw_text(screen, small_font, "No boxed targets", (panel_x, y))
    else:
        for line, is_selected in target_lines:
            color = (255, 255, 255) if is_selected else (190, 196, 205)
            draw_text(screen, small_font, line, (panel_x, y), color)
            y += 21

    pygame.display.flip()


def draw_detection_overlays(screen, font, detections, selected_target_index=0):
    targets = boxed_detections(detections)
    selected_index = normalize_target_index(selected_target_index, targets)

    for index, det in enumerate(targets):
        is_selected = index == selected_index

        x1, y1, x2, y2 = det["box"]
        x1 = int(np.clip(x1, 0, CAMERA_WIDTH - 1))
        y1 = int(np.clip(y1, 0, CAMERA_HEIGHT - 1))
        x2 = int(np.clip(x2, 0, CAMERA_WIDTH - 1))
        y2 = int(np.clip(y2, 0, CAMERA_HEIGHT - 1))
        color = detection_color(det["class_id"])
        zone = det.get("zone")
        if zone:
            label = f"#{index + 1} {det['name']} {det['confidence']:.0%} -> {zone}"
        else:
            label = f"#{index + 1} {det['name']} {det['confidence']:.0%}"

        rect = pygame.Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        pygame.draw.rect(screen, color, rect, 4 if is_selected else 2)
        if is_selected:
            pygame.draw.rect(screen, (255, 255, 255), rect.inflate(6, 6), 2)
        label_surface = font.render(label, True, (12, 14, 18))
        label_rect = label_surface.get_rect()
        label_rect.topleft = (x1, max(0, y1 - label_rect.height - 6))
        bg_rect = label_rect.inflate(8, 4)
        pygame.draw.rect(screen, color, bg_rect)
        screen.blit(label_surface, label_rect)


def handle_keydown_events(joint_cmd, gripper_cmd, vision=None):
    should_quit = False
    toggle_auto_center = False
    next_center_target = False
    start_grab = False
    cancel_grab = False
    home_requested = False
    record_zone = None

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            should_quit = True
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                should_quit = True
            elif event.key == pygame.K_p:
                gripper_cmd = GRIPPER_CLOSED
                print("Gripper: closed")
            elif event.key == pygame.K_o:
                gripper_cmd = GRIPPER_OPEN
                print("Gripper: open")
            elif event.key == pygame.K_h:
                joint_cmd[:] = wrist_down_home_pose()
                home_requested = True
                print("Moving to home position with wrist down")
            elif event.key == pygame.K_c:
                toggle_auto_center = True
            elif event.key == pygame.K_n:
                next_center_target = True
            elif event.key == pygame.K_g:
                start_grab = True
            elif event.key == pygame.K_x:
                cancel_grab = True
            elif event.key == pygame.K_t and vision is not None and hasattr(vision, "print_stats"):
                vision.print_stats()
            elif event.key == pygame.K_1:
                record_zone = "ZONE_A"
                print("Requested record: ZONE_A")
            elif event.key == pygame.K_2:
                record_zone = "ZONE_B"
                print("Requested record: ZONE_B")
            elif event.key == pygame.K_3:
                record_zone = "ZONE_C"
                print("Requested record: ZONE_C")

    return (
        should_quit,
        gripper_cmd,
        toggle_auto_center,
        next_center_target,
        start_grab,
        cancel_grab,
        home_requested,
        record_zone,
    )


def apply_arrow_key_motion(joint_cmd, timestep):
    keys = pygame.key.get_pressed()
    step = JOINT_SPEED_RAD_PER_SEC * timestep

    joint_cmd[0] += (int(keys[pygame.K_LEFT]) - int(keys[pygame.K_RIGHT])) * step
    joint_cmd[1] += (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])) * step
    joint_cmd[2] += (int(keys[pygame.K_d]) - int(keys[pygame.K_a])) * step
    joint_cmd[3] += (int(keys[pygame.K_w]) - int(keys[pygame.K_s])) * step

    np.clip(joint_cmd, QArmMini.LIMITS_MIN, QArmMini.LIMITS_MAX, out=joint_cmd)


def apply_auto_center_motion(joint_cmd, vision, timestep, args, selected_target_index):
    detection, targets, target_index = selected_target_detection(vision, selected_target_index)
    if detection is None:
        return "waiting for targets", 0

    box = detection["box"]
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    target_x, target_y = center_target(args)

    error_x = center_x - target_x
    error_y = center_y - target_y
    move_x = 0.0 if abs(error_x) <= args.center_deadband_px else error_x
    move_y = 0.0 if abs(error_y) <= args.center_deadband_px else error_y

    if move_x == 0.0 and move_y == 0.0:
        return f"#{target_index + 1}/{len(targets)} centered {short_text(detection['name'], 18)}", target_index

    normalized_x = np.clip(move_x / (CAMERA_WIDTH / 2), -1.0, 1.0)
    normalized_y = np.clip(move_y / (CAMERA_HEIGHT / 2), -1.0, 1.0)
    max_step = args.center_max_speed * timestep

    joint_cmd[0] += args.center_base_sign * normalized_x * max_step
    joint_cmd[1] += args.center_shoulder_sign * normalized_y * max_step
    np.clip(joint_cmd, QArmMini.LIMITS_MIN, QArmMini.LIMITS_MAX, out=joint_cmd)

    name = short_text(detection["name"], 16)
    return f"#{target_index + 1}/{len(targets)} {name}: dx {error_x:.0f}, dy {error_y:.0f}", target_index


def select_next_target(vision, selected_target_index):
    targets = target_detections(vision)
    if not targets:
        return 0, "no targets to select"

    selected_target_index = normalize_target_index(selected_target_index, targets)
    selected_target_index = (selected_target_index + 1) % len(targets)
    detection = targets[selected_target_index]
    return selected_target_index, selected_target_status(detection, targets, selected_target_index)


def selected_target_detection(vision, selected_target_index):
    targets = target_detections(vision)
    if not targets:
        return None, targets, 0

    target_index = normalize_target_index(selected_target_index, targets)
    return targets[target_index], targets, target_index


def first_target_detection(vision):
    targets = target_detections(vision)
    if not targets:
        return None
    return targets[0]


def selected_target_status(detection, targets, target_index):
    name = short_text(detection["name"], 18)
    return f"#{target_index + 1}/{len(targets)} {name}"


def target_detections(vision):
    if vision is None:
        return []
    return boxed_detections(getattr(vision, "detections", []))


def boxed_detections(detections):
    return [
        det
        for det in (detections or [])
        if isinstance(det, dict) and det.get("box") is not None
    ]


def normalize_target_index(selected_target_index, targets):
    if not targets:
        return 0
    return selected_target_index % len(targets)


def target_list_lines(vision, selected_target_index, max_lines=4):
    targets = target_detections(vision)
    if not targets:
        return []

    selected_index = normalize_target_index(selected_target_index, targets)
    indexes = list(range(min(max_lines, len(targets))))
    if selected_index not in indexes:
        indexes[-1] = selected_index

    lines = []
    for index in indexes:
        det = targets[index]
        marker = ">" if index == selected_index else " "
        zone = det.get("zone", "")
        zone_text = f" -> {zone}" if zone else ""
        lines.append(
            (
                f"{marker}#{index + 1} {short_text(det['name'], 14)} {det['confidence']:.0%}{zone_text}",
                index == selected_index,
            )
        )
    return lines


def center_target(args):
    target_y_ratio = float(np.clip(args.center_target_y_ratio, 0.0, 1.0))
    return CAMERA_WIDTH * CENTER_TARGET_X_RATIO, CAMERA_HEIGHT * target_y_ratio


def target_center_error(detection, args):
    box = detection["box"]
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    target_x, target_y = center_target(args)
    return center_x - target_x, center_y - target_y


def grab_center_deadband(args):
    if args.grab_center_deadband_px is not None:
        return args.grab_center_deadband_px
    return args.center_deadband_px


def is_target_centered_for_grab(detection, args):
    error_x, error_y = target_center_error(detection, args)
    deadband = grab_center_deadband(args)
    return abs(error_x) <= deadband and abs(error_y) <= deadband


def detection_area_ratio(detection):
    box = detection.get("box")
    if box is None:
        return 0.0
    x1, y1, x2, y2 = box
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / float(CAMERA_WIDTH * CAMERA_HEIGHT)


def move_end_effector_delta(joint_cmd, arm_math, delta_pos):
    current_pos, _, gamma = arm_math.forward_kinematics(joint_cmd)
    target_pos = np.asarray(current_pos, dtype=np.float64) + np.asarray(delta_pos, dtype=np.float64)
    try:
        _, _, num_solutions, theta_opt = arm_math.inverse_kinematics(target_pos, gamma, joint_cmd)
    except Exception as exc:
        return False, f"IK error: {short_text(exc, 28)}"
    theta_opt = np.asarray(theta_opt, dtype=np.float64).reshape(-1)

    if num_solutions <= 0 or theta_opt.size < 4:
        return False, f"no IK solution at {format_position(target_pos)}"
    if not np.isfinite(theta_opt[:4]).all():
        return False, "IK returned non-finite joint values"

    next_joint_cmd = theta_opt[:4]
    if np.any(next_joint_cmd < QArmMini.LIMITS_MIN) or np.any(next_joint_cmd > QArmMini.LIMITS_MAX):
        return False, "IK solution outside joint limits"

    joint_cmd[:] = next_joint_cmd
    np.clip(joint_cmd, QArmMini.LIMITS_MIN, QArmMini.LIMITS_MAX, out=joint_cmd)
    return True, ""


def format_position(position):
    return "(" + ", ".join(f"{float(value):.3f}" for value in position[:3]) + ")"


def move_with_delta_candidates(joint_cmd, arm_math, candidates):
    last_message = "no movement candidates"
    for label, delta_pos in candidates:
        if np.linalg.norm(delta_pos) <= 0:
            continue
        ok, message = move_end_effector_delta(joint_cmd, arm_math, delta_pos)
        if ok:
            return True, "", label
        last_message = f"{label}: {message}"
    return False, last_message, ""


def approach_delta_candidates(joint_cmd, arm_math, args):
    primary_axis = axis_from_current_pose(joint_cmd, arm_math, args.grab_approach_axis)
    fallback_axes = {
        "tool": axis_from_current_pose(joint_cmd, arm_math, "tool"),
        "radial": axis_from_current_pose(joint_cmd, arm_math, "radial"),
        "z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
    }
    sign = 1.0 if args.grab_approach_sign >= 0 else -1.0
    step = max(0.0, args.grab_approach_step)
    scales = (1.0, 0.5, 0.25)

    candidates = []
    add_axis_candidates(candidates, args.grab_approach_axis, primary_axis, sign, step, scales)
    for axis_name, axis in fallback_axes.items():
        if axis_name != args.grab_approach_axis:
            add_axis_candidates(candidates, axis_name, axis, sign, step, scales)

    return candidates


def add_axis_candidates(candidates, axis_name, axis, sign, step, scales):
    axis = normalize_vector(axis)
    if axis is None:
        return

    for direction_sign, direction_name in ((sign, "primary"), (-sign, "opposite")):
        for scale in scales:
            label = f"{axis_name} {direction_name} {step * scale:.3f}m"
            delta = axis * direction_sign * step * scale
            candidates.append((label, delta))


def axis_from_current_pose(joint_cmd, arm_math, axis_name):
    current_pos, rotation, _ = arm_math.forward_kinematics(joint_cmd)
    if axis_name == "tool":
        return np.asarray(rotation[:, 2], dtype=np.float64)
    if axis_name == "z":
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return radial_axis_from_position(current_pos, joint_cmd)


def radial_axis_from_position(current_pos, joint_cmd):
    direction_xy = np.asarray(current_pos[:2], dtype=np.float64)
    norm = np.linalg.norm(direction_xy)
    if norm < 1e-6:
        return np.array([np.cos(joint_cmd[0]), np.sin(joint_cmd[0]), 0.0], dtype=np.float64)
    direction_xy = direction_xy / norm
    return np.array([direction_xy[0], direction_xy[1], 0.0], dtype=np.float64)


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < 1e-9 or not np.isfinite(norm):
        return None
    return vector / norm


def resolve_repo_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def load_calibration_map(args):
    path = resolve_repo_path(args.calibration_map)
    if not path.exists():
        print(f"Warning: calibration map not found: {path}")
        print("         Run run/calibrate_qarm_camera.py before using g pick/place.")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Warning: could not load calibration map {path}: {exc}")
        return None

    transform = select_calibration_transform(data, args.calibration_transform)
    if transform is None:
        print(f"Warning: calibration map has no usable pixel-to-robot transform: {path}")
        return None

    workspace_points = calibration_workspace_points(data)
    table_z = calibration_table_z(data, workspace_points)
    center_xy, span_y = calibration_workspace_center(workspace_points)
    if table_z is None or center_xy is None:
        print(f"Warning: calibration map is missing workspace XYZ points: {path}")
        return None

    print(
        "Calibration loaded: "
        f"{path} ({transform['kind']}, table z {table_z:.3f} m)"
    )
    return {
        "path": path,
        "transform": transform,
        "table_z": table_z,
        "center_xy": center_xy,
        "span_y": span_y,
    }


def select_calibration_transform(data, requested_kind):
    transforms = data.get("transforms", {}) if isinstance(data, dict) else {}
    ordered_kinds = ("homography", "affine") if requested_kind == "auto" else (requested_kind,)
    for kind in ordered_kinds:
        key = f"pixel_to_robot_xy_{kind}"
        if key not in transforms:
            continue
        matrix = np.asarray(transforms[key], dtype=np.float64)
        if kind == "homography" and matrix.shape == (3, 3):
            return {"kind": kind, "matrix": matrix}
        if kind == "affine" and matrix.shape == (2, 3):
            return {"kind": kind, "matrix": matrix}
    return None


def calibration_workspace_points(data):
    points = []
    for record in data.get("points", []):
        workspace = record.get("workspace_m")
        if workspace is None:
            continue
        workspace = np.asarray(workspace, dtype=np.float64).reshape(-1)
        if workspace.size >= 3 and np.isfinite(workspace[:3]).all():
            points.append(workspace[:3])
    return points


def calibration_table_z(data, workspace_points):
    z_value = data.get("transforms", {}).get("table_z_mean_m")
    if z_value is not None:
        return float(z_value)
    if workspace_points:
        return float(np.mean([point[2] for point in workspace_points]))
    return None


def calibration_workspace_center(workspace_points):
    if not workspace_points:
        return None, 0.0
    xy_points = np.asarray([point[:2] for point in workspace_points], dtype=np.float64)
    center_xy = np.mean(xy_points, axis=0)
    span_y = float(np.ptp(xy_points[:, 1]) / 2.0) if len(xy_points) > 1 else 0.0
    return center_xy, span_y


def detection_box_center(detection):
    box = detection.get("box")
    if box is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in box]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float64)


def pixel_to_robot_xyz(pixel_xy, calibration, args):
    if calibration is None:
        raise ValueError("No calibration map loaded.")

    transform = calibration["transform"]
    u, v = [float(value) for value in pixel_xy[:2]]
    if transform["kind"] == "homography":
        homogeneous = transform["matrix"] @ np.array([u, v, 1.0], dtype=np.float64)
        if abs(homogeneous[2]) < 1e-9:
            raise ValueError("Calibration homography produced an invalid point.")
        xy = homogeneous[:2] / homogeneous[2]
    else:
        xy = transform["matrix"] @ np.array([u, v, 1.0], dtype=np.float64)

    z = calibration["table_z"] + args.pick_z_offset_m
    return np.array([xy[0], xy[1], z], dtype=np.float64)


def zone_release_xyz(zone, calibration, args):
    zone = zone if zone in ("ZONE_A", "ZONE_B", "ZONE_C") else "ZONE_B"
    overrides = {
        "ZONE_A": args.zone_a_xyz,
        "ZONE_B": args.zone_b_xyz,
        "ZONE_C": args.zone_c_xyz,
    }
    if overrides[zone] is not None:
        return np.asarray(overrides[zone], dtype=np.float64).copy(), zone

    center_x, center_y = calibration["center_xy"]
    side_offset = max(abs(args.zone_side_offset_m), calibration.get("span_y", 0.0))
    further_right_offset = max(abs(args.zone_further_right_offset_m), side_offset * 1.6)
    release_z = calibration["table_z"] + args.zone_release_z_offset_m

    if zone == "ZONE_A":
        y = center_y + side_offset
    elif zone == "ZONE_C":
        y = center_y - further_right_offset
    else:
        y = center_y - side_offset

    return np.array([center_x, y, release_z], dtype=np.float64), zone


def solve_ik_for_position(joint_cmd, arm_math, xyz, gamma):
    try:
        _, _, num_solutions, theta_opt = arm_math.inverse_kinematics(
            np.asarray(xyz, dtype=np.float64),
            gamma,
            joint_cmd,
        )
    except Exception as exc:
        return False, None, f"IK error: {short_text(exc, 40)}"

    theta_opt = np.asarray(theta_opt, dtype=np.float64).reshape(-1)
    if num_solutions <= 0 or theta_opt.size < 4:
        return False, None, f"no IK solution for {format_position(xyz)}"
    if not np.isfinite(theta_opt[:4]).all():
        return False, None, "IK returned non-finite joints"

    target = theta_opt[:4]
    if np.any(target < QArmMini.LIMITS_MIN) or np.any(target > QArmMini.LIMITS_MAX):
        return False, None, "IK solution outside joint limits"
    return True, target, ""


def step_joints_toward(joint_cmd, target, timestep, args):
    target = np.asarray(target, dtype=np.float64)
    delta = target - joint_cmd
    if np.max(np.abs(delta)) <= args.pick_joint_tolerance:
        joint_cmd[:] = target
        return True

    max_step = max(0.0, args.pick_joint_speed) * timestep
    if max_step <= 0.0:
        joint_cmd[:] = target
        return True

    scale = min(1.0, max_step / max(np.max(np.abs(delta)), 1e-9))
    joint_cmd[:] = joint_cmd + delta * scale
    np.clip(joint_cmd, QArmMini.LIMITS_MIN, QArmMini.LIMITS_MAX, out=joint_cmd)
    return False


class GrabController:
    def __init__(self, calibration):
        self.calibration = calibration
        self.active = False
        self.state = "idle"
        self.status = "idle"
        self.plan = None
        self.joint_target = None
        self.home_settle_ticks = 0
        self.close_ticks = 0
        self.open_ticks = 0
        self.grip_settle_ticks = 0
        self.post_close_ticks = 0
        self.pick_gamma = None
        self.selected_target_index = 0
        self.grip_hold_joint_cmd = None

    def start(self, vision, selected_target_index):
        if self.calibration is None:
            self._fail("calibration missing")
            return 0, GRIPPER_OPEN

        self.active = True
        self.plan = None
        self.joint_target = None
        self.close_ticks = 0
        self.open_ticks = 0
        self.grip_settle_ticks = 0
        self.post_close_ticks = 0
        self.home_settle_ticks = 0
        self.grip_hold_joint_cmd = None
        self.selected_target_index = 0
        self._enter("move_home", "moving home")
        print("Grab: moving to wrist-down home before reading the first target")
        return 0, GRIPPER_OPEN

    def cancel(self):
        if not self.active:
            self.state = "idle"
            self.status = "idle"
            return

        print("Grab: cancelled")
        self.active = False
        self.state = "idle"
        self.status = "cancelled"
        self.grip_hold_joint_cmd = None

    def update(self, joint_cmd, gripper_cmd, vision, timestep, args, arm_math, selected_target_index):
        if not self.active:
            return gripper_cmd, selected_target_index

        if self.state == "move_home":
            reached = step_joints_toward(joint_cmd, wrist_down_home_pose(), timestep, args)
            self.status = "moving home"
            if reached:
                self.home_settle_ticks = max(0, args.pick_home_settle_ticks)
                self._enter("settle_home", "home settle")
            return GRIPPER_OPEN, 0

        if self.state == "settle_home":
            joint_cmd[:] = wrist_down_home_pose()
            if self.home_settle_ticks > 0:
                self.home_settle_ticks -= 1
                self.status = f"home settle {self.home_settle_ticks}"
                return GRIPPER_OPEN, 0
            self._enter("acquire_target", "reading first target")
            return GRIPPER_OPEN, 0

        if self.state == "acquire_target":
            detection = first_target_detection(vision)
            if detection is None:
                self.status = "waiting for target"
                return GRIPPER_OPEN, 0
            # Map detection name to zone if zone not provided by vision backend
            if detection.get("zone") is None:
                det_name = detection.get("name") or ""
                detection["zone"] = ZONE_MAP.get(det_name.lower(), "ZONE_B")

            if not self._build_plan(detection, args, arm_math, joint_cmd):
                return GRIPPER_OPEN, 0
            self._enter("move_object_approach", "object approach")
            return GRIPPER_OPEN, 0

        move_states = {
            "move_object_approach": ("object_approach_xyz", "move_object_pick", "approach object"),
            "move_object_pick": ("object_xyz", "grip_settle", "move to object"),
            "move_lift": ("lift_xyz", "move_zone_approach", "lift object"),
            "move_zone_approach": ("zone_approach_xyz", "move_zone_release", "move to zone"),
            "move_zone_release": ("zone_release_xyz", "opening", "release height"),
        }
        if self.state in move_states:
            target_key, next_state, label = move_states[self.state]
            reached = self._move_to_xyz(joint_cmd, arm_math, self.plan[target_key], timestep, args, label)
            if not self.active:
                return gripper_cmd, self.selected_target_index
            if reached:
                self._enter(next_state, next_state, joint_cmd)
            gripper = GRIPPER_CLOSED if self.state in ("move_lift", "move_zone_approach", "move_zone_release") else GRIPPER_OPEN
            return gripper, self.selected_target_index

        if self.state == "grip_settle":
            return self._update_grip_settle(joint_cmd, args)

        if self.state == "closing":
            return self._update_closing(joint_cmd, args)

        if self.state == "post_close":
            return self._update_post_close(joint_cmd, args)

        if self.state == "opening":
            self.open_ticks += 1
            self.status = f"release {self.open_ticks}/{max(1, args.release_open_ticks)}"
            if self.open_ticks >= max(1, args.release_open_ticks):
                self._complete()
            return GRIPPER_OPEN, self.selected_target_index

        self._fail(f"unknown state {self.state}")
        return gripper_cmd, self.selected_target_index

    def _build_plan(self, detection, args, arm_math, joint_cmd):
        center_px = detection_box_center(detection)
        if center_px is None:
            self._fail("first target has no bounding box")
            return False

        try:
            object_xyz = pixel_to_robot_xyz(center_px, self.calibration, args)
        except Exception as exc:
            self._fail(str(exc))
            return False

        zone = detection.get("zone", "ZONE_B")
        zone_release, zone = zone_release_xyz(zone, self.calibration, args)
        object_approach = object_xyz + np.array([0.0, 0.0, args.pick_approach_m])
        lift_xyz = object_xyz + np.array([0.0, 0.0, args.grab_lift_m])
        zone_approach = zone_release + np.array([0.0, 0.0, args.zone_approach_m])
        zone_approach[2] = max(zone_approach[2], lift_xyz[2])
        _, _, self.pick_gamma = arm_math.forward_kinematics(wrist_down_home_pose())

        self.plan = {
            "name": detection.get("name", "object"),
            "zone": zone,
            "center_px": center_px,
            "object_xyz": object_xyz,
            "object_approach_xyz": object_approach,
            "lift_xyz": lift_xyz,
            "zone_release_xyz": zone_release,
            "zone_approach_xyz": zone_approach,
        }
        print(
            "Grab target: "
            f"{self.plan['name']} at px ({center_px[0]:.0f}, {center_px[1]:.0f}) "
            f"-> xyz {format_position(object_xyz)} -> {zone}"
        )
        return True

    def _move_to_xyz(self, joint_cmd, arm_math, xyz, timestep, args, label):
        if self.joint_target is None:
            ok, target, message = solve_ik_for_position(joint_cmd, arm_math, xyz, self.pick_gamma)
            if not ok:
                self._fail(message)
                return False
            self.joint_target = target

        reached = step_joints_toward(joint_cmd, self.joint_target, timestep, args)
        self.status = f"{label} {format_position(xyz)}"
        return reached

    def _update_grip_settle(self, joint_cmd, args):
        if self.grip_hold_joint_cmd is not None:
            joint_cmd[:] = self.grip_hold_joint_cmd

        total_ticks = max(0, args.grip_settle_ticks)
        if total_ticks == 0:
            self._enter("closing", "closing gripper", joint_cmd)
            return GRIPPER_OPEN, self.selected_target_index

        self.grip_settle_ticks += 1
        self.status = f"settle still {self.grip_settle_ticks}/{total_ticks}"
        if self.grip_settle_ticks >= total_ticks:
            self._enter("closing", "closing gripper", joint_cmd)
        return GRIPPER_OPEN, self.selected_target_index

    def _update_closing(self, joint_cmd, args):
        if self.grip_hold_joint_cmd is not None:
            joint_cmd[:] = self.grip_hold_joint_cmd

        self.close_ticks += 1
        self.status = f"grip still {self.close_ticks}/{max(1, args.grab_close_ticks)}"
        if self.close_ticks >= max(1, args.grab_close_ticks):
            self._enter("post_close", "post grip hold", joint_cmd)
        return GRIPPER_CLOSED, self.selected_target_index

    def _update_post_close(self, joint_cmd, args):
        if self.grip_hold_joint_cmd is not None:
            joint_cmd[:] = self.grip_hold_joint_cmd

        total_ticks = max(0, args.grip_post_close_ticks)
        if total_ticks == 0:
            self._enter("move_lift", "lifting")
            return GRIPPER_CLOSED, self.selected_target_index

        self.post_close_ticks += 1
        self.status = f"post grip still {self.post_close_ticks}/{total_ticks}"
        if self.post_close_ticks >= total_ticks:
            self._enter("move_lift", "lifting")
        return GRIPPER_CLOSED, self.selected_target_index

    def _enter(self, state, status, joint_cmd=None):
        self.state = state
        self.status = status
        self.joint_target = None
        if state == "grip_settle":
            self.grip_settle_ticks = 0
            self.grip_hold_joint_cmd = None if joint_cmd is None else joint_cmd.copy()
        elif state == "closing":
            self.close_ticks = 0
            self.grip_hold_joint_cmd = None if joint_cmd is None else joint_cmd.copy()
        elif state == "post_close":
            self.post_close_ticks = 0
            self.grip_hold_joint_cmd = None if joint_cmd is None else joint_cmd.copy()
        elif state == "opening":
            self.open_ticks = 0
            self.grip_hold_joint_cmd = None
        else:
            self.grip_hold_joint_cmd = None

    def _fail(self, message):
        self.active = False
        self.state = "idle"
        self.status = f"failed: {short_text(message, 22)}"
        self.grip_hold_joint_cmd = None
        print(f"Grab failed: {message}")

    def _complete(self):
        name = self.plan["name"] if self.plan else "object"
        zone = self.plan["zone"] if self.plan else "zone"
        self.active = False
        self.state = "idle"
        self.status = "complete"
        self.grip_hold_joint_cmd = None
        print(f"Grab: placed {name} in {zone}")


class InactiveVision:
    def __init__(self, status, lines):
        self.status = status
        self.lines = lines
        self.detections = []

    def annotate(self, frame, frame_number):
        return frame

    def summary_lines(self, max_lines=3):
        return self.lines[:max_lines]


class YoloClassifier:
    def __init__(self, model, classes, confidence, every_n_frames, half):
        self.model = model
        self.classes = classes
        self.confidence = confidence
        self.every_n_frames = every_n_frames
        self.half = half
        self.enabled = True
        self.detections = []
        self.status = "pit-yolo ready"
        self.fps = None

    def annotate(self, frame, frame_number):
        if not self.enabled:
            return frame

        if frame_number % self.every_n_frames == 0 or not self.detections:
            try:
                prepared = self.model.pre_process(np.ascontiguousarray(frame))
                result = self.model.predict(
                    prepared,
                    classes=self.classes,
                    confidence=self.confidence,
                    verbose=False,
                    half=self.half,
                )
                self.detections = self._extract_detections(result)
                self.fps = getattr(self.model, "FPS", None)
                self.status = self._status_text()
            except Exception as exc:
                self.enabled = False
                self.status = "error; disabled"
                print(f"Warning: YOLO inference failed and has been disabled: {exc}")
                return frame

        return frame

    def summary_lines(self, max_lines=3):
        if not self.detections:
            return ["No objects"]

        return [
            f"{det['name']} {det['confidence']:.0%}"
            for det in self.detections[:max_lines]
        ]

    def _status_text(self):
        object_count = len(self.detections)
        noun = "object" if object_count == 1 else "objects"
        if self.fps:
            return f"pit-yolo {object_count} {noun}, {self.fps:.0f} FPS"
        return f"pit-yolo {object_count} {noun}"

    def _extract_detections(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        names = result.names

        detections = []
        for box, class_id, confidence in zip(boxes, classes, confidences):
            if isinstance(names, dict):
                name = names.get(int(class_id), str(class_id))
            else:
                name = str(class_id)

            detections.append(
                {
                    "box": box,
                    "class_id": int(class_id),
                    "name": name,
                    "confidence": float(confidence),
                }
            )

        detections.sort(key=lambda det: det["confidence"], reverse=True)
        return detections


class RoboflowClassifier:
    def __init__(
        self,
        client,
        workspace_name,
        workflow_id,
        fallback_workflow_id,
        image_key,
        use_cache,
        use_annotated_image,
        class_filter,
        confidence,
        every_n_frames,
    ):
        self.client = client
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.fallback_workflow_id = fallback_workflow_id
        self.image_key = image_key
        self.use_cache = use_cache
        self.use_annotated_image = use_annotated_image
        self.class_filter = set(class_filter) if class_filter else None
        self.confidence = confidence
        self.every_n_frames = every_n_frames
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.detections = []
        self.status = "roboflow workflow ready"
        self.last_error = None
        self.last_detection = None
        self.last_annotated_frame = None
        self.frame_count = 0
        self.detection_count = 0
        self.start_time = time.time()

    def annotate(self, frame, frame_number):
        self._collect_finished_request()
        self.frame_count += 1

        if self.future is None and frame_number % self.every_n_frames == 0:
            self.future = self.executor.submit(
                self._run_workflow_request,
                np.ascontiguousarray(frame.copy()),
            )
            self.status = "roboflow workflow request..."

        if self.use_annotated_image and self.last_annotated_frame is not None:
            return self.last_annotated_frame
        return frame

    def summary_lines(self, max_lines=3):
        if self.last_error:
            return [self.last_error]
        if not self.detections:
            return ["No objects"]

        lines = []
        for det in self.detections[:max_lines]:
            zone = det.get("zone", "ZONE_D")
            lines.append(f"{det['name']} {det['confidence']:.0%} -> {zone}")
        return lines

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def print_stats(self):
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        print("\n=== ROBOFLOW STATISTICS ===")
        print(f"Frames seen: {self.frame_count}")
        print(f"Detections: {self.detection_count}")
        print(f"Workflow FPS: {fps:.1f}")
        print(f"Workflow: {self.workspace_name}/{self.workflow_id}")
        if self.fallback_workflow_id and self.fallback_workflow_id != self.workflow_id:
            print(f"Fallback workflow: {self.fallback_workflow_id}")
        if self.last_detection:
            print(
                "Last detected: "
                f"{self.last_detection['name']} -> {self.last_detection['zone']}"
            )

    def _collect_finished_request(self):
        if self.future is None or not self.future.done():
            return

        try:
            result, workflow_id = self.future.result()
            output, predictions = self._parse_workflow_result(result)
            self.detections = self._predictions_to_detections(predictions)
            self.detection_count += len(self.detections)
            self.last_error = None
            self.last_annotated_frame = self._decode_annotated_frame(output)
            object_count = len(self.detections)
            noun = "object" if object_count == 1 else "objects"
            self.status = f"roboflow workflow {object_count} {noun}"
            if self.detections:
                self.last_detection = self.detections[0]
                self._print_detections()
        except Exception as exc:
            self.status = "roboflow workflow error"
            self.last_error = short_text(exc)
            print(f"Warning: Roboflow workflow request failed: {exc}")
        finally:
            self.future = None

    def _run_workflow_request(self, frame):
        try:
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={self.image_key: frame},
                use_cache=self.use_cache,
            )
            return result, self.workflow_id
        except Exception as exc:
            if not self.fallback_workflow_id or self.fallback_workflow_id == self.workflow_id:
                raise

            print(
                "Warning: primary Roboflow workflow failed; trying fallback "
                f"{self.fallback_workflow_id}. Primary error: {short_text(exc, 120)}"
            )
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.fallback_workflow_id,
                images={self.image_key: frame},
                use_cache=self.use_cache,
            )
            return result, self.fallback_workflow_id

    def _parse_workflow_result(self, result):
        try:
            output = result[0]
            predictions = output.get("predictions", {}).get("predictions", [])
        except (IndexError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Could not parse Roboflow workflow output: {exc}") from exc

        if not isinstance(predictions, list):
            predictions = []

        return output, predictions

    def _predictions_to_detections(self, predictions):
        detections = []
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            detection = self._prediction_to_detection(prediction)
            if detection is not None:
                detections.append(detection)
        detections.sort(key=lambda det: det["confidence"], reverse=True)
        return detections

    def _decode_annotated_frame(self, output):
        if not self.use_annotated_image or not isinstance(output, dict):
            return None

        encoded = output.get("annotated_image")
        if not encoded:
            return None

        try:
            image = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
            frame = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
            if frame.shape[:2] != (CAMERA_HEIGHT, CAMERA_WIDTH):
                frame = resize_image(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            return frame
        except Exception as exc:
            print(f"Warning: could not decode Roboflow annotated image: {exc}")
            return None

    def _print_detections(self):
        print(f"\n[Roboflow frame {self.frame_count}] {len(self.detections)} object(s) detected:")
        for index, det in enumerate(self.detections, start=1):
            box = det.get("box")
            if box is None:
                center = "(n/a,n/a)"
            else:
                x1, y1, x2, y2 = box
                center = f"({(x1 + x2) / 2:.0f},{(y1 + y2) / 2:.0f})"
            print(
                f"  {index}. {det['name']} ({det['confidence']:.2f}) "
                f"at {center} -> {det['zone']}"
            )

        best = self.detections[0]
        print(
            f"  NEXT PICK: {best['name']} ({best['confidence']:.2f}) "
            f"-> {best['zone']}"
        )

    def _collect_detections(self, value, detections, parent_key=None):
        if isinstance(value, list):
            for item in value:
                self._collect_detections(item, detections, parent_key=parent_key)
            return

        if not isinstance(value, dict):
            return

        detection = self._prediction_to_detection(value)
        if detection is not None:
            detections.append(detection)
            return

        if parent_key == "predictions" and self._looks_like_classification_predictions(value):
            detections.extend(self._extract_classification_predictions(value))
            return

        for key, child in value.items():
            self._collect_detections(child, detections, parent_key=key)

    def _prediction_to_detection(self, prediction):
        name = (
            prediction.get("class")
            or prediction.get("class_name")
            or prediction.get("label")
            or prediction.get("predicted_class")
        )
        if name is None:
            return None

        confidence = self._prediction_confidence(prediction)
        if confidence is None:
            confidence = 1.0

        name = str(name)
        if self.class_filter and name not in self.class_filter:
            return None
        if confidence < self.confidence:
            return None

        zone = ZONE_MAP.get(name, "ZONE_D")
        return {
            "box": self._prediction_box(prediction),
            "class_id": self._prediction_class_id(prediction, name),
            "name": name,
            "confidence": confidence,
            "zone": zone,
        }

    def _extract_classification_predictions(self, predictions):
        detections = []
        for name, value in predictions.items():
            if isinstance(value, dict):
                confidence = self._prediction_confidence(value)
            else:
                confidence = self._to_float(value)

            if confidence is None:
                continue

            name = str(name)
            if self.class_filter and name not in self.class_filter:
                continue
            if confidence < self.confidence:
                continue

            detections.append(
                {
                    "box": None,
                    "class_id": stable_class_id(name),
                    "name": name,
                    "confidence": confidence,
                    "zone": ZONE_MAP.get(name, "ZONE_D"),
                }
            )

        return detections

    def _looks_like_classification_predictions(self, predictions):
        if not predictions:
            return False

        for value in predictions.values():
            if isinstance(value, dict):
                if self._prediction_confidence(value) is None:
                    return False
            elif self._to_float(value) is None:
                return False

        return True

    def _prediction_confidence(self, prediction):
        for key in ("confidence", "score", "probability"):
            if key in prediction:
                return self._to_float(prediction[key])
        return None

    def _prediction_class_id(self, prediction, name):
        class_id = self._to_float(prediction.get("class_id"))
        if class_id is None:
            return stable_class_id(name)
        return int(class_id)

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _prediction_box(prediction):
        if all(key in prediction for key in ("x", "y", "width", "height")):
            x = float(prediction["x"])
            y = float(prediction["y"])
            width = float(prediction["width"])
            height = float(prediction["height"])
            return [
                x - width / 2,
                y - height / 2,
                x + width / 2,
                y + height / 2,
            ]

        if all(key in prediction for key in ("x1", "y1", "x2", "y2")):
            return [
                float(prediction["x1"]),
                float(prediction["y1"]),
                float(prediction["x2"]),
                float(prediction["y2"]),
            ]

        return None


def stable_class_id(name):
    return sum(ord(char) for char in str(name))


def detection_color(class_id):
    palette = [
        (52, 211, 153),
        (96, 165, 250),
        (251, 191, 36),
        (248, 113, 113),
        (167, 139, 250),
        (45, 212, 191),
    ]
    return palette[class_id % len(palette)]


def main():
    args = parse_args()
    print_controls(args.camera_index, not args.no_camera)

    pygame.init()
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("QArm Mini Keyboard Control + Camera")
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 22)

    cap = None if args.no_camera else open_camera(args.camera_index)
    vision = load_vision_backend(args) if cap is not None else None
    calibration = load_calibration_map(args)

    # load taught zones file (optional)
    zones = load_zones_file(args.zones_file) if getattr(args, "zones_file", None) else {}
    # if workspace coords present in zones file, use them as overrides when CLI args missing
    if getattr(args, "zone_a_xyz", None) is None and zones.get("ZONE_A", {}).get("workspace_m"):
        args.zone_a_xyz = zones["ZONE_A"]["workspace_m"]
    if getattr(args, "zone_b_xyz", None) is None and zones.get("ZONE_B", {}).get("workspace_m"):
        args.zone_b_xyz = zones["ZONE_B"]["workspace_m"]
    if getattr(args, "zone_c_xyz", None) is None and zones.get("ZONE_C", {}).get("workspace_m"):
        args.zone_c_xyz = zones["ZONE_C"]["workspace_m"]

    myMiniArm = QArmMini(hardware=1, id=3)
    arm_math = QArmMiniFunctions()
    grab_controller = GrabController(calibration)
    timer = QTimer(sampleRate=SAMPLE_RATE_HZ, totalTime=RUN_TIME_SECONDS)

    joint_cmd = wrist_down_home_pose()
    gripper_cmd = GRIPPER_OPEN
    frame_number = 0
    auto_center_enabled = False
    center_status = "center off"
    selected_target_index = 0

    try:
        zones_locked = all(z in zones and zones[z].get("workspace_m") for z in ("ZONE_A", "ZONE_B", "ZONE_C"))
        while timer.check():
            (
                should_quit,
                gripper_cmd,
                toggle_auto_center,
                next_center_target,
                start_grab,
                cancel_grab,
                home_requested,
                record_zone,
            ) = handle_keydown_events(
                joint_cmd,
                gripper_cmd,
                vision,
            )
            if should_quit:
                break
            if home_requested:
                grab_controller.cancel()
            if toggle_auto_center:
                auto_center_enabled = not auto_center_enabled
                if auto_center_enabled:
                    selected_target_index = 0
                    detection, targets, target_index = selected_target_detection(vision, selected_target_index)
                    center_status = (
                        selected_target_status(detection, targets, target_index)
                        if detection is not None
                        else "waiting for targets"
                    )
                else:
                    center_status = "center off"
                print(f"Camera centering: {'on' if auto_center_enabled else 'off'}")
            if next_center_target:
                if grab_controller.active and grab_controller.state != "centering":
                    print("Center target: finish or cancel grab before switching targets")
                else:
                    selected_target_index, center_status = select_next_target(vision, selected_target_index)
                    print(f"Center target: {center_status}")
            if cancel_grab:
                grab_controller.cancel()
            if start_grab:
                selected_target_index, gripper_cmd = grab_controller.start(vision, selected_target_index)
                auto_center_enabled = False
                center_status = "center off"

            # Auto-pick: if enabled and controller idle, start a grab on the first detection
            if (
                getattr(args, "auto_pick", False)
                and not grab_controller.active
                and vision is not None
                and first_target_detection(vision) is not None
            ):
                selected_target_index, gripper_cmd = grab_controller.start(vision, selected_target_index)
                auto_center_enabled = False
                center_status = "center off"

            if not grab_controller.active:
                apply_arrow_key_motion(joint_cmd, timer.get_sample_time())
            if auto_center_enabled and not grab_controller.active:
                center_status, selected_target_index = apply_auto_center_motion(
                    joint_cmd,
                    vision,
                    timer.get_sample_time(),
                    args,
                    selected_target_index,
                )
            gripper_cmd, selected_target_index = grab_controller.update(
                joint_cmd,
                gripper_cmd,
                vision,
                timer.get_sample_time(),
                args,
                arm_math,
                selected_target_index,
            )
            myMiniArm.read_write_std(joint_cmd, gripper_cmd)

            # If user requested a zone record (press 1/2/3), capture measured joints and FK then save
            if record_zone is not None:
                try:
                    # if zones already all recorded, ignore unless user holds Shift
                    mods = pygame.key.get_mods()
                    shift_held = bool(mods & pygame.KMOD_SHIFT)
                    existing = zones.get(record_zone, {}).get("workspace_m")
                    if zones_locked and not shift_held:
                        print(f"All zones already recorded; hold Shift+{record_zone[-1]} to overwrite.")
                    elif existing and not shift_held:
                        print(f"{record_zone} already recorded. Hold Shift+{record_zone[-1]} to overwrite.")
                    else:
                        measured = getattr(myMiniArm, "positionMeasured", None)
                        if measured is not None:
                            joints_meas = np.asarray(measured, dtype=float)[:4].copy()
                        else:
                            joints_meas = np.asarray(joint_cmd, dtype=float).copy()
                        # compute FK
                        try:
                            pos, _, _ = arm_math.forward_kinematics(joints_meas)
                            workspace_m = [float(x) for x in pos]
                        except Exception:
                            workspace_m = None

                        entry = {
                            "joint_cmd_rad": [float(x) for x in list(joints_meas)],
                            "joint_cmd_deg": [float(x) for x in list(np.rad2deg(joints_meas))],
                            "workspace_m": workspace_m,
                        }
                        zones[record_zone] = entry
                        saved = save_zones_file(args.zones_file, zones)
                        print(f"Recorded {record_zone}: {entry}")
                        if saved:
                            print(f"Saved zones to {args.zones_file}")
                        # recompute locked state
                        zones_locked = all(z in zones and zones[z].get("workspace_m") for z in ("ZONE_A", "ZONE_B", "ZONE_C"))
                        if zones_locked:
                            print("All three zones recorded — recording now locked. Hold Shift+1/2/3 to overwrite.")
                except Exception as exc:
                    print(f"Error recording zone {record_zone}: {exc}")

            frame = read_camera_frame(cap)
            if frame is not None:
                frame_number += 1
                if vision is not None:
                    frame = vision.annotate(frame, frame_number)

            draw_keyboard_window(
                font,
                small_font,
                gripper_cmd,
                frame_to_surface(frame),
                vision,
                auto_center_enabled,
                center_status,
                center_target(args),
                selected_target_index,
                grab_controller.status,
            )

            timer.sleep()

    except KeyboardInterrupt:
        print("Received user terminate command.")

    finally:
        if cap is not None:
            cap.release()
        if vision is not None and hasattr(vision, "shutdown"):
            vision.shutdown()
        myMiniArm.terminate()
        pygame.quit()


if __name__ == "__main__":
    main()
