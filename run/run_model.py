#-----------------------------------------------------------------------------#
#---------------- Keyboard + Camera Control - QArm Mini -----------------------#
#-----------------------------------------------------------------------------#

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import pygame
from pal.products.qarm_mini import QArmMini
from pal.utilities.timing import QTimer


SAMPLE_RATE_HZ = 5.0
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
    "plastic bottles": "ZONE_A",
    "paper cup": "ZONE_B",
    "metal cans": "ZONE_C",
    "paper crumble": "ZONE_B",
    "paper box": "ZONE_B",
    "marker": "ZONE_D",
    "pen": "ZONE_D",
}

CENTER_TARGET_X_RATIO = 0.5
DEFAULT_CENTER_TARGET_Y_RATIO = 0.5
DEFAULT_CENTER_DEADBAND_PX = 30
DEFAULT_CENTER_BASE_SIGN = -1.0
DEFAULT_CENTER_SHOULDER_SIGN = 1.0


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

    args = parser.parse_args()
    if not args.roboflow_workflow_id and args.roboflow_model_id:
        args.roboflow_workflow_id = args.roboflow_model_id
    return args


def parse_bool_env(value, default=False):
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


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
    print("  w/s         wrist up/down")
    print("  p           close gripper")
    print("  o           open gripper")
    print("  h           home position")
    print("  c           toggle camera/object centering")
    print("  n           center next detected target")
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
        "h home, c center camera",
        "n next center target",
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
    y += 32

    draw_text(screen, font, "Targets", (panel_x, y), (240, 240, 240))
    y += 26
    target_lines = target_list_lines(vision, selected_target_index, max_lines=3)
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
                joint_cmd[:] = QArmMini.HOME_POSE
                print("Moving to home position")
            elif event.key == pygame.K_c:
                toggle_auto_center = True
            elif event.key == pygame.K_n:
                next_center_target = True
            elif event.key == pygame.K_t and vision is not None and hasattr(vision, "print_stats"):
                vision.print_stats()

    return should_quit, gripper_cmd, toggle_auto_center, next_center_target


def apply_arrow_key_motion(joint_cmd, timestep):
    keys = pygame.key.get_pressed()
    step = JOINT_SPEED_RAD_PER_SEC * timestep

    joint_cmd[0] += (int(keys[pygame.K_LEFT]) - int(keys[pygame.K_RIGHT])) * step
    joint_cmd[1] += (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])) * step
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

    myMiniArm = QArmMini(hardware=1, id=3)
    timer = QTimer(sampleRate=SAMPLE_RATE_HZ, totalTime=RUN_TIME_SECONDS)

    joint_cmd = QArmMini.HOME_POSE.copy()
    gripper_cmd = GRIPPER_OPEN
    frame_number = 0
    auto_center_enabled = False
    center_status = "center off"
    selected_target_index = 0

    try:
        while timer.check():
            should_quit, gripper_cmd, toggle_auto_center, next_center_target = handle_keydown_events(
                joint_cmd,
                gripper_cmd,
                vision,
            )
            if should_quit:
                break
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
                selected_target_index, center_status = select_next_target(vision, selected_target_index)
                print(f"Center target: {center_status}")

            apply_arrow_key_motion(joint_cmd, timer.get_sample_time())
            if auto_center_enabled:
                center_status, selected_target_index = apply_auto_center_motion(
                    joint_cmd,
                    vision,
                    timer.get_sample_time(),
                    args,
                    selected_target_index,
                )
            myMiniArm.read_write_std(joint_cmd, gripper_cmd)

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
