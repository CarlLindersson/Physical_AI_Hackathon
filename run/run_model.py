#-----------------------------------------------------------------------------#
#---------------- Keyboard + Camera Control - QArm Mini -----------------------#
#-----------------------------------------------------------------------------#

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pygame
from pal.products.qarm_mini import QArmMini
from pal.utilities.timing import QTimer


SAMPLE_RATE_HZ = 5.0
RUN_TIME_SECONDS = 300.0
JOINT_SPEED_RAD_PER_SEC = np.pi / 4

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
PANEL_WIDTH = 320
WINDOW_WIDTH = CAMERA_WIDTH + PANEL_WIDTH
WINDOW_HEIGHT = CAMERA_HEIGHT

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0


def parse_args():
    load_local_env()
    parser = argparse.ArgumentParser(description="QArm Mini keyboard control with live camera feed.")
    parser.add_argument("--camera-index", type=int, default=1, help="OpenCV camera index to use.")
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
        default=os.environ.get("ROBOFLOW_WORKSPACE_NAME"),
        help="Roboflow workspace name for run_workflow. Defaults to ROBOFLOW_WORKSPACE_NAME.",
    )
    parser.add_argument(
        "--roboflow-workflow-id",
        default=os.environ.get("ROBOFLOW_WORKFLOW_ID"),
        help="Roboflow workflow ID for run_workflow. Defaults to ROBOFLOW_WORKFLOW_ID.",
    )
    parser.add_argument(
        "--roboflow-image-key",
        default=os.environ.get("ROBOFLOW_IMAGE_KEY", "image"),
        help="Workflow image input name. Defaults to ROBOFLOW_IMAGE_KEY or 'image'.",
    )
    parser.add_argument(
        "--roboflow-confidence",
        type=float,
        default=0.4,
        help="Roboflow confidence threshold from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--roboflow-every-n-frames",
        type=int,
        default=8,
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
    print("  p           close gripper")
    print("  o           open gripper")
    print("  h           home position")
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

    try:
        from inference_sdk import InferenceConfiguration

        client.configure(
            InferenceConfiguration(confidence_threshold=args.roboflow_confidence)
        )
    except Exception as exc:
        print(f"Warning: could not set Roboflow confidence on the client: {exc}")
        print("         Predictions will still be filtered locally.")

    print(
        "Roboflow workflow configured: "
        f"{args.roboflow_workspace_name}/{args.roboflow_workflow_id}"
    )
    return RoboflowClassifier(
        client=client,
        workspace_name=args.roboflow_workspace_name,
        workflow_id=args.roboflow_workflow_id,
        image_key=args.roboflow_image_key,
        use_cache=not args.no_roboflow_cache,
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


def draw_camera_feed(screen, font, camera_surface, vision):
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
    if vision:
        draw_detection_overlays(screen, font, vision.detections)


def draw_keyboard_window(font, small_font, gripper_cmd, camera_surface, vision):
    screen = pygame.display.get_surface()
    screen.fill((20, 22, 26))
    draw_camera_feed(screen, small_font, camera_surface, vision)

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

    for heading, items in (
        ("Arrows", ("Base left/right", "Shoulder up/down")),
        ("Gripper", ("p closes", "o opens")),
        ("Other", ("h homes the arm", "q or Esc quits")),
    ):
        draw_text(screen, font, heading, (panel_x, y), (240, 240, 240))
        y += 26
        for item in items:
            draw_text(screen, small_font, item, (panel_x, y))
            y += 22
        y += 10

    draw_text(screen, font, "Vision", (panel_x, y), (240, 240, 240))
    y += 26
    draw_text(screen, small_font, vision.status if vision else "off", (panel_x, y))
    y += 22

    if vision:
        for line in vision.summary_lines():
            draw_text(screen, small_font, line, (panel_x, y))
            y += 22

    pygame.display.flip()


def draw_detection_overlays(screen, font, detections):
    for det in detections:
        if det["box"] is None:
            continue

        x1, y1, x2, y2 = det["box"]
        x1 = int(np.clip(x1, 0, CAMERA_WIDTH - 1))
        y1 = int(np.clip(y1, 0, CAMERA_HEIGHT - 1))
        x2 = int(np.clip(x2, 0, CAMERA_WIDTH - 1))
        y2 = int(np.clip(y2, 0, CAMERA_HEIGHT - 1))
        color = detection_color(det["class_id"])
        label = f"{det['name']} {det['confidence']:.0%}"

        pygame.draw.rect(screen, color, pygame.Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1)), 2)
        label_surface = font.render(label, True, (12, 14, 18))
        label_rect = label_surface.get_rect()
        label_rect.topleft = (x1, max(0, y1 - label_rect.height - 6))
        bg_rect = label_rect.inflate(8, 4)
        pygame.draw.rect(screen, color, bg_rect)
        screen.blit(label_surface, label_rect)


def handle_keydown_events(joint_cmd, gripper_cmd):
    should_quit = False

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

    return should_quit, gripper_cmd


def apply_arrow_key_motion(joint_cmd, timestep):
    keys = pygame.key.get_pressed()
    step = JOINT_SPEED_RAD_PER_SEC * timestep

    joint_cmd[0] += (int(keys[pygame.K_LEFT]) - int(keys[pygame.K_RIGHT])) * step
    joint_cmd[1] += (int(keys[pygame.K_UP]) - int(keys[pygame.K_DOWN])) * step

    np.clip(joint_cmd, QArmMini.LIMITS_MIN, QArmMini.LIMITS_MAX, out=joint_cmd)


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
        image_key,
        use_cache,
        class_filter,
        confidence,
        every_n_frames,
    ):
        self.client = client
        self.workspace_name = workspace_name
        self.workflow_id = workflow_id
        self.image_key = image_key
        self.use_cache = use_cache
        self.class_filter = set(class_filter) if class_filter else None
        self.confidence = confidence
        self.every_n_frames = every_n_frames
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.detections = []
        self.status = "roboflow workflow ready"
        self.last_error = None

    def annotate(self, frame, frame_number):
        self._collect_finished_request()

        if self.future is None and frame_number % self.every_n_frames == 0:
            image = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
            self.future = self.executor.submit(
                self.client.run_workflow,
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={self.image_key: image},
                use_cache=self.use_cache,
            )
            self.status = "roboflow workflow request..."

        return frame

    def summary_lines(self, max_lines=3):
        if self.last_error:
            return [self.last_error]
        if not self.detections:
            return ["No objects"]

        return [
            f"{det['name']} {det['confidence']:.0%}"
            for det in self.detections[:max_lines]
        ]

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _collect_finished_request(self):
        if self.future is None or not self.future.done():
            return

        try:
            result = self.future.result()
            self.detections = self._extract_detections(result)
            self.last_error = None
            object_count = len(self.detections)
            noun = "object" if object_count == 1 else "objects"
            self.status = f"roboflow workflow {object_count} {noun}"
        except Exception as exc:
            self.status = "roboflow workflow error"
            self.last_error = short_text(exc)
            print(f"Warning: Roboflow workflow request failed: {exc}")
        finally:
            self.future = None

    def _extract_detections(self, result):
        detections = []
        self._collect_detections(result, detections)
        detections.sort(key=lambda det: det["confidence"], reverse=True)
        return detections

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

        return {
            "box": self._prediction_box(prediction),
            "class_id": stable_class_id(name),
            "name": name,
            "confidence": confidence,
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

    try:
        while timer.check():
            should_quit, gripper_cmd = handle_keydown_events(joint_cmd, gripper_cmd)
            if should_quit:
                break

            apply_arrow_key_motion(joint_cmd, timer.get_sample_time())
            myMiniArm.read_write_std(joint_cmd, gripper_cmd)

            frame = read_camera_frame(cap)
            if frame is not None:
                frame_number += 1
                if vision is not None:
                    frame = vision.annotate(frame, frame_number)

            draw_keyboard_window(font, small_font, gripper_cmd, frame_to_surface(frame), vision)

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
