#!/usr/bin/env python3
"""Five-point camera-to-QArm calibration workflow.

This tool supports the workflow needed before object-detector pick poses:

1. Move the QArm Mini to a camera calibration pose.
2. Drag five image markers onto visible table calibration targets.
3. Jog the robot to each corresponding target and press r to record.
4. Save a JSON calibration map with image points, joint angles, FK XYZ points,
   and fitted image-pixel to robot-XY transforms.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np


CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
GRIPPER_OPEN = 0.0
WRIST_DOWN_DEG = 0.0

WINDOW_NAME = "QArm Camera Calibration"
POINT_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left", "center")

KEY_LEFT = {2424832, 81, 65361}
KEY_UP = {2490368, 82, 65362}
KEY_RIGHT = {2555904, 83, 65363}
KEY_DOWN = {2621440, 84, 65364}
KEY_ENTER = {10, 13}
KEY_BACKSPACE = {8, 127, 3014656}
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_W = 0x57
VK_E = 0x45
VK_D = 0x44
VK_S = 0x53


def parse_args() -> argparse.Namespace:
    load_local_env()
    parser = argparse.ArgumentParser(
        description="Calibrate camera pixels to QArm Mini table/workspace points."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=int(os.environ.get("CAMERA_INDEX", "1")),
        help="OpenCV camera index. Defaults to CAMERA_INDEX or 1.",
    )
    parser.add_argument("--frame-width", type=int, default=CAMERA_WIDTH)
    parser.add_argument("--frame-height", type=int, default=CAMERA_HEIGHT)
    parser.add_argument(
        "--output",
        default="run/calibration_map.json",
        help="JSON file written after all five points are recorded.",
    )
    parser.add_argument(
        "--marker-margin-ratio",
        type=float,
        default=0.16,
        help="Initial marker inset from image edges, as a fraction of frame size.",
    )
    parser.add_argument(
        "--wrist-down-deg",
        type=float,
        default=WRIST_DOWN_DEG,
        help=(
            "Wrist joint angle used for the downward-looking calibration pose. "
            "Default 0 deg keeps the arm at home except for the wrist, which points down."
        ),
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=20.0,
        help="Robot command/update rate in Hz.",
    )
    parser.add_argument(
        "--home-settle-seconds",
        type=float,
        default=2.0,
        help="How long to hold the home pose before rotating the wrist down.",
    )
    parser.add_argument(
        "--pose-move-seconds",
        type=float,
        default=2.0,
        help="How long to interpolate from home to the wrist-down pose.",
    )
    parser.add_argument(
        "--joint-step-deg",
        type=float,
        default=1.0,
        help="Initial jog increment for robot joint controls.",
    )
    parser.add_argument("--robot-id", type=int, default=3, help="QArm Mini board id.")
    parser.add_argument(
        "--virtual",
        action="store_true",
        help="Connect to the virtual QArm Mini instead of hardware.",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Run UI without connecting to robot; records commanded poses only.",
    )
    return parser.parse_args()


def load_local_env() -> None:
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
        if env_file in seen or not env_file.exists():
            continue
        seen.add(env_file)

        for line in env_file.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def ensure_quanser_python_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    quanser_python = (
        project_root
        / "academic_resources"
        / "Quanser_Academic_Resources-dev-windows"
        / "0_libraries"
        / "python"
    )
    if quanser_python.exists() and str(quanser_python) not in sys.path:
        sys.path.insert(0, str(quanser_python))


class RobotContext:
    def __init__(self, args: argparse.Namespace):
        ensure_quanser_python_path()
        self.arm = None
        self.arm_math = None
        self.qarm_class = None
        self.home_pose = np.array([0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
        self.limits_min = np.array([-10 * np.pi / 18, -np.pi / 6, -5 * np.pi / 6, -np.pi / 10])
        self.limits_max = np.array([4 * np.pi / 3, 13 * np.pi / 12, np.pi / 6, 8 * np.pi / 9])
        self.joint_order = ["yaw", "shoulder", "elbow", "wrist"]

        try:
            from hal.content.qarm_mini import QArmMiniFunctions
            from pal.products.qarm_mini import QArmMini

            self.qarm_class = QArmMini
            self.arm_math = QArmMiniFunctions()
            self.home_pose = QArmMini.HOME_POSE.copy()
            self.limits_min = QArmMini.LIMITS_MIN.copy()
            self.limits_max = QArmMini.LIMITS_MAX.copy()
        except Exception as exc:
            if not args.no_robot:
                raise RuntimeError(
                    "Could not import Quanser QArm Mini modules. "
                    "Run setup first or use --no-robot for camera-only testing."
                ) from exc
            print(f"Warning: QArm Mini modules unavailable in --no-robot mode: {exc}")

        if not args.no_robot:
            hardware = 0 if args.virtual else 1
            self.arm = self.qarm_class(hardware=hardware, id=args.robot_id)

    def terminate(self) -> None:
        if self.arm is not None and hasattr(self.arm, "terminate"):
            self.arm.terminate()


class CalibrationApp:
    def __init__(self, args: argparse.Namespace, robot: RobotContext):
        self.args = args
        self.robot = robot
        self.frame_size = (args.frame_width, args.frame_height)
        self.image_points = initial_image_points(
            args.frame_width,
            args.frame_height,
            args.marker_margin_ratio,
        )
        self.selected_index = 0
        self.drag_index = None
        self.marker_nudge_px = 1
        self.joint_step_deg = args.joint_step_deg
        self.last_measured = None
        self.last_workspace = None
        self.async_keyboard = WindowsAsyncKeyboard()

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            nearest = self.nearest_marker(x, y)
            if nearest is None:
                self.image_points[self.selected_index] = self.clip_pixel((x, y))
                self.drag_index = self.selected_index
            else:
                self.selected_index = nearest
                self.drag_index = nearest
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_index is not None:
            self.image_points[self.drag_index] = self.clip_pixel((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_index = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.image_points[self.selected_index] = self.clip_pixel((x, y))

    def nearest_marker(self, x: int, y: int, radius_px: float = 30.0) -> int | None:
        point = np.array([x, y], dtype=float)
        distances = [
            np.linalg.norm(point - np.asarray(marker, dtype=float))
            for marker in self.image_points
        ]
        nearest = int(np.argmin(distances))
        if distances[nearest] <= radius_px:
            return nearest
        return None

    def clip_pixel(self, point: tuple[int, int]) -> list[float]:
        x, y = point
        return [
            float(np.clip(x, 0, self.args.frame_width - 1)),
            float(np.clip(y, 0, self.args.frame_height - 1)),
        ]

    def align_image_points(self, cap: cv2.VideoCapture) -> list[list[float]]:
        print("\nStep 1: align image markers")
        print("  Drag each marker onto the matching visible calibration target.")
        print("  1-5 select a marker, arrow keys nudge it, right-click moves selected marker.")
        print("  Press Space or Enter when the five points line up. Press q to cancel.\n")

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)

        while True:
            frame = read_camera_frame(cap, self.args.frame_width, self.args.frame_height)
            display = frame.copy()
            draw_markers(display, self.image_points, self.selected_index, [False] * len(POINT_NAMES))
            draw_panel(
                display,
                [
                    "Align image points",
                    "Drag targets. 1-5 select. Arrows nudge. Space/Enter locks.",
                    f"Selected: {self.selected_index + 1} {POINT_NAMES[self.selected_index]}",
                ],
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            if self.handle_marker_key(key):
                return [point.copy() for point in self.image_points]

    def handle_marker_key(self, key: int) -> bool:
        char = key_to_char(key)
        if char in ("q", "\x1b"):
            raise SystemExit("Calibration cancelled during image marker alignment.")
        if char in (" ", "\r", "\n") or key in KEY_ENTER:
            return True
        if char and char in "12345":
            self.selected_index = int(char) - 1
            return False
        if char in ("[", "-"):
            self.marker_nudge_px = max(1, self.marker_nudge_px - 1)
            return False
        if char in ("]", "+", "="):
            self.marker_nudge_px = min(25, self.marker_nudge_px + 1)
            return False

        dx, dy = 0, 0
        if key in KEY_LEFT:
            dx = -self.marker_nudge_px
        elif key in KEY_RIGHT:
            dx = self.marker_nudge_px
        elif key in KEY_UP:
            dy = -self.marker_nudge_px
        elif key in KEY_DOWN:
            dy = self.marker_nudge_px

        if dx or dy:
            x, y = self.image_points[self.selected_index]
            self.image_points[self.selected_index] = self.clip_pixel((int(x + dx), int(y + dy)))
        return False

    def record_robot_points(
        self,
        cap: cv2.VideoCapture,
        joint_cmd: np.ndarray,
    ) -> list[dict[str, Any]]:
        print("\nStep 2: record robot points")
        print("  Jog the QArm tip/camera reference to each table target, then press r.")
        print("  Hold Arrow keys for yaw/shoulder. Hold w/s for wrist.")
        print("  Hold e/d to jog elbow if you need extra reach.")
        print("  n/p select next/previous target. +/- changes jog step. Enter saves when all are recorded.\n")

        records: list[dict[str, Any] | None] = [None] * len(POINT_NAMES)

        while True:
            frame = read_camera_frame(cap, self.args.frame_width, self.args.frame_height)
            self.apply_held_key_jog(joint_cmd)
            self.update_robot(joint_cmd)
            display = frame.copy()
            draw_markers(
                display,
                self.image_points,
                self.selected_index,
                [record is not None for record in records],
            )
            draw_panel(display, self.robot_status_lines(joint_cmd, records))
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKeyEx(int(1000 / max(1.0, self.args.sample_rate)))
            if key == -1:
                continue

            done = self.handle_robot_key(key, joint_cmd, records)
            if done:
                return [record for record in records if record is not None]

    def update_robot(self, joint_cmd: np.ndarray) -> None:
        np.clip(joint_cmd, self.robot.limits_min, self.robot.limits_max, out=joint_cmd)
        if self.robot.arm is not None:
            self.robot.arm.read_write_std(joint_cmd, GRIPPER_OPEN)
            measured = np.asarray(self.robot.arm.positionMeasured, dtype=float).reshape(-1)
            if measured.size >= 4 and np.isfinite(measured[:4]).all():
                self.last_measured = measured[:4].copy()
        else:
            self.last_measured = joint_cmd.copy()

        self.last_workspace = fk_workspace(self.robot, self.last_measured)

    def handle_robot_key(
        self,
        key: int,
        joint_cmd: np.ndarray,
        records: list[dict[str, Any] | None],
    ) -> bool:
        char = key_to_char(key)
        if char in ("q", "\x1b"):
            raise SystemExit("Calibration cancelled during robot point recording.")
        if char in ("+", "="):
            self.joint_step_deg = min(10.0, self.joint_step_deg * 2.0)
            print(f"Jog step: {self.joint_step_deg:.3f} deg")
            return False
        if char in ("-", "_"):
            self.joint_step_deg = max(0.05, self.joint_step_deg / 2.0)
            print(f"Jog step: {self.joint_step_deg:.3f} deg")
            return False
        if char == "n":
            self.selected_index = (self.selected_index + 1) % len(POINT_NAMES)
            return False
        if char == "p":
            self.selected_index = (self.selected_index - 1) % len(POINT_NAMES)
            return False
        if char == "h":
            joint_cmd[:] = calibration_pose(self.robot, self.args)
            print("Returned to calibration home pose with wrist down")
            return False
        if key in KEY_BACKSPACE:
            records[self.selected_index] = None
            print(f"Cleared point {self.selected_index + 1}")
            return False
        if char == "r":
            records[self.selected_index] = self.make_record(joint_cmd)
            print_record(records[self.selected_index])
            if not all(record is not None for record in records):
                self.selected_index = next_unrecorded_index(records, self.selected_index)
            return False
        if (char in (" ", "\r", "\n") or key in KEY_ENTER) and all(
            record is not None for record in records
        ):
            return True

        if self.async_keyboard.available and is_robot_jog_key(key, char):
            return False

        self.apply_joint_jog(key, char, joint_cmd)
        return False

    def apply_held_key_jog(self, joint_cmd: np.ndarray) -> None:
        if not self.async_keyboard.available:
            return

        step = np.deg2rad(self.joint_step_deg)
        if self.async_keyboard.is_down(VK_LEFT):
            joint_cmd[0] += step
        if self.async_keyboard.is_down(VK_RIGHT):
            joint_cmd[0] -= step
        if self.async_keyboard.is_down(VK_UP):
            joint_cmd[1] += step
        if self.async_keyboard.is_down(VK_DOWN):
            joint_cmd[1] -= step
        if self.async_keyboard.is_down(VK_E):
            joint_cmd[2] += step
        if self.async_keyboard.is_down(VK_D):
            joint_cmd[2] -= step
        if self.async_keyboard.is_down(VK_W):
            joint_cmd[3] += step
        if self.async_keyboard.is_down(VK_S):
            joint_cmd[3] -= step
        np.clip(joint_cmd, self.robot.limits_min, self.robot.limits_max, out=joint_cmd)

    def apply_joint_jog(self, key: int, char: str, joint_cmd: np.ndarray) -> None:
        step = np.deg2rad(self.joint_step_deg)
        if key in KEY_LEFT:
            joint_cmd[0] += step
        elif key in KEY_RIGHT:
            joint_cmd[0] -= step
        elif key in KEY_UP:
            joint_cmd[1] += step
        elif key in KEY_DOWN:
            joint_cmd[1] -= step
        elif char == "e":
            joint_cmd[2] += step
        elif char == "d":
            joint_cmd[2] -= step
        elif char == "w":
            joint_cmd[3] += step
        elif char == "s":
            joint_cmd[3] -= step
        np.clip(joint_cmd, self.robot.limits_min, self.robot.limits_max, out=joint_cmd)

    def make_record(self, joint_cmd: np.ndarray) -> dict[str, Any]:
        measured = self.last_measured.copy() if self.last_measured is not None else joint_cmd.copy()
        workspace = fk_workspace(self.robot, measured)
        gamma = workspace["gamma_rad"] if workspace is not None else None
        position = workspace["position_m"] if workspace is not None else None

        return {
            "point_index": self.selected_index + 1,
            "name": POINT_NAMES[self.selected_index],
            "image_px": round_list(self.image_points[self.selected_index], 3),
            "joint_cmd_rad": round_list(joint_cmd, 8),
            "joint_cmd_deg": round_list(np.rad2deg(joint_cmd), 4),
            "joint_measured_rad": round_list(measured, 8),
            "joint_measured_deg": round_list(np.rad2deg(measured), 4),
            "workspace_m": round_list(position, 8) if position is not None else None,
            "gamma_rad": round_float(gamma, 8) if gamma is not None else None,
            "gamma_deg": round_float(np.rad2deg(gamma), 4) if gamma is not None else None,
        }

    def robot_status_lines(
        self,
        joint_cmd: np.ndarray,
        records: list[dict[str, Any] | None],
    ) -> list[str]:
        selected = f"{self.selected_index + 1} {POINT_NAMES[self.selected_index]}"
        recorded_count = sum(record is not None for record in records)
        measured = self.last_measured if self.last_measured is not None else joint_cmd
        joint_text = ", ".join(f"{value:.1f}" for value in np.rad2deg(measured))
        lines = [
            f"Record robot point: {selected}",
            f"Recorded {recorded_count}/5. r records. n/p selects. Enter saves when complete.",
            "Hold arrows: yaw/shoulder | e/d: elbow | w/s: wrist | r: record",
            f"Jog step {self.joint_step_deg:.3f} deg | joints deg [{joint_text}]",
        ]
        if self.last_workspace is not None:
            pos = self.last_workspace["position_m"]
            lines.append(f"FK XYZ m [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        return lines


def initial_image_points(width: int, height: int, margin_ratio: float) -> list[list[float]]:
    margin_ratio = float(np.clip(margin_ratio, 0.02, 0.45))
    left = width * margin_ratio
    right = width * (1.0 - margin_ratio)
    top = height * margin_ratio
    bottom = height * (1.0 - margin_ratio)
    return [
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom],
        [width / 2.0, height / 2.0],
    ]


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def read_camera_frame(cap: cv2.VideoCapture, width: int, height: int) -> np.ndarray:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Failed to read from camera.")
    frame = np.ascontiguousarray(np.asarray(frame))
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def calibration_pose(robot: RobotContext, args: argparse.Namespace) -> np.ndarray:
    pose = robot.home_pose.copy()
    pose[3] = np.deg2rad(args.wrist_down_deg)
    np.clip(pose, robot.limits_min, robot.limits_max, out=pose)
    return pose


def move_to_initial_calibration_pose(robot: RobotContext, args: argparse.Namespace) -> np.ndarray:
    target = calibration_pose(robot, args)
    if robot.arm is None:
        print("Robot disabled; using home-with-wrist-down pose as commanded joint state.")
        return target

    print("Moving QArm Mini to standard home pose...")
    hold_pose(robot.arm, robot.home_pose, GRIPPER_OPEN, args.home_settle_seconds, args.sample_rate)
    print(
        "Moving to calibration home pose: "
        f"home joints with wrist={args.wrist_down_deg:.1f} deg for a downward camera view..."
    )
    interpolate_pose(
        robot.arm,
        robot.home_pose,
        target,
        GRIPPER_OPEN,
        args.pose_move_seconds,
        args.sample_rate,
    )
    return target


def hold_pose(arm: Any, pose: np.ndarray, gripper: float, seconds: float, rate_hz: float) -> None:
    steps = max(1, int(seconds * max(1.0, rate_hz)))
    delay = 1.0 / max(1.0, rate_hz)
    for _ in range(steps):
        arm.read_write_std(pose, gripper)
        time.sleep(delay)


def interpolate_pose(
    arm: Any,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    gripper: float,
    seconds: float,
    rate_hz: float,
) -> None:
    steps = max(1, int(seconds * max(1.0, rate_hz)))
    delay = 1.0 / max(1.0, rate_hz)
    for step_index in range(steps + 1):
        alpha = step_index / steps
        pose = start_pose + alpha * (target_pose - start_pose)
        arm.read_write_std(pose, gripper)
        time.sleep(delay)


def fk_workspace(robot: RobotContext, joints: np.ndarray | None) -> dict[str, Any] | None:
    if robot.arm_math is None or joints is None:
        return None
    try:
        position, rotation, gamma = robot.arm_math.forward_kinematics(joints)
    except Exception as exc:
        print(f"Warning: forward kinematics failed: {exc}")
        return None
    return {
        "position_m": np.asarray(position, dtype=float).reshape(-1)[:3],
        "gamma_rad": float(gamma),
    }


def next_unrecorded_index(records: list[dict[str, Any] | None], current_index: int) -> int:
    for offset in range(1, len(records) + 1):
        candidate = (current_index + offset) % len(records)
        if records[candidate] is None:
            return candidate
    return current_index


def is_robot_jog_key(key: int, char: str) -> bool:
    return key in KEY_LEFT or key in KEY_RIGHT or key in KEY_UP or key in KEY_DOWN or char in ("e", "d", "w", "s")


class WindowsAsyncKeyboard:
    def __init__(self):
        try:
            self.user32 = ctypes.windll.user32
            self.available = True
        except Exception:
            self.user32 = None
            self.available = False

    def is_down(self, virtual_key: int) -> bool:
        if not self.available:
            return False
        return bool(self.user32.GetAsyncKeyState(virtual_key) & 0x8000)


def draw_markers(
    frame: np.ndarray,
    points: list[list[float]],
    selected_index: int,
    recorded: list[bool],
) -> None:
    for index, point in enumerate(points):
        x, y = int(round(point[0])), int(round(point[1]))
        is_selected = index == selected_index
        color = (50, 220, 120) if recorded[index] else (40, 180, 255)
        if is_selected:
            color = (0, 255, 255)
        radius = 11 if is_selected else 8
        cv2.circle(frame, (x, y), radius, color, 2)
        cv2.drawMarker(frame, (x, y), color, cv2.MARKER_CROSS, 28, 2)
        cv2.putText(
            frame,
            f"{index + 1}:{POINT_NAMES[index]}",
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_panel(frame: np.ndarray, lines: list[str]) -> None:
    if not lines:
        return
    line_height = 22
    panel_height = 12 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 22 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def key_to_char(key: int) -> str:
    if key == -1:
        return ""
    low = key & 0xFF
    if low == 27:
        return "\x1b"
    if 0 <= low < 256:
        char = chr(low)
        if char.isprintable() or char in ("\r", "\n", " "):
            return char.lower()
    return ""


def build_calibration_document(
    args: argparse.Namespace,
    robot: RobotContext,
    image_points: list[list[float]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    transforms = compute_transforms(records)
    calibration_map = {}
    for index, record in enumerate(records, start=1):
        calibration_map[f"img{index}_to_space{index}"] = {
            "image_px": record["image_px"],
            "space_m": record["workspace_m"],
            "joint_measured_rad": record["joint_measured_rad"],
        }

    return {
        "created_at_unix": time.time(),
        "camera": {
            "index": args.camera_index,
            "width": args.frame_width,
            "height": args.frame_height,
        },
        "robot": {
            "joint_order": robot.joint_order,
            "home_pose_rad": round_list(robot.home_pose, 8),
            "home_pose_deg": round_list(np.rad2deg(robot.home_pose), 4),
            "calibration_pose_rad": round_list(calibration_pose(robot, args), 8),
            "calibration_pose_deg": round_list(np.rad2deg(calibration_pose(robot, args)), 4),
            "wrist_down_deg": args.wrist_down_deg,
            "connected": robot.arm is not None,
            "mode": "no_robot" if args.no_robot else ("virtual" if args.virtual else "hardware"),
        },
        "image_points": [
            {"point_index": i + 1, "name": POINT_NAMES[i], "image_px": round_list(point, 3)}
            for i, point in enumerate(image_points)
        ],
        "points": records,
        "calibration_map": calibration_map,
        "transforms": transforms,
    }


def compute_transforms(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid_records = [record for record in records if record.get("workspace_m") is not None]
    if len(valid_records) < 3:
        return {
            "status": "not_enough_workspace_points",
            "message": "Need at least 3 FK workspace points for an affine transform.",
        }

    image = np.asarray([record["image_px"] for record in valid_records], dtype=float)
    world = np.asarray([record["workspace_m"][:2] for record in valid_records], dtype=float)
    z_values = np.asarray([record["workspace_m"][2] for record in valid_records], dtype=float)

    affine = fit_affine(image, world)
    result = {
        "pixel_to_robot_xy_affine": round_matrix(affine["matrix"], 10),
        "affine_mean_error_m": round_float(affine["mean_error_m"], 8),
        "affine_max_error_m": round_float(affine["max_error_m"], 8),
        "table_z_mean_m": round_float(np.mean(z_values), 8),
        "table_z_std_m": round_float(np.std(z_values), 8),
    }
    inverse_affine = safe_invert_affine(affine["matrix"])
    if inverse_affine is not None:
        result["robot_xy_to_pixel_affine"] = round_matrix(inverse_affine, 10)

    if len(valid_records) >= 4:
        homography, _ = cv2.findHomography(image.astype(np.float32), world.astype(np.float32), 0)
        if homography is not None and np.isfinite(homography).all():
            projected = cv2.perspectiveTransform(
                image.reshape(1, -1, 2).astype(np.float32),
                homography,
            ).reshape(-1, 2)
            errors = np.linalg.norm(projected - world, axis=1)
            result.update(
                {
                    "pixel_to_robot_xy_homography": round_matrix(homography, 10),
                    "homography_mean_error_m": round_float(np.mean(errors), 8),
                    "homography_max_error_m": round_float(np.max(errors), 8),
                }
            )
            inverse_homography = safe_inverse_matrix(homography)
            if inverse_homography is not None:
                result["robot_xy_to_pixel_homography"] = round_matrix(inverse_homography, 10)

    return result


def fit_affine(image: np.ndarray, world: np.ndarray) -> dict[str, Any]:
    design = np.column_stack([image[:, 0], image[:, 1], np.ones(len(image))])
    coeff_x = np.linalg.lstsq(design, world[:, 0], rcond=None)[0]
    coeff_y = np.linalg.lstsq(design, world[:, 1], rcond=None)[0]
    matrix = np.vstack([coeff_x, coeff_y])
    predicted = np.column_stack([design @ coeff_x, design @ coeff_y])
    errors = np.linalg.norm(predicted - world, axis=1)
    return {
        "matrix": matrix,
        "mean_error_m": float(np.mean(errors)),
        "max_error_m": float(np.max(errors)),
    }


def safe_invert_affine(matrix: np.ndarray) -> np.ndarray | None:
    affine3 = np.vstack([matrix, np.array([0.0, 0.0, 1.0])])
    inverse = safe_inverse_matrix(affine3)
    if inverse is None:
        return None
    return inverse[:2, :]


def safe_inverse_matrix(matrix: np.ndarray) -> np.ndarray | None:
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(inverse).all():
        return None
    return inverse


def round_list(values: Any, decimals: int) -> list[float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    return [round(float(value), decimals) for value in array]


def round_matrix(values: Any, decimals: int) -> list[list[float]]:
    array = np.asarray(values, dtype=float)
    return [[round(float(value), decimals) for value in row] for row in array]


def round_float(value: Any, decimals: int) -> float:
    return round(float(value), decimals)


def print_record(record: dict[str, Any]) -> None:
    point = record["point_index"]
    image = record["image_px"]
    space = record["workspace_m"]
    joints = record["joint_measured_deg"]
    print(f"Recorded point {point} {record['name']}")
    print(f"  image: ({image[0]:.1f}, {image[1]:.1f}) px")
    if space is not None:
        print(f"  space: ({space[0]:.4f}, {space[1]:.4f}, {space[2]:.4f}) m")
    print(f"  joints deg: {joints}")


def print_summary(document: dict[str, Any], output_path: Path) -> None:
    print("\nCalibration map")
    print("=" * 32)
    for index, record in enumerate(document["points"], start=1):
        image = record["image_px"]
        space = record["workspace_m"]
        if space is None:
            space_text = "space unavailable"
        else:
            space_text = f"space{index}=({space[0]:.4f}, {space[1]:.4f}, {space[2]:.4f}) m"
        print(f"img{index}=({image[0]:.1f}, {image[1]:.1f}) px -> {space_text}")

    transforms = document.get("transforms", {})
    if "affine_mean_error_m" in transforms:
        print(f"Affine mean error: {transforms['affine_mean_error_m']:.6f} m")
    if "homography_mean_error_m" in transforms:
        print(f"Homography mean error: {transforms['homography_mean_error_m']:.6f} m")
    print(f"Saved calibration to {output_path}")


def main() -> None:
    args = parse_args()
    cap = None
    robot = None
    try:
        robot = RobotContext(args)
        joint_cmd = move_to_initial_calibration_pose(robot, args)
        cap = open_camera(args.camera_index, args.frame_width, args.frame_height)

        app = CalibrationApp(args, robot)
        image_points = app.align_image_points(cap)
        records = app.record_robot_points(cap, joint_cmd)

        document = build_calibration_document(args, robot, image_points, records)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        print_summary(document, output_path)
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        if robot is not None:
            robot.terminate()


if __name__ == "__main__":
    main()
