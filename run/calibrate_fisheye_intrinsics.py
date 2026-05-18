#!/usr/bin/env python3
"""Capture checkerboard images and calibrate fisheye camera intrinsics.

Outputs a JSON file containing the camera matrix K, fisheye distortion
coefficients D, RMS reprojection error, image size, and checkerboard metadata.
Use the same camera resolution here that you use in the robot picking scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np


WINDOW_NAME = "QArm Fisheye Intrinsic Calibration"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
DEFAULT_OUTPUT = "run/intrinsic_calibration.json"


def parse_args() -> argparse.Namespace:
    load_local_env()
    parser = argparse.ArgumentParser(
        description="Calibrate fisheye camera intrinsics using a checkerboard."
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
        "--cols",
        type=int,
        default=9,
        help="Number of inner checkerboard corners across columns.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Number of inner checkerboard corners across rows.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help="Checkerboard square size in any unit. Use meters if you want extrinsics in meters.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=20,
        help="Recommended minimum number of captured checkerboard views.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="JSON output path for intrinsic calibration variables.",
    )
    parser.add_argument(
        "--save-frames-dir",
        default=None,
        help="Optional folder to save captured checkerboard frames.",
    )
    parser.add_argument(
        "--auto-capture",
        action="store_true",
        help="Automatically capture stable detections instead of pressing Space/C.",
    )
    parser.add_argument(
        "--auto-delay-seconds",
        type=float,
        default=0.8,
        help="Minimum delay between auto-captures.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=60,
        help="Stop auto-capture after this many samples.",
    )
    parser.add_argument(
        "--subpix-window",
        type=int,
        default=3,
        help="Half-window size for cornerSubPix refinement.",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.0,
        help="Balance for preview undistortion after calibration: 0 crops more, 1 keeps more FOV.",
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


def checkerboard_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    points = np.zeros((1, cols * rows, 3), np.float64)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[0, :, :2] = grid * square_size
    return points


def detect_checkerboard(
    frame: np.ndarray,
    pattern_size: tuple[int, int],
    subpix_window: int,
) -> tuple[bool, np.ndarray | None, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None, gray

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        1e-6,
    )
    window = (max(1, subpix_window), max(1, subpix_window))
    corners = cv2.cornerSubPix(gray, corners, window, (-1, -1), criteria)
    corners = corners.reshape(1, -1, 2).astype(np.float64)
    return True, corners, gray


def draw_status(
    frame: np.ndarray,
    lines: list[str],
    found: bool,
) -> None:
    panel_height = 22 * len(lines) + 12
    overlay = frame.copy()
    color = (20, 80, 35) if found else (20, 20, 20)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], panel_height), color, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (10, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def capture_samples(args: argparse.Namespace) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    cap = open_camera(args.camera_index, args.frame_width, args.frame_height)
    pattern_size = (args.cols, args.rows)
    object_template = checkerboard_object_points(args.cols, args.rows, args.square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    save_dir = Path(args.save_frames_dir) if args.save_frames_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    last_auto_capture = 0.0
    calibrated_preview = None
    undistort_preview = False

    print("\nFisheye intrinsic calibration")
    print("=" * 36)
    print(f"Checkerboard inner corners: {args.cols} x {args.rows}")
    print("Move the checkerboard around the image, especially near the edges.")
    print("Press Space or c to capture a detected view.")
    print("Press Enter to calibrate after enough samples, u to toggle undistorted preview after calibration, q to quit.\n")

    try:
        while True:
            frame = read_camera_frame(cap, args.frame_width, args.frame_height)
            display = frame.copy()
            found, corners, _ = detect_checkerboard(frame, pattern_size, args.subpix_window)

            if found and corners is not None:
                cv2.drawChessboardCorners(display, pattern_size, corners.reshape(-1, 1, 2), found)

            if undistort_preview and calibrated_preview is not None:
                display = undistort_frame(display, calibrated_preview, args.balance)

            lines = [
                f"Samples: {len(image_points)}/{args.min_samples} recommended",
                "Space/c capture | Enter calibrate | u preview | q quit",
                "Use center, edges, corners, tilted views, and varied distances",
                "Detected checkerboard" if found else "No checkerboard detected",
            ]
            draw_status(display, lines, found)
            cv2.imshow(WINDOW_NAME, display)

            now = time.time()
            if (
                args.auto_capture
                and found
                and corners is not None
                and now - last_auto_capture >= args.auto_delay_seconds
                and len(image_points) < args.max_samples
            ):
                capture_sample(object_points, image_points, object_template, corners, frame, save_dir)
                last_auto_capture = now
                print(f"Captured sample {len(image_points)}")
                if len(image_points) >= args.max_samples:
                    print("Reached max samples; calibrating.")
                    break

            key = cv2.waitKeyEx(1)
            char = key_to_char(key)
            if char in ("q", "\x1b"):
                raise SystemExit("Intrinsic calibration cancelled.")
            if char in (" ", "c"):
                if not found or corners is None:
                    print("No checkerboard detected; sample not captured.")
                    continue
                capture_sample(object_points, image_points, object_template, corners, frame, save_dir)
                print(f"Captured sample {len(image_points)}")
            elif char in ("\r", "\n"):
                break
            elif char == "u":
                if calibrated_preview is None and len(image_points) >= 3:
                    calibrated_preview = calibrate_fisheye(
                        object_points,
                        image_points,
                        (args.frame_width, args.frame_height),
                    )
                undistort_preview = not undistort_preview

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return object_points, image_points, (args.frame_width, args.frame_height)


def capture_sample(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    object_template: np.ndarray,
    corners: np.ndarray,
    frame: np.ndarray,
    save_dir: Path | None,
) -> None:
    object_points.append(object_template.copy())
    image_points.append(corners.copy())
    if save_dir is not None:
        output = save_dir / f"checkerboard_{len(image_points):03d}.png"
        cv2.imwrite(str(output), frame)


def calibrate_fisheye(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    if len(image_points) < 3:
        raise RuntimeError("Need at least 3 checkerboard captures to calibrate.")

    k = np.zeros((3, 3), dtype=np.float64)
    d = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in object_points]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in object_points]
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-6,
    )
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )

    try:
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            k,
            d,
            rvecs,
            tvecs,
            flags,
            criteria,
        )
    except cv2.error:
        relaxed_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, k, d, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            k,
            d,
            rvecs,
            tvecs,
            relaxed_flags,
            criteria,
        )

    return {
        "rms": float(rms),
        "K": k,
        "D": d,
        "rvecs": rvecs,
        "tvecs": tvecs,
    }


def undistort_frame(frame: np.ndarray, calibration: dict[str, Any], balance: float) -> np.ndarray:
    h, w = frame.shape[:2]
    k = calibration["K"]
    d = calibration["D"]
    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k,
        d,
        (w, h),
        np.eye(3),
        balance=float(np.clip(balance, 0.0, 1.0)),
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        k,
        d,
        np.eye(3),
        new_k,
        (w, h),
        cv2.CV_16SC2,
    )
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)


def build_output_document(
    args: argparse.Namespace,
    image_size: tuple[int, int],
    calibration: dict[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    return {
        "camera_model": "opencv_fisheye",
        "created_at_unix": time.time(),
        "camera": {
            "index": args.camera_index,
            "width": image_size[0],
            "height": image_size[1],
        },
        "checkerboard": {
            "inner_corners_cols": args.cols,
            "inner_corners_rows": args.rows,
            "square_size": args.square_size,
        },
        "sample_count": sample_count,
        "rms_reprojection_error": calibration["rms"],
        "K": round_matrix(calibration["K"], 12),
        "D": round_matrix(calibration["D"], 12),
        "notes": [
            "Use cv2.fisheye.undistortPoints for object bbox centers before table mapping.",
            "Calibration is resolution-specific; keep run_model camera size identical.",
        ],
    }


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


def round_matrix(values: Any, decimals: int) -> list[list[float]]:
    array = np.asarray(values, dtype=float)
    return [[round(float(value), decimals) for value in row] for row in array]


def main() -> None:
    args = parse_args()
    object_points, image_points, image_size = capture_samples(args)
    if len(image_points) < args.min_samples:
        print(
            f"Warning: only {len(image_points)} samples captured; "
            f"{args.min_samples}+ is recommended for fisheye calibration."
        )

    calibration = calibrate_fisheye(object_points, image_points, image_size)
    output = build_output_document(args, image_size, calibration, len(image_points))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nIntrinsic calibration complete")
    print("=" * 36)
    print(f"Samples: {len(image_points)}")
    print(f"RMS reprojection error: {calibration['rms']:.6f}")
    print(f"K:\n{calibration['K']}")
    print(f"D:\n{calibration['D'].reshape(-1)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
