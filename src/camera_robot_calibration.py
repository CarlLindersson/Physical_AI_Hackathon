#!/usr/bin/env python3
"""Camera-to-robot calibration tool.

Click at least two known points in the camera view, then enter the robot world coordinates
for each point. The script computes a linear mapping:

    world_x = CAMERA_X_OFFSET + (pixel_x - frame_w/2) * PIXEL_TO_METER_X
    world_y = CAMERA_Y_OFFSET + (pixel_y - frame_h/2) * PIXEL_TO_METER_Y

This is the same mapping used by src/trash_sorting_detection.py.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

CLICK_INSTRUCTIONS = [
    "Click a point in the camera image where the object is located.",
    "Repeat for at least two distinct points.",
    "After clicking, enter the matching robot world X Y coordinates for each point.",
]

CLICK_COLOR = (0, 255, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate camera pixel to robot world mapping.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--frame-width", type=int, default=640, help="Camera frame width")
    parser.add_argument("--frame-height", type=int, default=480, help="Camera frame height")
    parser.add_argument("--points", type=int, default=3, help="Number of calibration points (min 2)")
    parser.add_argument("--output", type=str, default=".env.calibration", help="Path to save calibration variables")
    return parser.parse_args()


class CalibrationApp:
    def __init__(self, camera_index: int, frame_width: int, frame_height: int, points: int):
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.points = max(2, points)
        self.clicked = []
        self.image = None
        self.window_name = "Camera Calibration"

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicked) < self.points:
            self.clicked.append((x, y))
            print(f"Point {len(self.clicked)}: pixel=({x}, {y})")

    def capture_frame(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.camera_index}")

        print("Opening camera. Press space to capture a frame.")
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read from camera")
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                self.image = frame.copy()
                break
            elif key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit("Calibration cancelled")

        cap.release()
        cv2.destroyWindow(self.window_name)

    def collect_clicks(self):
        if self.image is None:
            raise RuntimeError("No frame available")

        self.image = cv2.resize(self.image, (self.frame_width, self.frame_height))
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\nCLICK INSTRUCTIONS")
        for line in CLICK_INSTRUCTIONS:
            print(f"- {line}")
        print(f"Click {self.points} points in the image and then press q.")

        while len(self.clicked) < self.points:
            display = self.image.copy()
            for idx, (x, y) in enumerate(self.clicked, start=1):
                cv2.circle(display, (x, y), 6, CLICK_COLOR, -1)
                cv2.putText(display, str(idx), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLICK_COLOR, 2)
            cv2.imshow(self.window_name, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        if len(self.clicked) < 2:
            raise RuntimeError("Need at least two calibration clicks.")

    def query_robot_points(self):
        robot_points = []
        for i, (px, py) in enumerate(self.clicked, start=1):
            while True:
                line = input(f"Enter robot X Y for point {i} at pixel ({px},{py}): ")
                try:
                    xw, yw = [float(v) for v in line.strip().split()[:2]]
                    robot_points.append((xw, yw))
                    break
                except Exception:
                    print("Invalid entry. Type two numbers separated by space.")
        return robot_points

    def compute_mapping(self, robot_points):
        center_x = self.frame_width / 2.0
        center_y = self.frame_height / 2.0

        pixel_deltas = []
        world_x = []
        world_y = []
        for (px, py), (xw, yw) in zip(self.clicked, robot_points):
            pixel_deltas.append((px - center_x, py - center_y))
            world_x.append(xw)
            world_y.append(yw)

        pixel_deltas = np.array(pixel_deltas, dtype=float)
        world_x = np.array(world_x, dtype=float)
        world_y = np.array(world_y, dtype=float)

        A_x = np.column_stack([pixel_deltas[:, 0], np.ones(len(pixel_deltas))])
        A_y = np.column_stack([pixel_deltas[:, 1], np.ones(len(pixel_deltas))])

        ox, bx = np.linalg.lstsq(A_x, world_x, rcond=None)[0]
        oy, by = np.linalg.lstsq(A_y, world_y, rcond=None)[0]

        return ox, bx, oy, by

    def print_and_save(self, output_path: Path, ox, bx, oy, by):
        print("\nCalibration results:")
        print(f"CAMERA_X_OFFSET={bx:.6f}")
        print(f"CAMERA_Y_OFFSET={by:.6f}")
        print(f"PIXEL_TO_METER_X={ox:.6f}")
        print(f"PIXEL_TO_METER_Y={oy:.6f}")
        print(f"FRAME_WIDTH={self.frame_width}")
        print(f"FRAME_HEIGHT={self.frame_height}")

        if output_path:
            content = (
                f"CAMERA_X_OFFSET={bx:.6f}\n"
                f"CAMERA_Y_OFFSET={by:.6f}\n"
                f"PIXEL_TO_METER_X={ox:.6f}\n"
                f"PIXEL_TO_METER_Y={oy:.6f}\n"
                f"CAMERA_WIDTH={self.frame_width}\n"
                f"CAMERA_HEIGHT={self.frame_height}\n"
            )
            output_path.write_text(content)
            print(f"Saved calibration to {output_path}")

    def run(self, output_file: str):
        self.capture_frame()
        self.collect_clicks()
        robot_points = self.query_robot_points()
        ox, bx, oy, by = self.compute_mapping(robot_points)
        self.print_and_save(Path(output_file), ox, bx, oy, by)


def main():
    args = parse_args()
    app = CalibrationApp(
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        points=args.points,
    )
    app.run(args.output)


if __name__ == "__main__":
    main()
