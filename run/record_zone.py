#!/usr/bin/env python3
"""Record current QArm joint pose for a named zone (ZONE_A/B/C).

Usage:
  python run/record_zone.py ZONE_A

This reads the robot's measured joint angles (radians) and FK XYZ,
and saves/updates `run/zones.json` with the entry for the zone.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

# Import local robot wrapper
from src.qarm_interface import QArmMiniRobot


def parse_args():
    parser = argparse.ArgumentParser(description="Record current robot pose for a zone")
    parser.add_argument("zone", choices=("ZONE_A", "ZONE_B", "ZONE_C"), help="Zone to record")
    parser.add_argument("--output", default="run/zones.json", help="Output JSON file")
    parser.add_argument("--use-sdk", action="store_true", help="Prefer Quanser SDK connection when available")
    parser.add_argument("--port", default=os.environ.get("ROBOT_PORT", "/dev/ttyUSB0"), help="Serial port if using serial fallback")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--no-robot", action="store_true", help="Do not connect to robot (for testing)")
    return parser.parse_args()


def load_zones(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_zones(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_measured_joints(robot: QArmMiniRobot) -> np.ndarray:
    # Try SDK measured readout
    if robot.arm is not None:
        try:
            measured = np.asarray(getattr(robot.arm, "positionMeasured", None), dtype=float)
            if measured is not None and measured.size >= 4 and np.isfinite(measured[:4]).all():
                return measured[:4].copy()
        except Exception:
            pass

    # Fall back to last commanded joints stored on wrapper
    if robot.joint_cmd is not None:
        return np.asarray(robot.joint_cmd, dtype=float).copy()

    raise RuntimeError("Could not read robot joint state")


def compute_fk(joints: np.ndarray):
    try:
        from hal.content.qarm_mini import QArmMiniFunctions
        arm_math = QArmMiniFunctions()
        pos, rot, gamma = arm_math.forward_kinematics(joints)
        return np.asarray(pos, dtype=float).tolist()
    except Exception:
        return None


def main():
    args = parse_args()
    out_path = Path(args.output)

    if args.no_robot:
        print("--no-robot specified; exiting")
        sys.exit(1)

    robot = QArmMiniRobot(port=args.port, baud=args.baud, use_sdk=args.use_sdk)
    if not robot.connected:
        print("Could not connect to robot. Aborting.")
        sys.exit(1)

    try:
        joints = get_measured_joints(robot)
    except Exception as exc:
        print(f"Error reading joints: {exc}")
        sys.exit(1)

    workspace = compute_fk(joints)

    entry = {
        "joint_cmd_rad": [float(x) for x in list(joints)],
        "joint_cmd_deg": [float(x) for x in list(np.rad2deg(joints))],
        "workspace_m": workspace,
    }

    zones = load_zones(out_path)
    zones[args.zone] = entry
    save_zones(out_path, zones)

    print(f"Recorded {args.zone} → {out_path}")
    print(json.dumps({args.zone: entry}, indent=2))


if __name__ == "__main__":
    main()
