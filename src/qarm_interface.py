#!/usr/bin/env python3
"""
QArmMini Robot Interface
Simple wrapper to send pick/place commands to the robot
Works with both serial connection and optional Quanser SDK
"""

import serial
import time
from typing import Optional, Dict

GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 1.0


class QArmMiniRobot:
    """
    Interface to QArmMini robot
    Can work with Quanser SDK (if available) or direct serial
    """
    
    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200, use_sdk: bool = False):
        """
        Initialize robot connection
        
        Args:
            port: Serial port (e.g., /dev/ttyUSB0, COM3)
            baud: Baud rate (usually 115200)
            use_sdk: Try to use Quanser SDK if available
        """
        self.port = port
        self.baud = baud
        self.connected = False
        self.use_sdk = use_sdk
        self.arm = None
        self.ser = None
        self.joint_cmd = None
        self.gripper_cmd = GRIPPER_OPEN
        
        # Try Quanser SDK first if requested
        if use_sdk:
            try:
                from pal.products.qarm_mini import QArmMini
                print("✓ Using Quanser QArm Mini SDK")
                self.arm = QArmMini(hardware=1, id=3)
                self.joint_cmd = self.arm.HOME_POSE.copy()
                self.gripper_cmd = GRIPPER_OPEN
                self.connected = True
                return
            except ImportError:
                print("⚠ Quanser SDK not available, falling back to serial")
            except Exception as e:
                print(f"⚠ Quanser SDK initialization failed: {e}")
                print("  Falling back to serial")
        
        # Fall back to serial connection
        try:
            self.ser = serial.Serial(port, baud, timeout=1)
            self.connected = True
            print(f"✓ Connected to robot at {port} ({baud} baud)")
            time.sleep(1)
        except Exception as e:
            print(f"✗ Failed to connect to robot: {e}")
            print(f"  Make sure robot is connected to {port}")
            self.connected = False
    
    def send_command(self, cmd: str) -> str:
        """Send a single command string to the robot."""
        if not self.connected:
            return ""

        try:
            if self.arm:
                if hasattr(self.arm, "send_command"):
                    response = self.arm.send_command(cmd)
                    print(f"[SDK CMD] {cmd} → {response}")
                    return response
                print(f"[SDK CMD] {cmd}")
                return "OK"
            self.ser.write(cmd.encode() + b'\r\n')
            response = self.ser.readline().decode().strip()
            return response
        except Exception as e:
            print(f"Serial error: {e}")
            return ""

    def move_to(self, x: float, y: float, z: float) -> bool:
        """Move the robot end effector to an absolute XYZ pose."""
        if not self.connected:
            return False

        if self.arm:
            if hasattr(self.arm, "move_cartesian"):
                self.arm.move_cartesian(x, y, z)
                return True
            if hasattr(self.arm, "move_to"):
                self.arm.move_to(x, y, z)
                return True
            print("⚠ SDK does not expose a cartesian move method")
            return False

        cmd = f"MOVE {x:.3f} {y:.3f} {z:.3f}"
        response = self.send_command(cmd)
        print(f"[ROBOT CMD] {cmd} → {response}")
        time.sleep(0.5)
        return True

    def pick_and_place(
        self,
        x: float,
        y: float,
        z: float,
        zone: str,
        zone_coords: Optional[Dict[str, float]] = None,
        grip_strength: float = GRIPPER_CLOSED,
        approach_offset: float = 0.08,
        lift_offset: float = 0.12,
        return_home: bool = True,
    ) -> bool:
        """
        Send pick and place command to robot
        
        Args:
            x, y, z: Object location
            zone: Target zone (e.g., "ZONE_A", "ZONE_B")
        
        Returns:
            True if command sent successfully
        """
        if not self.connected:
            return False

        zone_coords = zone_coords or {"x": 0.0, "y": 0.0, "z": z}
        approach_z = z + approach_offset
        lift_z = z + lift_offset
        deposit_z = zone_coords.get("z", z)
        deposit_approach_z = deposit_z + approach_offset

        try:
            if self.arm:
                if not self.move_to(x, y, approach_z):
                    return False
                if not self.move_to(x, y, z):
                    return False
                self.gripper_close(grip_strength)
                if not self.move_to(x, y, lift_z):
                    return False
                if not self.move_to(zone_coords["x"], zone_coords["y"], deposit_approach_z):
                    return False
                if not self.move_to(zone_coords["x"], zone_coords["y"], deposit_z):
                    return False
                self.gripper_open()
                if return_home:
                    self.home()
                return True

            self.move_to(x, y, approach_z)
            self.move_to(x, y, z)
            self.gripper_close(grip_strength)
            self.move_to(x, y, lift_z)
            self.move_to(zone_coords['x'], zone_coords['y'], deposit_approach_z)
            self.move_to(zone_coords['x'], zone_coords['y'], deposit_z)
            self.gripper_open()
            self.move_to(zone_coords['x'], zone_coords['y'], deposit_approach_z)
            if return_home:
                self.home()
            return True
        except Exception as e:
            print(f"Robot command error: {e}")
            return False
    
    def home(self) -> bool:
        """Move robot to home position"""
        if not self.connected:
            return False
        
        try:
            if self.arm:
                if hasattr(self.arm, "read_write_std") and self.joint_cmd is not None:
                    self.joint_cmd = self.arm.HOME_POSE.copy()
                    self.arm.read_write_std(self.joint_cmd, self.gripper_cmd)
                    print("[SDK CMD] HOME")
                    return True
                if hasattr(self.arm, "home"):
                    self.arm.home()
                    print("[SDK CMD] HOME")
                    return True
                print("[SDK CMD] HOME")
                return True
            else:
                response = self.send_command("HOME")
                print(f"[ROBOT CMD] HOME → {response}")
                return True
        except Exception as e:
            print(f"Home command error: {e}")
            return False
    
    def gripper_open(self) -> bool:
        """Open gripper."""
        if not self.connected:
            return False

        try:
            self.gripper_cmd = GRIPPER_OPEN
            if self.arm:
                if hasattr(self.arm, "read_write_std"):
                    self.arm.read_write_std(self.joint_cmd, self.gripper_cmd)
                    print(f"[SDK CMD] GRIP OPEN ({self.gripper_cmd:.2f})")
                    return True
                if hasattr(self.arm, "set_gripper"):
                    self.arm.set_gripper(self.gripper_cmd)
                    print(f"[SDK CMD] GRIP OPEN ({self.gripper_cmd:.2f})")
                    return True
                print("[SDK CMD] GRIP OPEN")
                return True

            response = self.send_command("GRIP OPEN")
            print(f"[ROBOT CMD] GRIPPER OPEN → {response}")
            return True
        except Exception as e:
            print(f"Gripper error: {e}")
            return False

    def gripper_close(self, strength: float = GRIPPER_CLOSED) -> bool:
        """Close gripper with optional strength."""
        if not self.connected:
            return False

        self.gripper_cmd = min(max(strength, 0.0), 1.0)
        try:
            if self.arm:
                if hasattr(self.arm, "read_write_std"):
                    self.arm.read_write_std(self.joint_cmd, self.gripper_cmd)
                    print(f"[SDK CMD] GRIP CLOSE ({self.gripper_cmd:.2f})")
                    return True
                if hasattr(self.arm, "set_gripper"):
                    self.arm.set_gripper(self.gripper_cmd)
                    print(f"[SDK CMD] GRIP CLOSE ({self.gripper_cmd:.2f})")
                    return True
                print(f"[SDK CMD] GRIP CLOSE ({self.gripper_cmd:.2f})")
                return True

            response = self.send_command("GRIP CLOSE")
            print(f"[ROBOT CMD] GRIPPER CLOSE → {response}")
            return True
        except Exception as e:
            print(f"Gripper error: {e}")
            return False
    
    def close(self):
        """Close connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial connection closed")
    
    def __del__(self):
        self.close()
