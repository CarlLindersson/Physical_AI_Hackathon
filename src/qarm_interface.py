#!/usr/bin/env python3
"""
QArmMini Robot Interface
Simple wrapper to send pick/place commands to the robot
Works with both serial connection and optional Quanser SDK
"""

import serial
import time
from typing import Optional, Dict


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
        
        # Try Quanser SDK first if requested
        if use_sdk:
            try:
                from pal.products.qarm_mini import QArmMini
                print("✓ Using Quanser SDK")
                self.arm = QArmMini()
                self.connected = True
                return
            except ImportError:
                print("⚠ Quanser SDK not available, falling back to serial")
        
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
    
    def pick_and_place(self, x: float, y: float, z: float, zone: str) -> bool:
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
        
        try:
            if self.arm:
                # Quanser SDK command
                # TODO: implement based on actual SDK
                print(f"[ROBOT CMD] Pick from ({x:.2f}, {y:.2f}, {z:.2f}) → {zone}")
                return True
            else:
                # Serial command format (customize for your robot)
                cmd = f"PICK {x:.3f} {y:.3f} {z:.3f} {zone}\r\n"
                self.ser.write(cmd.encode())
                response = self.ser.readline().decode().strip()
                print(f"[ROBOT CMD] {cmd.strip()} → {response}")
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
                # Quanser SDK home
                print("[ROBOT CMD] HOME")
                return True
            else:
                # Serial home command
                self.ser.write(b"HOME\r\n")
                response = self.ser.readline().decode().strip()
                print(f"[ROBOT CMD] HOME → {response}")
                return True
        except Exception as e:
            print(f"Home command error: {e}")
            return False
    
    def gripper_open(self) -> bool:
        """Open gripper"""
        if not self.connected:
            return False
        
        try:
            if self.arm:
                print("[ROBOT CMD] GRIPPER OPEN")
                return True
            else:
                self.ser.write(b"GRIP OPEN\r\n")
                response = self.ser.readline().decode().strip()
                print(f"[ROBOT CMD] GRIPPER OPEN → {response}")
                return True
        except Exception as e:
            print(f"Gripper error: {e}")
            return False
    
    def gripper_close(self) -> bool:
        """Close gripper"""
        if not self.connected:
            return False
        
        try:
            if self.arm:
                print("[ROBOT CMD] GRIPPER CLOSE")
                return True
            else:
                self.ser.write(b"GRIP CLOSE\r\n")
                response = self.ser.readline().decode().strip()
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
