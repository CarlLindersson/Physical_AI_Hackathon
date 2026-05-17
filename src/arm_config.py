"""
QArmMini Configuration
Customize these values based on your specific robot setup
"""

# Arm specifications
ARM_CONFIG = {
    "port": "/dev/ttyUSB0",  # Change if your port differs (COM3 on Windows, /dev/ttyUSB0 on Linux/Mac)
    "baud_rate": 115200,
    "joint_limits": {
        "base": (-180, 180),
        "shoulder": (-90, 90),
        "elbow": (-90, 90),
        "wrist": (-180, 180),
        "pitch": (-90, 90),
        "roll": (-180, 180),
    },
    "gripper_force": 50,  # Percentage
}

# Object detection settings
DETECTION_CONFIG = {
    "min_confidence": 0.5,  # Minimum confidence score
    "max_distance_pixels": 50,  # Track object if within this distance
    "frame_width": 640,
    "frame_height": 480,
}

# Movement speeds
SPEED_CONFIG = {
    "base_speed": 30,  # degrees/sec
    "arm_speed": 20,
    "gripper_speed": 50,
}
