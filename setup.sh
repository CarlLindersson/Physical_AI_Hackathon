#!/bin/bash
# Quick setup script for QArmMini Vision Demo

set -e  # Exit on any error

echo "============================================"
echo "QArmMini Vision Demo - Setup"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Create venv
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "============================================"
echo "✓ Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Activate venv: source .venv/bin/activate"
echo "2. Find robot port: ls /dev/tty* | grep usb"
echo "3. Update port in src/arm_config.py"
echo "4. Test camera: python src/robot_vision_demo.py --no-robot"
echo "5. Run demo: python src/robot_vision_demo.py"
echo ""
