#!/bin/bash
# ==============================================================================
# VM Setup Script - Fix NumPy Compatibility
# ==============================================================================
# Run this FIRST on the VM before running experiments
# ==============================================================================

set -e

echo "=========================================="
echo "SETTING UP ENVIRONMENT"
echo "=========================================="

# Check Python version
echo "Python version: $(python --version 2>&1)"

# Fix NumPy compatibility issue
echo ""
echo "Fixing NumPy compatibility..."
echo "Current NumPy version: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not installed')"

# Uninstall NumPy 2.x if present
pip uninstall -y numpy 2>/dev/null || true

# Install requirements with correct NumPy version
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="
python -c "
import sys
print('Python:', sys.version)

import numpy
print('NumPy:', numpy.__version__)
assert numpy.__version__ < '2.0.0', 'NumPy must be <2.0'

import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

import torchvision
print('TorchVision:', torchvision.__version__)

import cv2
print('OpenCV:', cv2.__version__)

import scipy
print('SciPy:', scipy.__version__)

import matplotlib
print('Matplotlib:', matplotlib.__version__)

print()
print('✓ All dependencies installed correctly!')
"

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "You can now run experiments:"
echo "  ./run_experiments.sh          (sequential, safer)"
echo "  ./run_experiments_fast.sh     (parallel, faster)"
echo ""
