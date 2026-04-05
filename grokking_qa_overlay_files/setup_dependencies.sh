#!/bin/bash
# Quick dependency installer for QA overlay

echo "Installing QA Overlay Dependencies..."
echo ""

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found. Please install Python pip first."
    exit 1
fi

# Install core dependencies
echo "Installing PyTorch (CPU version for quick setup)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Installing data/plotting libraries..."
pip install pandas matplotlib numpy

echo ""
echo "✓ Dependencies installed!"
echo ""
echo "Quick test:"
python -c "import torch; import pandas; import matplotlib; print('✓ All imports successful')"
echo ""
echo "Ready to run: python test_qa_logger.py"
