#!/bin/bash
# install_ml_deps.sh — Install ML dependencies for the pipeline

echo "════════════════════════════════════════════════════════"
echo "  Installing ML Dependencies for Plant NDVI Pipeline"
echo "════════════════════════════════════════════════════════"
echo ""

# Navigate to base directory
cd /Users/nithints/Drone_imagery_file

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo ""
    echo "Please create it first:"
    echo "  python3 -m venv .venv"
    echo ""
    exit 1
fi

echo "✓ Found virtual environment"
echo ""

# Activate virtual environment
source .venv/bin/activate

echo "Installing required packages..."
echo ""

# Install packages
pip install --upgrade pip
pip install pandas scikit-learn xgboost lightgbm

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Installation Complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Installed packages:"
echo "  • pandas        - Data manipulation"
echo "  • scikit-learn  - Random Forest models"
echo "  • xgboost       - XGBoost models"
echo "  • lightgbm      - LightGBM models"
echo ""
echo "Next steps:"
echo "  1. Run the full pipeline:"
echo "     .venv/bin/python plant_ndvi_pipeline/run_pipeline.py"
echo ""
echo "  2. Or run just the ML steps:"
echo "     .venv/bin/python plant_ndvi_pipeline/run_ml_pipeline.py"
echo ""
