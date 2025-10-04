#!/bin/bash

# Script to move old files to notebooks and clean up

echo "Setting up new project structure..."

# Create notebooks directory if it doesn't exist
mkdir -p notebooks

# Move notebook to notebooks directory
if [ -f "weather_prediction_system.ipynb" ]; then
    mv weather_prediction_system.ipynb notebooks/
    echo "✓ Moved notebook to notebooks/"
fi

# Remove old Python files (they're now in src/)
if [ -f "app.py" ] && [ -f "src/weather_prediction/api.py" ]; then
    rm app.py
    echo "✓ Removed old app.py"
fi

if [ -f "weather_predictor.py" ] && [ -f "src/weather_prediction/model.py" ]; then
    rm weather_predictor.py
    echo "✓ Removed old weather_predictor.py"
fi

if [ -f "config.py" ] && [ -f "src/weather_prediction/config.py" ]; then
    rm config.py
    echo "✓ Removed old config.py"
fi

if [ -f "train.py" ] && [ -f "scripts/train.py" ]; then
    rm train.py
    echo "✓ Removed old train.py"
fi

# Remove intermediate CSV files
echo "Cleaning up intermediate data files..."
rm -f data/test_dfs.csv data/train_inputs.csv data/train_target.csv 
rm -f data/validation_inputs.csv data/validation_target.csv
echo "✓ Removed intermediate CSV files"

# Create directories
mkdir -p logs model

echo "✓ Project structure updated successfully!"
echo ""
echo "New structure:"
tree -L 2 -I '__pycache__|*.pyc|.git|venv|env|.venv|model|data'