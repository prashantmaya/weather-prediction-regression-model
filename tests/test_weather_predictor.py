"""Unit tests for WeatherPredictor class."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.weather_prediction.model import WeatherPredictor


@pytest.fixture
def sample_data():
    """Create sample weather data for testing."""
    return pd.DataFrame({
        'Location': ['Sydney', 'Melbourne', 'Brisbane'],
        'MinTemp': [15.0, 12.0, 18.0],
        'MaxTemp': [25.0, 22.0, 28.0],
        'Rainfall': [0.0, 2.0, 0.0],
        'Evaporation': [4.0, 3.0, 5.0],
        'Sunshine': [8.0, 6.0, 9.0],
        'WindGustDir': ['N', 'S', 'E'],
        'WindGustSpeed': [35.0, 40.0, 30.0],
        'WindDir9am': ['N', 'S', 'E'],
        'WindDir3pm': ['NE', 'SW', 'E'],
        'WindSpeed9am': [15.0, 20.0, 10.0],
        'WindSpeed3pm': [20.0, 25.0, 15.0],
        'Humidity9am': [70.0, 75.0, 65.0],
        'Humidity3pm': [60.0, 65.0, 55.0],
        'Pressure9am': [1015.0, 1012.0, 1018.0],
        'Pressure3pm': [1013.0, 1010.0, 1016.0],
        'Cloud9am': [5.0, 6.0, 4.0],
        'Cloud3pm': [4.0, 5.0, 3.0],
        'Temp9am': [20.0, 18.0, 22.0],
        'Temp3pm': [24.0, 21.0, 27.0],
        'RainToday': ['No', 'Yes', 'No'],
        'RainTomorrow': ['No', 'Yes', 'No']
    })


def test_initialization():
    """Test WeatherPredictor initialization."""
    predictor = WeatherPredictor()
    assert predictor is not None
    assert not predictor.is_fitted
    assert len(predictor.numerical_cols) == 16
    assert len(predictor.categorical_cols) == 5


def test_fit(sample_data):
    """Test model fitting."""
    predictor = WeatherPredictor()
    predictor.fit(sample_data)
    
    assert predictor.is_fitted
    assert 'train' in predictor.metrics
    assert 'accuracy' in predictor.metrics['train']


def test_predict_before_fit(sample_data):
    """Test that prediction raises error before fitting."""
    predictor = WeatherPredictor()
    test_input = sample_data.drop('RainTomorrow', axis=1).iloc[[0]]
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        predictor.predict(test_input)


def test_predict_after_fit(sample_data):
    """Test prediction after fitting."""
    predictor = WeatherPredictor()
    predictor.fit(sample_data)
    
    test_input = sample_data.drop('RainTomorrow', axis=1).iloc[[0]]
    prediction, probability = predictor.predict(test_input)
    
    assert prediction in ['Yes', 'No']
    assert 0 <= probability <= 1


def test_save_and_load(sample_data):
    """Test saving and loading model."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "test_model"
        
        # Train and save
        predictor = WeatherPredictor()
        predictor.fit(sample_data)
        predictor.save(str(model_dir))
        
        # Check files exist
        assert (model_dir / "imputer.joblib").exists()
        assert (model_dir / "scaler.joblib").exists()
        assert (model_dir / "encoder.joblib").exists()
        assert (model_dir / "model.joblib").exists()
        
        # Load and test
        loaded_predictor = WeatherPredictor.load(str(model_dir))
        assert loaded_predictor.is_fitted
        
        test_input = sample_data.drop('RainTomorrow', axis=1).iloc[[0]]
        prediction, probability = loaded_predictor.predict(test_input)
        
        assert prediction in ['Yes', 'No']
        assert 0 <= probability <= 1


def test_missing_columns():
    """Test error handling for missing columns."""
    predictor = WeatherPredictor()
    invalid_data = pd.DataFrame({'Location': ['Sydney']})
    
    with pytest.raises(ValueError, match="Missing required columns"):
        predictor._validate_data(invalid_data, is_training=True)
```