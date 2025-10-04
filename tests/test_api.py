"""Tests for FastAPI application."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def mock_model():
    """Create a mock weather predictor."""
    mock = Mock()
    mock.predict.return_value = ("Yes", 0.85)
    mock.numerical_cols = ['MinTemp', 'MaxTemp']
    mock.categorical_cols = ['Location', 'RainToday']
    mock.target_col = 'RainTomorrow'
    return mock


@pytest.fixture
def client(mock_model):
    """Create test client with mocked model."""
    with patch('src.weather_prediction.api.WeatherPredictor.load', return_value=mock_model):
        from src.weather_prediction.api import app
        with TestClient(app) as test_client:
            yield test_client


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_endpoint(client):
    """Test prediction endpoint."""
    payload = {
        "Location": "Sydney",
        "MinTemp": 18.0,
        "MaxTemp": 25.0,
        "Rainfall": 0.0,
        "Evaporation": 4.0,
        "Sunshine": 8.0,
        "WindGustDir": "NE",
        "WindGustSpeed": 35.0,
        "WindDir9am": "N",
        "WindDir3pm": "NE",
        "WindSpeed9am": 15.0,
        "WindSpeed3pm": 20.0,
        "Humidity9am": 70.0,
        "Humidity3pm": 60.0,
        "Pressure9am": 1015.0,
        "Pressure3pm": 1013.0,
        "Cloud9am": 5.0,
        "Cloud3pm": 4.0,
        "Temp9am": 20.0,
        "Temp3pm": 24.0,
        "RainToday": "No"
    }
    
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "will_rain_tomorrow" in data
    assert "probability" in data
    assert "timestamp" in data


def test_predict_invalid_input(client):
    """Test prediction with invalid input."""
    payload = {
        "Location": "Sydney",
        "MinTemp": 18.0,
        # Missing required fields
    }
    
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_model_info_endpoint(client):
    """Test model info endpoint."""
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "numerical_features" in data
    assert "categorical_features" in data

