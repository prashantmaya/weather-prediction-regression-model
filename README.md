# 🌦️ Weather Prediction System

[![CI/CD Pipeline](https://github.com/YOUR_USERNAME/weather-prediction-system/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/weather-prediction-system/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready weather prediction system that uses logistic regression to predict whether it will rain tomorrow based on today's weather conditions. Built with FastAPI, scikit-learn, and Docker.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Production-Ready API**: FastAPI-based REST API with automatic documentation
- **Machine Learning**: Logistic regression model with preprocessing pipeline
- **Model Persistence**: Save and load trained models
- **Validation**: Input validation with Pydantic
- **Logging**: Comprehensive logging system
- **Testing**: Unit tests with pytest
- **Docker Support**: Containerized application
- **CI/CD**: GitHub Actions pipeline
- **Health Checks**: Monitoring and health check endpoints
- **Configuration Management**: Environment-based configuration
- **Error Handling**: Robust error handling and validation

## 🛠️ Tech Stack

- **Python 3.10+**
- **FastAPI** - Modern web framework
- **scikit-learn** - Machine learning
- **Pandas** - Data manipulation
- **Pydantic** - Data validation
- **Docker** - Containerization
- **pytest** - Testing
- **GitHub Actions** - CI/CD

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/weather-prediction-system.git
cd weather-prediction-system

# Build and run with Docker Compose
docker-compose up -d

# Access the API at http://localhost:8000
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (if not already trained)
python train.py

# Run the API server
uvicorn app:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- pip
- virtualenv (recommended)
- Docker (for containerized deployment)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/weather-prediction-system.git
cd weather-prediction-system

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Copy environment variables template
cp .env.example .env

# Train the model
python train.py
```

## 💻 Usage

### Starting the API Server

```bash
# Development mode with auto-reload
uvicorn app:app --reload

# Production mode
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Making Predictions

#### Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
         }'
```

#### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/predict"
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

response = requests.post(url, json=payload)
print(response.json())
```

## 📚 API Documentation

### Endpoints

| Method | Endpoint               | Description              |
| ------ | ---------------------- | ------------------------ |
| GET    | `/`                  | Root endpoint            |
| GET    | `/health`            | Health check             |
| POST   | `/api/v1/predict`    | Make prediction          |
| GET    | `/api/v1/model/info` | Get model information    |
| GET    | `/docs`              | Swagger UI documentation |
| GET    | `/redoc`             | ReDoc documentation      |

### Response Format

```json
{
  "will_rain_tomorrow": true,
  "probability": 0.85,
  "timestamp": "2025-10-04T12:00:00Z"
}
```

## 🎓 Model Training

### Training the Model

```bash
# Basic training
python train.py

# Using make
make train
```

### Model Metrics

The trained model provides the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Metrics are displayed during training and saved with the model.

### Custom Training

```python
from weather_predictor import WeatherPredictor
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")

# Train model
predictor = WeatherPredictor()
predictor.fit(data)

# Save model
predictor.save("custom_model_dir")
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_weather_predictor.py

# Using make
make test
```

### Test Coverage

The project aims for >80% test coverage. View the coverage report:

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

## 🐳 Docker Deployment

### Building the Image

```bash
# Using Docker Compose
docker-compose build

# Using Docker directly
docker build -t weather-prediction-api:latest .

# Using make
make docker-build
```

### Running the Container

```bash
# Using Docker Compose
docker-compose up -d

# Using Docker directly
docker run -p 8000:8000 weather-prediction-api:latest

# Using make
make docker-run
```

### Stopping the Container

```bash
docker-compose down
# or
make docker-stop
```

## 🔧 Configuration

Configuration is managed through environment variables. Copy `.env.example` to `.env` and customize:

```bash
# Application
APP_NAME=Weather Prediction API
DEBUG=False
LOG_LEVEL=INFO

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model
MODEL_DIR=model
DATA_DIR=data
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format code (`make format`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Use isort for import sorting
- Add type hints where applicable
- Write docstrings for functions and classes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Weather data from [Australian Government Bureau of Meteorology](http://www.bom.gov.au/)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- ML powered by [scikit-learn](https://scikit-learn.org/)

## 📞 Contact

Your Name - [@yourhandle](https://twitter.com/yourhandle)

Project Link: [https://github.com/YOUR_USERNAME/weather-prediction-system](https://github.com/YOUR_USERNAME/weather-prediction-system)

## 📈 Roadmap

- [ ] Add more ML models (Random Forest, XGBoost)
- [ ] Implement model versioning
- [ ] Add monitoring and observability (Prometheus, Grafana)
- [ ] Create web UI for predictions
- [ ] Add batch prediction endpoint
- [ ] Implement model retraining pipeline
- [ ] Add feature importance visualization
- [ ] Deploy to cloud (AWS/GCP/Azure)

---

Made with ❤️ by Prashant Maya (Prashant Soni)
