"""Weather Prediction System - A production-ready ML application."""

__version__ = "1.0.0"
__author__ = "Prashant Soni"

from .model import WeatherPredictor
from .config import settings

__all__ = ["WeatherPredictor", "settings"]
