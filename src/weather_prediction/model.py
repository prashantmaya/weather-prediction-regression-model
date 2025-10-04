"""Weather prediction model with preprocessing pipeline."""
import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

logger = logging.getLogger(__name__)


class WeatherPredictor:
    """Weather prediction model with preprocessing pipeline."""
    
    def __init__(self):
        """Initialize the weather predictor."""
        self.numerical_cols = [
            'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
            'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
            'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
            'Cloud3pm', 'Temp9am', 'Temp3pm'
        ]
        self.categorical_cols = [
            'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'
        ]
        self.target_col = 'RainTomorrow'
        
        # Initialize preprocessing components
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.model = LogisticRegression(
            solver="liblinear",
            tol=0.00001,
            max_iter=1000,
            random_state=42
        )
        self.is_fitted = False
        self.metrics = {}
        
    def _validate_data(self, data: pd.DataFrame, is_training: bool = False):
        """Validate input data structure."""
        required_cols = self.numerical_cols + self.categorical_cols
        if is_training:
            required_cols.append(self.target_col)
        
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.debug(f"Data validation passed. Shape: {data.shape}")
        
    def fit(self, train_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None):
        """
        Fit the model and all preprocessing components.
        
        Args:
            train_data: Training dataset
            validation_data: Optional validation dataset for metrics
        
        Returns:
            self
        """
        try:
            logger.info("Starting model training...")
            self._validate_data(train_data, is_training=True)
            
            # Prepare target
            y_train = train_data[self.target_col]
            logger.info(f"Training samples: {len(y_train)}, Positive class: {(y_train == 'Yes').sum()}")
            
            # Fit imputer and transform numerical columns
            logger.debug("Fitting imputer on numerical features...")
            self.imputer.fit(train_data[self.numerical_cols])
            X_numerical = self.imputer.transform(train_data[self.numerical_cols])
            
            # Fit scaler and transform numerical data
            logger.debug("Fitting scaler on numerical features...")
            self.scaler.fit(X_numerical)
            X_numerical_scaled = self.scaler.transform(X_numerical)
            
            # Fit encoder and transform categorical data
            logger.debug("Fitting encoder on categorical features...")
            X_categorical = train_data[self.categorical_cols].fillna("Unknown")
            self.encoder.fit(X_categorical)
            X_categorical_encoded = self.encoder.transform(X_categorical)
            
            # Combine features
            X_train = np.hstack([X_numerical_scaled, X_categorical_encoded])
            logger.info(f"Final feature shape: {X_train.shape}")
            
            # Fit the model
            logger.info("Training logistic regression model...")
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Calculate training metrics
            train_predictions = self.model.predict(X_train)
            train_proba = self.model.predict_proba(X_train)[:, 1]
            
            self.metrics['train'] = {
                'accuracy': accuracy_score(y_train, train_predictions),
                'precision': precision_score(y_train, train_predictions, pos_label='Yes'),
                'recall': recall_score(y_train, train_predictions, pos_label='Yes'),
                'f1': f1_score(y_train, train_predictions, pos_label='Yes'),
                'roc_auc': roc_auc_score((y_train == 'Yes').astype(int), train_proba)
            }
            
            logger.info(f"Training metrics: {self.metrics['train']}")
            
            # Calculate validation metrics if provided
            if validation_data is not None:
                logger.info("Calculating validation metrics...")
                y_val = validation_data[self.target_col]
                X_val = self._preprocess(validation_data)
                val_predictions = self.model.predict(X_val)
                val_proba = self.model.predict_proba(X_val)[:, 1]
                
                self.metrics['validation'] = {
                    'accuracy': accuracy_score(y_val, val_predictions),
                    'precision': precision_score(y_val, val_predictions, pos_label='Yes'),
                    'recall': recall_score(y_val, val_predictions, pos_label='Yes'),
                    'f1': f1_score(y_val, val_predictions, pos_label='Yes'),
                    'roc_auc': roc_auc_score((y_val == 'Yes').astype(int), val_proba)
                }
                
                logger.info(f"Validation metrics: {self.metrics['validation']}")
            
            logger.info("Model training completed successfully!")
            return self
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
            raise
    
    def _preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess input data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Handle numerical features
        X_numerical = self.imputer.transform(data[self.numerical_cols])
        X_numerical_scaled = self.scaler.transform(X_numerical)
        
        # Handle categorical features
        X_categorical = data[self.categorical_cols].fillna("Unknown")
        X_categorical_encoded = self.encoder.transform(X_categorical)
        
        # Combine features
        X = np.hstack([X_numerical_scaled, X_categorical_encoded])
        return X
    
    def predict(self, input_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Make a prediction for input data.
        
        Args:
            input_data: DataFrame with weather features
        
        Returns:
            Tuple of (prediction, probability)
        """
        try:
            self._validate_data(input_data, is_training=False)
            X = self._preprocess(input_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][list(self.model.classes_).index(prediction)]
            
            return prediction, float(probability)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            raise
    
    def save(self, model_dir: str = "model"):
        """
        Save the model and all preprocessing components.
        
        Args:
            model_dir: Directory to save model files
        """
        try:
            model_path = Path(model_dir)
            model_path.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Saving model to {model_dir}...")
            
            # Save all components
            joblib.dump(self.imputer, model_path / "imputer.joblib")
            joblib.dump(self.scaler, model_path / "scaler.joblib")
            joblib.dump(self.encoder, model_path / "encoder.joblib")
            joblib.dump(self.model, model_path / "model.joblib")
            
            # Save metadata
            metadata = {
                'numerical_cols': self.numerical_cols,
                'categorical_cols': self.categorical_cols,
                'target_col': self.target_col,
                'is_fitted': self.is_fitted,
                'metrics': self.metrics
            }
            joblib.dump(metadata, model_path / "metadata.joblib")
            
            logger.info(f"Model saved successfully to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise
        
    @classmethod
    def load(cls, model_dir: str = "model") -> "WeatherPredictor":
        """
        Load a saved model and all preprocessing components.
        
        Args:
            model_dir: Directory containing model files
        
        Returns:
            Loaded WeatherPredictor instance
        """
        try:
            model_path = Path(model_dir)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            logger.info(f"Loading model from {model_dir}...")
            
            predictor = cls()
            
            # Load all components
            predictor.imputer = joblib.load(model_path / "imputer.joblib")
            predictor.scaler = joblib.load(model_path / "scaler.joblib")
            predictor.encoder = joblib.load(model_path / "encoder.joblib")
            predictor.model = joblib.load(model_path / "model.joblib")
            
            # Load metadata if available
            metadata_file = model_path / "metadata.joblib"
            if metadata_file.exists():
                metadata = joblib.load(metadata_file)
                predictor.is_fitted = metadata.get('is_fitted', True)
                predictor.metrics = metadata.get('metrics', {})
            else:
                predictor.is_fitted = True
            
            logger.info(f"Model loaded successfully from {model_dir}")
            logger.info(f"Model metrics: {predictor.metrics}")
            
            return predictor
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise
