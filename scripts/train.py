"""Training script for weather prediction model."""
import logging
import sys
from pathlib import Path

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.weather_prediction.model import WeatherPredictor
from src.weather_prediction.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the raw data.
    
    Args:
        data_path: Path to the CSV data file
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    logger.info(f"Original data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Remove rows with missing target values
    logger.info("Removing rows with missing target values...")
    df = df.dropna(subset=["RainToday", "RainTomorrow"])
    
    logger.info(f"Data shape after removing missing targets: {df.shape}")
    logger.info(f"Target distribution:\n{df['RainTomorrow'].value_counts()}")
    
    return df


def train_model():
    """Main training function."""
    try:
        # Set paths
        data_path = Path(settings.DATA_DIR) / settings.TRAIN_DATA_FILE
        model_dir = Path(settings.MODEL_DIR)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load and preprocess data
        logger.info("=" * 50)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 50)
        
        raw_df = load_and_preprocess_data(data_path)
        
        # Split into train and validation sets
        logger.info("Splitting data into train and validation sets...")
        train_df, val_df = train_test_split(
            raw_df,
            test_size=0.2,
            random_state=42,
            stratify=raw_df['RainTomorrow']
        )
        
        logger.info(f"Training set size: {len(train_df)}")
        logger.info(f"Validation set size: {len(val_df)}")
        
        # Initialize and train the model
        logger.info("Initializing model...")
        predictor = WeatherPredictor()
        
        logger.info("Training model...")
        predictor.fit(train_df, validation_data=val_df)
        
        # Save the model
        logger.info(f"Saving model to {model_dir}...")
        predictor.save(str(model_dir))
        
        # Print final metrics
        logger.info("=" * 50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info(f"\nTraining Metrics:")
        for metric, value in predictor.metrics.get('train', {}).items():
            logger.info(f"  {metric}: {value:.4f}")
        
        if 'validation' in predictor.metrics:
            logger.info(f"\nValidation Metrics:")
            for metric, value in predictor.metrics.get('validation', {}).items():
                logger.info(f"  {metric}: {value:.4f}")
        
        logger.info(f"\nModel saved to: {model_dir}")
        
        return predictor
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    train_model()
