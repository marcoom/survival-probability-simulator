"""
Train the Random Forest model using the processed dataset and save it with 
preprocessing pipeline.

This module loads processed Titanic data, creates a complete sklearn Pipeline
with preprocessing and Random Forest classifier, trains the model with
specified hyperparameters, evaluates performance, and saves the pipeline
for production use.
"""

import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def setup_logging():
    """Configure logging for model training operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_processed_data(file_path):
    """
    Load and validate the processed Titanic dataset.
    
    Args:
        file_path (str): Path to the processed parquet file
        
    Returns:
        tuple: (X, y) features and target arrays
        
    Raises:
        FileNotFoundError: If the processed data file doesn't exist
        ValueError: If required columns are missing
    """
    try:
        df = pd.read_parquet(file_path)
        logging.info(f"Loaded processed dataset with shape: {df.shape}")
        
        # Validate target column exists
        if 'survived' not in df.columns:
            raise ValueError("Target column 'survived' not found in dataset")
        
        # Separate features and target
        X = df.drop('survived', axis=1)
        y = df['survived']
        
        logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        logging.info(f"Feature columns: {list(X.columns)}")
        
        return X, y
        
    except FileNotFoundError:
        logging.error(f"Processed data file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading processed data: {e}")
        raise


class FeaturePassthrough(BaseEstimator, TransformerMixin):
    """
    Simple passthrough transformer for already processed features.
    
    Since the data is already preprocessed, this transformer simply
    passes through the features while maintaining pipeline compatibility.
    """
    
    def fit(self, X, y=None):
        """Fit the transformer (no-op for passthrough)."""
        logging.info("Fitting passthrough transformer")
        return self
    
    def transform(self, X):
        """Transform the features (passthrough)."""
        return X


def create_model_pipeline():
    """
    Create sklearn Pipeline with preprocessing and Random Forest classifier.
    
    Returns:
        Pipeline: Complete sklearn pipeline ready for training
    """
    logging.info("Creating model pipeline...")
    
    # Random Forest with exact hyperparameters from PRD
    rf_classifier = RandomForestClassifier(
        n_estimators=1000,
        max_depth=9,
        min_samples_leaf=1,
        min_samples_split=2,
        max_features=0.2,
        max_samples=0.2,
        bootstrap=True,
        random_state=42
    )
    
    # Create pipeline with preprocessing and classifier
    pipeline = Pipeline([
        ('preprocessor', FeaturePassthrough()),
        ('classifier', rf_classifier)
    ])
    
    logging.info("Model pipeline created successfully")
    return pipeline


def train_and_evaluate_model(pipeline, X, y):
    """
    Train the model pipeline and evaluate performance.
    
    Args:
        pipeline (Pipeline): sklearn pipeline to train
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        
    Returns:
        Pipeline: Trained pipeline
    """
    logging.info("Training model...")
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    logging.info("Model training completed")
    
    # Evaluate performance
    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    logging.info(f"Training accuracy: {train_accuracy:.4f}")
    logging.info(f"Test accuracy: {test_accuracy:.4f}")
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_predictions))
    
    # Feature importance
    rf_model = pipeline.named_steps['classifier']
    feature_importances = rf_model.feature_importances_
    feature_names = X.columns
    
    print("\nFeature Importances:")
    print("-" * 30)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:25}: {row['importance']:.4f}")
    
    return pipeline


def save_model_pipeline(pipeline, output_path):
    """
    Save trained pipeline to disk using joblib.
    
    Args:
        pipeline (Pipeline): Trained sklearn pipeline
        output_path (str): Path for output joblib file
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save pipeline
        joblib.dump(pipeline, output_path)
        logging.info(f"Model pipeline saved to: {output_path}")
        
        # Verify the saved model can be loaded
        loaded_pipeline = joblib.load(output_path)
        logging.info("Model pipeline verification successful")
        
        print(f"\nModel pipeline successfully saved to: {output_path}")
        
    except Exception as e:
        logging.error(f"Error saving model pipeline: {e}")
        raise


def main():
    """Main model training pipeline."""
    setup_logging()
    
    try:
        # Set up paths
        project_root = Path(__file__).parent.parent
        processed_data_path = project_root / 'data' / 'titanic_processed.parquet'
        model_output_path = project_root / 'model' / 'titanic_model_pipeline.joblib'
        
        logging.info("Starting Titanic model training pipeline...")
        
        # Load processed data
        X, y = load_processed_data(processed_data_path)
        
        # Create model pipeline
        pipeline = create_model_pipeline()
        
        # Train and evaluate model
        trained_pipeline = train_and_evaluate_model(pipeline, X, y)
        
        # Save trained pipeline
        save_model_pipeline(trained_pipeline, model_output_path)
        
        logging.info("Model training pipeline completed successfully!")
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        logging.error(f"Model training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()