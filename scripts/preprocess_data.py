"""
Clean and preprocess the raw Titanic dataset for model training and 
visualization.

This module loads the raw Titanic dataset, applies comprehensive preprocessing
including missing value imputation, outlier removal, feature engineering,
and scaling, then saves the processed data as a parquet file.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def setup_logging():
    """Configure logging for preprocessing operations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_raw_data(file_path):
    """
    Load and validate the raw Titanic dataset.
    
    Args:
        file_path (str): Path to the raw CSV file
        
    Returns:
        pd.DataFrame: Raw dataset
        
    Raises:
        FileNotFoundError: If the raw data file doesn't exist
        pd.errors.EmptyDataError: If the file is empty or corrupted
    """
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded raw dataset with shape: {df.shape}")
        
        # Validate expected columns
        expected_cols = ['PassengerId', 'Survived', 'Pclass', 'Name', 
                        'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 
                        'Fare', 'Cabin', 'Embarked']
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
            
        return df
        
    except FileNotFoundError:
        logging.error(f"Raw data file not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading raw data: {e}")
        raise


def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns (name, ticket, cabin, passengerid).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with unnecessary columns removed
    """
    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    
    if existing_cols:
        df = df.drop(columns=existing_cols)
        logging.info(f"Dropped columns: {existing_cols}")
    
    return df


def handle_missing_values(df):
    """
    Handle missing values with group-based imputation.
    
    - Drop rows with missing embarked values (only 2 rows)
    - Age: Group-based median imputation using pclass, sibsp_bin, parch_bin
    - Fare: Median imputation
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    logging.info("Handling missing values...")
    
    # Drop rows with missing embarked values
    initial_shape = df.shape[0]
    df = df.dropna(subset=['Embarked'])
    df = df.reset_index(drop=True)
    dropped_embarked = initial_shape - df.shape[0]
    logging.info(f"Dropped {dropped_embarked} rows with missing embarked")
    
    # Create auxiliary columns for age imputation
    df['sibsp_bin'] = df['SibSp'].apply(lambda x: '0' if x == 0 else '>0')
    df['parch_bin'] = df['Parch'].apply(
        lambda x: '0' if x == 0 else ('1_or_2' if x <= 2 else '3_plus')
    )
    
    # Group-based median imputation for age
    if df['Age'].isnull().sum() > 0:
        grouped_median = df.groupby(['Pclass', 'sibsp_bin', 'parch_bin'])
        df['Age'] = grouped_median['Age'].transform(
            lambda x: x.fillna(x.median())
        )
        logging.info("Applied group-based median imputation for age")
    
    # Median imputation for fare
    if df['Fare'].isnull().sum() > 0:
        fare_median = df['Fare'].median()
        df['Fare'] = df['Fare'].fillna(fare_median)
        logging.info("Applied median imputation for fare")
    
    # Drop auxiliary columns
    df = df.drop(columns=['sibsp_bin', 'parch_bin'])
    
    return df


def remove_outliers(df):
    """
    Remove outliers outside 3 standard deviations for age, fare, sibsp, parch.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed
    """
    logging.info("Removing outliers...")
    initial_shape = df.shape[0]
    
    columns_to_check = ['Age', 'Fare', 'SibSp', 'Parch']
    
    for col in columns_to_check:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            before_count = df.shape[0]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after_count = df.shape[0]
            
            removed = before_count - after_count
            if removed > 0:
                logging.info(f"Removed {removed} outliers from {col}")
    
    df = df.reset_index(drop=True)
    total_removed = initial_shape - df.shape[0]
    logging.info(f"Total outliers removed: {total_removed}")
    
    return df


def engineer_features(df):
    """
    Create engineered features and apply one-hot encoding.
    
    - Create 'child' feature (age ≤ 15)
    - One-hot encode sex, embarked, and pclass
    - Drop redundant sex_female column, keep only male
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    logging.info("Engineering features...")
    
    # Create child feature
    df['child'] = (df['Age'] <= 15).astype(int)
    
    # One-hot encode sex
    sex_dummies = pd.get_dummies(df['Sex'], prefix='sex', dtype=int)
    df = pd.concat([df, sex_dummies], axis=1)
    df = df.drop(columns=['Sex'])
    
    # Keep only male column, drop female
    if 'sex_female' in df.columns:
        df = df.drop(columns=['sex_female'])
    if 'sex_male' in df.columns:
        df = df.rename(columns={'sex_male': 'male'})
    
    # One-hot encode embarked
    embarked_dummies = pd.get_dummies(
        df['Embarked'], prefix='embark_town', dtype=int
    )
    df = pd.concat([df, embarked_dummies], axis=1)
    df = df.drop(columns=['Embarked'])
    
    # One-hot encode pclass
    pclass_dummies = pd.get_dummies(df['Pclass'], prefix='class', dtype=int)
    # Map numerical class to proper names
    pclass_mapping = {
        'class_1': 'class_First',
        'class_2': 'class_Second', 
        'class_3': 'class_Third'
    }
    pclass_dummies = pclass_dummies.rename(columns=pclass_mapping)
    df = pd.concat([df, pclass_dummies], axis=1)
    df = df.drop(columns=['Pclass'])
    
    logging.info("Feature engineering completed")
    return df


def scale_features(df):
    """
    Apply feature scaling.
    
    - Age: Custom scaling (0-122 range) -> age_scaled
    - Sibsp and Parch: MinMaxScaler -> sibsp_scaled, parch_scaled
    - Fare: StandardScaler -> fare_scaled
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with scaled features
    """
    logging.info("Scaling features...")
    
    # Custom age scaling (0-122 range)
    if 'Age' in df.columns:
        df['age_scaled'] = df['Age'] / 122.0
        df = df.drop(columns=['Age'])
    
    # MinMaxScaler for sibsp and parch
    if 'SibSp' in df.columns:
        scaler_sibsp = MinMaxScaler()
        df['sibsp_scaled'] = scaler_sibsp.fit_transform(
            df[['SibSp']]
        ).flatten()
        df = df.drop(columns=['SibSp'])
    
    if 'Parch' in df.columns:
        scaler_parch = MinMaxScaler()
        df['parch_scaled'] = scaler_parch.fit_transform(
            df[['Parch']]
        ).flatten()
        df = df.drop(columns=['Parch'])
    
    # StandardScaler for fare
    if 'Fare' in df.columns:
        scaler_fare = StandardScaler()
        df['fare_scaled'] = scaler_fare.fit_transform(
            df[['Fare']]
        ).flatten()
        df = df.drop(columns=['Fare'])
    
    logging.info("Feature scaling completed")
    return df


def clean_column_names(df):
    """
    Clean and standardize column names to match expected schema.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with cleaned column names
    """
    # Rename to match expected schema exactly as specified in PRD
    column_mapping = {
        'Survived': 'survived',  # Target variable should be lowercase
        'embark_town_C': 'embark_town_Cherbourg',
        'embark_town_Q': 'embark_town_Queenstown', 
        'embark_town_S': 'embark_town_Southampton',
        'class_First': 'class_First',
        'class_Second': 'class_Second',
        'class_Third': 'class_Third'
    }
    
    # Apply mapping if columns exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df


def save_processed_data(df, output_path):
    """
    Save processed dataset to parquet format.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path for output parquet file
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to parquet
        df.to_parquet(output_path, index=False)
        logging.info(f"Saved processed data to: {output_path}")
        logging.info(f"Final dataset shape: {df.shape}")
        logging.info(f"Columns: {list(df.columns)}")
        
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")
        raise


def main():
    """Main preprocessing pipeline."""
    setup_logging()
    
    try:
        # Set up paths
        project_root = Path(__file__).parent.parent
        raw_data_path = project_root / 'data' / 'titanic_raw.csv'
        processed_data_path = project_root / 'data' / 'titanic_processed.parquet'
        
        logging.info("Starting Titanic data preprocessing pipeline...")
        
        # Load raw data
        df = load_raw_data(raw_data_path)
        
        # Apply preprocessing steps
        df = drop_unnecessary_columns(df)
        df = handle_missing_values(df)
        df = remove_outliers(df)
        df = engineer_features(df)
        df = scale_features(df)
        df = clean_column_names(df)
        
        # Save processed data
        save_processed_data(df, processed_data_path)
        
        logging.info("Preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logging.error(f"Preprocessing pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()