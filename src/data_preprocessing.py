"""
Data preprocessing pipeline for fake job posting detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

from utils import clean_text, combine_text_features, calculate_text_statistics, load_data, save_data, check_data_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Class for preprocessing job posting data."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize preprocessor.
        
        Args:
            data_path: Path to raw data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        
    def load_raw_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Load raw data from CSV.
        
        Args:
            filepath: Path to CSV file (uses self.data_path if not provided)
            
        Returns:
            DataFrame with raw data
        """
        path = filepath or self.data_path
        if not path:
            raise ValueError("No data path provided")
            
        self.df = load_data(path)
        logger.info(f"Loaded {len(self.df)} rows")
        return self.df
    
    def explore_data(self) -> dict:
        """
        Perform exploratory data analysis.
        
        Returns:
            Dictionary with EDA results
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        logger.info("Performing exploratory data analysis...")
        
        # Data quality check
        quality_report = check_data_quality(self.df, target_col='fraudulent')
        
        logger.info(f"Total rows: {quality_report['total_rows']}")
        logger.info(f"Total columns: {quality_report['total_columns']}")
        logger.info(f"Class distribution: {quality_report.get('class_distribution', {})}")
        logger.info(f"Missing values per column: {quality_report['missing_values']}")
        
        return quality_report
    
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess the data: clean text, handle missing values, create features.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        logger.info("Starting preprocessing...")
        df = self.df.copy()
        
        # Clean text columns
        text_columns = ['title', 'description', 'company_profile', 'requirements', 'benefits']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)
                logger.info(f"Cleaned {col}")
        
        # Combine text features
        df['combined_text'] = combine_text_features(df)
        
        # Calculate text statistics
        text_stats = df['combined_text'].apply(calculate_text_statistics)
        for stat_name in ['char_count', 'word_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length']:
            df[f'text_{stat_name}'] = [stats[stat_name] for stats in text_stats]
        
        # Handle missing values in other columns
        df['location'] = df['location'].fillna('Unknown')
        df['department'] = df['department'].fillna('Unknown')
        df['salary_range'] = df['salary_range'].fillna('Not specified')
        
        # Create binary features
        df['has_company_profile'] = df['company_profile'].notna() & (df['company_profile'] != '')
        df['has_requirements'] = df['requirements'].notna() & (df['requirements'] != '')
        df['has_benefits'] = df['benefits'].notna() & (df['benefits'] != '')
        
        # Ensure target column exists and is binary
        if 'fraudulent' in df.columns:
            df['fraudulent'] = df['fraudulent'].astype(int)
        
        self.processed_df = df
        logger.info("Preprocessing complete!")
        
        return df
    
    def split_data(self, test_size: float = 0.15, val_size: float = 0.15, 
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of data for validation set (from remaining after test)
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.processed_df is None:
            raise ValueError("No processed data. Call preprocess() first.")
        
        df = self.processed_df.copy()
        
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['fraudulent'] if 'fraudulent' in df.columns else None
        )
        
        # Second split: train vs val
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),  # Adjust val_size relative to train+val
            random_state=random_state,
            stratify=train_val_df['fraudulent'] if 'fraudulent' in train_val_df.columns else None
        )
        
        logger.info(f"Train set: {len(train_df)} rows")
        logger.info(f"Validation set: {len(val_df)} rows")
        logger.info(f"Test set: {len(test_df)} rows")
        
        return train_df, val_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                           test_df: pd.DataFrame, base_path: str = 'data/processed'):
        """
        Save processed data splits to CSV files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            base_path: Base directory path for saving
        """
        save_data(train_df, f'{base_path}/train.csv')
        save_data(val_df, f'{base_path}/val.csv')
        save_data(test_df, f'{base_path}/test.csv')
        logger.info("Processed data saved successfully!")


def main():
    """Main function to run preprocessing pipeline."""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data (update path as needed)
    # preprocessor.load_raw_data('data/raw/fake_job_postings.csv')
    # 
    # # Explore data
    # preprocessor.explore_data()
    # 
    # # Preprocess
    # preprocessor.preprocess()
    # 
    # # Split data
    # train_df, val_df, test_df = preprocessor.split_data()
    # 
    # # Save processed data
    # preprocessor.save_processed_data(train_df, val_df, test_df)
    
    logger.info("Preprocessing pipeline ready. Update data path and uncomment main() code to run.")


if __name__ == "__main__":
    main()

