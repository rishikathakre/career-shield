"""
Utility functions for fake job posting detection project.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, special characters, and normalizing whitespace.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove special characters but keep alphanumeric and basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def combine_text_features(df: pd.DataFrame) -> pd.Series:
    """
    Combine multiple text columns into a single text feature.
    
    Args:
        df: DataFrame with columns like 'title', 'description', 'company_profile', etc.
        
    Returns:
        Series of combined text
    """
    text_columns = ['title', 'description', 'company_profile', 'requirements', 'benefits']
    combined_text = df[text_columns].fillna('').apply(
        lambda row: ' '.join(row.astype(str)), axis=1
    )
    return combined_text


def calculate_text_statistics(text: str) -> Dict[str, float]:
    """
    Calculate basic text statistics.
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0
        }
    
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length
    }


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filepath: Path to save CSV file
    """
    try:
        # Create directory if it doesn't exist
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} rows to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def check_data_quality(df: pd.DataFrame, target_col: str = 'fraudulent') -> Dict:
    """
    Check data quality and return summary statistics.
    
    Args:
        df: DataFrame to check
        target_col: Name of target column
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
    }
    
    if target_col in df.columns:
        quality_report['class_distribution'] = df[target_col].value_counts().to_dict()
        quality_report['class_balance'] = df[target_col].value_counts(normalize=True).to_dict()
    
    return quality_report

