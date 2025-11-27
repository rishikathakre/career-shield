"""
Baseline model: TF-IDF + Logistic Regression for fake job posting detection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel:
    """Baseline model using TF-IDF and Logistic Regression."""
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize baseline model.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            ngram_range: Range of n-grams to use
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.95
        )
        self.model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        self.is_trained = False
        
    def train(self, X_train: pd.Series, y_train: pd.Series):
        """
        Train the baseline model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
        """
        logger.info("Vectorizing training data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        logger.info(f"Feature matrix shape: {X_train_vec.shape}")
        
        logger.info("Training Logistic Regression model...")
        self.model.fit(X_train_vec, y_train)
        self.is_trained = True
        logger.info("Training complete!")
        
    def predict(self, X: pd.Series) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Text data to predict
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Text data to predict
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_vec = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vec)
    
    def evaluate(self, X: pd.Series, y: pd.Series) -> dict:
        """
        Evaluate model performance.
        
        Args:
            X: Test text data
            y: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1_score': f1_score(y, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
        
        return metrics
    
    def save(self, model_dir: str = 'data/models'):
        """
        Save model and vectorizer.
        
        Args:
            model_dir: Directory to save model files
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Cannot save untrained model.")
        
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, f'{model_dir}/baseline_model.pkl')
        joblib.dump(self.vectorizer, f'{model_dir}/baseline_vectorizer.pkl')
        logger.info(f"Model saved to {model_dir}")
    
    def load(self, model_dir: str = 'data/models'):
        """
        Load model and vectorizer.
        
        Args:
            model_dir: Directory containing model files
        """
        self.model = joblib.load(f'{model_dir}/baseline_model.pkl')
        self.vectorizer = joblib.load(f'{model_dir}/baseline_vectorizer.pkl')
        self.is_trained = True
        logger.info(f"Model loaded from {model_dir}")


def main():
    """Main function to train baseline model."""
    # Load processed data
    # train_df = pd.read_csv('data/processed/train.csv')
    # val_df = pd.read_csv('data/processed/val.csv')
    # test_df = pd.read_csv('data/processed/test.csv')
    # 
    # # Initialize model
    # model = BaselineModel(max_features=5000, ngram_range=(1, 2))
    # 
    # # Train model
    # model.train(train_df['combined_text'], train_df['fraudulent'])
    # 
    # # Evaluate on validation set
    # logger.info("Evaluating on validation set...")
    # val_metrics = model.evaluate(val_df['combined_text'], val_df['fraudulent'])
    # 
    # # Evaluate on test set
    # logger.info("Evaluating on test set...")
    # test_metrics = model.evaluate(test_df['combined_text'], test_df['fraudulent'])
    # 
    # # Save model
    # model.save()
    # 
    # logger.info("Baseline model training complete!")
    
    logger.info("Baseline model ready. Update data paths and uncomment main() code to run.")


if __name__ == "__main__":
    main()

