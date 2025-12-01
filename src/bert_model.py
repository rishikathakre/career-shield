"""
Fine-tuned BERT/DistilBERT model for fake job posting detection.
"""

import pandas as pd
import numpy as np
import torch

# Ensure accelerate is available before importing transformers
try:
    import accelerate
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "accelerate>=0.26.0", "-q"])
    import accelerate

from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datasets import Dataset
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BERTModel:
    """Fine-tuned BERT/DistilBERT model for classification."""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', num_labels: int = 2):
        """
        Initialize BERT model.
        
        Args:
            model_name: Hugging Face model name (e.g., 'distilbert-base-uncased', 'bert-base-uncased')
            num_labels: Number of classification labels
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.trainer = None
        self.is_trained = False
        
    def tokenize_data(self, texts: pd.Series, max_length: int = 512) -> dict:
        """
        Tokenize text data.
        
        Args:
            texts: Series of text data
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with tokenized data
        """
        return self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
    
    def prepare_dataset(self, texts: pd.Series, labels: pd.Series = None) -> Dataset:
        """
        Prepare dataset for training/evaluation.
        
        Args:
            texts: Series of text data
            labels: Series of labels (optional for inference)
            
        Returns:
            Hugging Face Dataset
        """
        encodings = self.tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors=None
        )
        
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask']
        }
        if labels is not None:
            dataset_dict['labels'] = labels.tolist()
        
        return Dataset.from_dict(dataset_dict)
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Evaluation predictions tuple (predictions, labels)
            
        Returns:
            Dictionary with metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, average='binary'),
            'recall': recall_score(labels, predictions, average='binary'),
            'f1': f1_score(labels, predictions, average='binary')
        }
    
    def train(self, train_texts: pd.Series, train_labels: pd.Series,
              val_texts: pd.Series = None, val_labels: pd.Series = None,
              output_dir: str = 'data/models/bert',
              num_epochs: int = 3,
              batch_size: int = 16,
              learning_rate: float = 2e-5):
        """
        Train the BERT model.
        
        Args:
            train_texts: Training text data
            train_labels: Training labels
            val_texts: Validation text data (optional)
            val_labels: Validation labels (optional)
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        logger.info("Preparing training dataset...")
        train_dataset = self.prepare_dataset(train_texts, train_labels)
        
        # Prepare validation dataset if provided
        eval_dataset = None
        if val_texts is not None and val_labels is not None:
            logger.info("Preparing validation dataset...")
            eval_dataset = self.prepare_dataset(val_texts, val_labels)
        
        # Ensure accelerate is imported before creating TrainingArguments
        try:
            import accelerate
            logger.info(f"accelerate version: {accelerate.__version__}")
        except ImportError:
            logger.error("accelerate not found. Please install: pip install accelerate>=0.26.0")
            raise
        
        # Check if GPU is available
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU - training will be slower. Consider using Google Colab with free GPU.")
            logger.info("Optimizing for CPU: reducing batch size and using fewer workers.")
        
        # Optimize batch size for CPU
        effective_batch_size = batch_size if use_gpu else min(batch_size, 4)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=effective_batch_size,
            per_device_eval_batch_size=effective_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=50,  # More frequent logging
            eval_strategy='epoch' if eval_dataset else 'no',
            save_strategy='epoch',
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model='f1' if eval_dataset else None,
            greater_is_better=True,
            fp16=use_gpu,  # Use mixed precision on GPU for faster training
            dataloader_num_workers=0,  # 0 workers for CPU to avoid overhead
            dataloader_pin_memory=False,  # Disable pin memory on CPU
        )
        
        # Initialize trainer
        callbacks = []
        if eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        logger.info("Starting training...")
        self.trainer.train()
        self.is_trained = True
        logger.info("Training complete!")
        
        # Save model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
    
    def predict(self, texts: pd.Series) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            texts: Text data to predict
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        dataset = self.prepare_dataset(texts)
        predictions = self.trainer.predict(dataset)
        return np.argmax(predictions.predictions, axis=1)
    
    def predict_proba(self, texts: pd.Series) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: Text data to predict
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        dataset = self.prepare_dataset(texts)
        predictions = self.trainer.predict(dataset)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)
        return probs.numpy()
    
    def evaluate(self, texts: pd.Series, labels: pd.Series) -> dict:
        """
        Evaluate model performance.
        
        Args:
            texts: Test text data
            labels: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        dataset = self.prepare_dataset(texts, labels)
        predictions = self.trainer.predict(dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = labels.values
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
        
        return metrics
    
    def load(self, model_dir: str = 'data/models/bert'):
        """
        Load trained model.
        
        Args:
            model_dir: Directory containing saved model
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.is_trained = True
        logger.info(f"Model loaded from {model_dir}")


def main():
    """Main function to train BERT model."""
    # Load processed data
    # train_df = pd.read_csv('data/processed/train.csv')
    # val_df = pd.read_csv('data/processed/val.csv')
    # test_df = pd.read_csv('data/processed/test.csv')
    # 
    # # Initialize model (use DistilBERT for faster training)
    # model = BERTModel(model_name='distilbert-base-uncased')
    # 
    # # Train model
    # model.train(
    #     train_texts=train_df['combined_text'],
    #     train_labels=train_df['fraudulent'],
    #     val_texts=val_df['combined_text'],
    #     val_labels=val_df['fraudulent'],
    #     num_epochs=3,
    #     batch_size=16
    # )
    # 
    # # Evaluate on test set
    # logger.info("Evaluating on test set...")
    # test_metrics = model.evaluate(test_df['combined_text'], test_df['fraudulent'])
    # 
    # logger.info("BERT model training complete!")
    
    logger.info("BERT model ready. Update data paths and uncomment main() code to run.")


if __name__ == "__main__":
    main()

