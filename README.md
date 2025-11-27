# Fake Job Posting Detection Using NLP

## Project Overview
An NLP-based classifier that distinguishes real job postings from fake ones using semantic and sentiment cues. This project helps job seekers and recruitment platforms prevent fraud and improve trust.

## Features
- **Baseline Model**: TF-IDF + Logistic Regression for comparison
- **Advanced Model**: Fine-tuned BERT/DistilBERT classifier
- **Bonus Feature**: Impossible jobs detection (e.g., requiring 10 years of experience in technologies that are only 8 years old)
- **Interactive Dashboard**: Streamlit app for real-time predictions

## Tech Stack
- Python 3.8+
- TensorFlow / PyTorch
- Hugging Face Transformers (BERT/DistilBERT)
- Streamlit
- scikit-learn

## Dataset
Kaggle "Fake Job Posting Prediction" dataset (18K+ labeled job listings)

## Project Structure
```
fake-job-detection/
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Cleaned and preprocessed data
│   └── models/           # Saved models
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
├── dashboard/            # Streamlit dashboard
├── tests/                # Unit tests
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the repository
```powershell
git clone <repository-url>
cd "fake-job-detection"
```

### 2. Create virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download dataset
- Download from Kaggle: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
- Place the CSV file in `data/raw/` directory
- Or use Kaggle API:
```powershell
kaggle datasets download -d shivamb/real-or-fake-fake-jobposting-prediction -p data/raw/
```

### 5. Run the dashboard
```powershell
streamlit run dashboard/app.py
```

## Usage

### Training Models
```powershell
# Train baseline model
python src/baseline_model.py

# Train BERT model
python src/bert_model.py
```

### Using the Dashboard
1. Start the Streamlit app: `streamlit run dashboard/app.py`
2. Enter a job posting text in the input field
3. View predictions from both baseline and BERT models
4. Check for impossible job requirements

## Evaluation Metrics
- Accuracy: 93%+ (target)
- Precision, Recall, F1-score
- Confusion Matrix

## Project Timeline
1.5 months sprint including:
- Data exploration and preprocessing
- Model development and fine-tuning
- Evaluation and dashboard deployment

## Contributors
[Your Team Members]

## License
[Your License]

