# Career Shield: AI-Powered Job Fraud Detection

## Overview
An NLP-based fraud detection system that identifies fake job postings using fine-tuned DistilBERT. Achieves **98% accuracy** with **84% precision** on fake job detection.

## Features
- Fine-tuned DistilBERT classifier
- TF-IDF + Logistic Regression baseline (96.46% accuracy)
- Rule-based pattern matching for scam indicators
- Impossible requirements detection (validates experience vs. technology age)
- Interactive Streamlit dashboard with explainable results

## Tech Stack
- Python 3.8+, PyTorch, Hugging Face Transformers
- scikit-learn, pandas, Streamlit
- Git LFS (for large model files)

## Dataset
Kaggle "Fake Job Posting Prediction" - 17,880 labeled job postings (95.2% real, 4.8% fake)

## Setup Instructions

### Important: Git LFS Required
**The model file (`model.safetensors`, ~270MB) requires Git LFS.** Without it, the app won't work!

```powershell
# Install Git LFS first
winget install -e --id GitHub.GitLFS  # Windows
# brew install git-lfs                # macOS
# sudo apt-get install git-lfs        # Linux

git lfs install
```

### Installation

```powershell
# 1. Clone repository
git clone https://github.com/rishikathakre/career-shield.git
cd career-shield

# 2. If model.safetensors is only a few KB (not ~270MB), run:
git lfs pull

# 3. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the dashboard
streamlit run app.py
```

## Usage

1. Launch the Streamlit app: `streamlit run app.py`
2. Input a job description (paste text or enter URL)
3. View fraud detection results with explainable reasoning

### Retraining Models (Optional)

Pre-trained models are included. To retrain, run notebooks in `notebooks/` directory:
- `01_eda.ipynb` - Data exploration
- `02_preprocessing.ipynb` - Data cleaning
- `03_baseline_model.ipynb` - Baseline model
- `04_bert_model.ipynb` - DistilBERT fine-tuning

## Results

| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) |
|-------|----------|------------------|---------------|-----------------|
| Baseline (TF-IDF + LR) | 96.46% | 59.07% | 87.69% | 70.59% |
| **DistilBERT** | **98.0%** | **84.0%** | 60.0% | **70.0%** |

DistilBERT achieves +42% precision improvement with comparable F1-score.

## Project Structure

```
career-shield/
├── app.py              # Streamlit dashboard
├── models/             # DistilBERT model (requires Git LFS!)
├── data/               # Dataset and processed splits
├── src/                # Source code modules
├── notebooks/          # Jupyter notebooks
└── requirements.txt
```

## Contributors

Harsh Shrishrimal, Rishika Thakre - University of Maryland, College Park

