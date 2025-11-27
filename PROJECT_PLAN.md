# Fake Job Posting Detection Using NLP - Project Plan

## Project Overview
Develop an NLP-based classifier to distinguish real job postings from fake ones using semantic and sentiment cues.

## Tech Stack
- Python 3.8+
- TensorFlow / PyTorch
- Hugging Face Transformers (BERT/DistilBERT)
- Streamlit
- scikit-learn (for baseline models)
- pandas, numpy (data processing)

## Dataset
- **Source**: Kaggle "Fake Job Posting Prediction" dataset
- **Size**: 18K+ labeled job listings
- **Features**: title, description, company details, location, etc.
- **Target**: Binary classification (real vs fake)

## Project Phases

### Phase 1: Project Setup & Data Acquisition (Week 1, Days 1-3)
- [ ] Set up Python environment (requirements.txt)
- [ ] Download/prepare Kaggle dataset
- [ ] Initial data exploration (EDA)
- [ ] Data quality assessment (missing values, duplicates, class distribution)
- [ ] Create project structure

### Phase 2: Data Preprocessing & Feature Engineering (Week 1, Days 4-7)
- [ ] Text cleaning (remove HTML, special characters, normalize whitespace)
- [ ] Handle missing values
- [ ] Feature extraction:
  - Text length statistics
  - Keyword analysis (suspicious words/phrases)
  - Company information validation
  - Location validation
- [ ] Train/validation/test split (70/15/15 or 80/10/10)
- [ ] Text preprocessing for both models (tokenization, etc.)

### Phase 3: Baseline Model Development (Week 2, Days 1-4)
- [ ] Implement TF-IDF vectorization
- [ ] Train Logistic Regression classifier
- [ ] Hyperparameter tuning (grid search/random search)
- [ ] Evaluate baseline model:
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion matrix
- [ ] Save baseline model

### Phase 4: Advanced Model Development - BERT/DistilBERT (Week 2-3, Days 5-14)
- [ ] Choose model (DistilBERT for speed, BERT for accuracy)
- [ ] Set up Hugging Face Transformers pipeline
- [ ] Fine-tuning strategy:
  - Learning rate scheduling
  - Batch size optimization
  - Epoch selection
- [ ] Fine-tune on training set
- [ ] Validation monitoring (early stopping)
- [ ] Model evaluation:
  - Accuracy, Precision, Recall, F1-score
  - Comparison with baseline
- [ ] Save fine-tuned model

### Phase 5: Advanced Feature - Impossible Jobs Detection (Week 3, Days 10-14)
- [ ] Research technology timelines (e.g., transformers released 2017)
- [ ] Extract experience requirements from job descriptions
- [ ] Match technologies with their release dates
- [ ] Flag impossible combinations (e.g., "10 years LLM experience" when LLMs are 8 years old)
- [ ] Integrate as additional feature or separate classifier
- [ ] Evaluate impact on detection accuracy

### Phase 6: Model Evaluation & Comparison (Week 3-4, Days 15-21)
- [ ] Comprehensive evaluation on test set
- [ ] Error analysis (false positives/negatives)
- [ ] Feature importance analysis
- [ ] Model comparison report
- [ ] Performance metrics visualization

### Phase 7: Streamlit Dashboard Development (Week 4, Days 18-25)
- [ ] Design dashboard layout
- [ ] Implement input interface (text input for job posting)
- [ ] Real-time prediction display
- [ ] Visualization components:
  - Prediction confidence scores
  - Feature importance (if applicable)
  - Model comparison (baseline vs BERT)
- [ ] Add impossible jobs detection display
- [ ] Model selection toggle
- [ ] Sample examples showcase

### Phase 8: Testing & Documentation (Week 5-6, Days 26-30)
- [ ] Unit tests for key functions
- [ ] Integration testing
- [ ] Code documentation
- [ ] README with setup instructions
- [ ] User guide for dashboard
- [ ] Final report preparation

### Phase 9: Deployment & Finalization (Week 6, Days 31-42)
- [ ] Final model optimization
- [ ] Dashboard polish and bug fixes
- [ ] Performance optimization
- [ ] Final evaluation and results summary
- [ ] Project presentation preparation

## Project Structure
```
fake-job-detection/
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Cleaned and preprocessed data
│   └── models/           # Saved models
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   └── 04_bert_model.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── baseline_model.py
│   ├── bert_model.py
│   ├── impossible_jobs.py  # Bonus feature
│   ├── evaluation.py
│   └── utils.py
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── tests/
│   └── test_*.py
├── requirements.txt
├── README.md
└── PROJECT_PLAN.md
```

## Key Metrics & Goals
- **Primary Goal**: 93%+ accuracy in identifying fake job postings
- **Evaluation Metrics**:
  - Accuracy
  - Precision (minimize false positives)
  - Recall (minimize false negatives - important for fraud detection)
  - F1-score (balanced metric)
- **Bonus**: Impossible jobs detection feature

## Risk Mitigation
- **Data Quality**: Early EDA to identify and handle data issues
- **Model Overfitting**: Use validation set, early stopping, regularization
- **Computational Resources**: Start with DistilBERT if resources are limited
- **Timeline**: Prioritize core functionality, impossible jobs as stretch goal

## Next Steps
1. Set up project structure
2. Download and explore dataset
3. Begin data preprocessing
4. Implement baseline model
5. Fine-tune BERT model
6. Build Streamlit dashboard

