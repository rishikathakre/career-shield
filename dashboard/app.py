"""
Streamlit Dashboard for Fake Job Posting Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.baseline_model import BaselineModel
from src.bert_model import BERTModel
from src.impossible_jobs import ImpossibleJobsDetector

# Page configuration
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Fake Job Posting Detection System")
st.markdown("---")

# Initialize session state
if 'baseline_model' not in st.session_state:
    st.session_state.baseline_model = None
if 'bert_model' not in st.session_state:
    st.session_state.bert_model = None
if 'impossible_detector' not in st.session_state:
    st.session_state.impossible_detector = ImpossibleJobsDetector()


@st.cache_resource
def load_baseline_model():
    """Load baseline model."""
    try:
        model = BaselineModel()
        model.load('data/models')
        return model
    except Exception as e:
        st.warning(f"Could not load baseline model: {e}")
        return None


@st.cache_resource
def load_bert_model():
    """Load BERT model."""
    try:
        model = BERTModel()
        model.load('data/models/bert')
        return model
    except Exception as e:
        st.warning(f"Could not load BERT model: {e}")
        return None


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.radio(
        "Select Model",
        ["Baseline (TF-IDF + Logistic Regression)", "BERT/DistilBERT", "Both"]
    )
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    This tool helps detect fake job postings using NLP techniques.
    
    **Features:**
    - Baseline model (TF-IDF + Logistic Regression)
    - Fine-tuned BERT/DistilBERT model
    - Impossible jobs detection (bonus feature)
    
    **How to use:**
    1. Enter a job posting in the text area
    2. Click "Analyze" to get predictions
    3. View results and confidence scores
    """)


# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Single Analysis", "📊 Batch Analysis", "📈 Model Comparison"])

with tab1:
    st.header("Analyze a Job Posting")
    
    # Text input
    job_text = st.text_area(
        "Enter job posting text:",
        height=300,
        placeholder="Paste the job posting description here..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button and job_text:
        with st.spinner("Analyzing job posting..."):
            results = {}
            
            # Load models if needed
            if model_choice in ["Baseline (TF-IDF + Logistic Regression)", "Both"]:
                baseline_model = load_baseline_model()
                if baseline_model:
                    try:
                        pred = baseline_model.predict(pd.Series([job_text]))[0]
                        proba = baseline_model.predict_proba(pd.Series([job_text]))[0]
                        results['baseline'] = {
                            'prediction': 'Fake' if pred == 1 else 'Real',
                            'confidence': float(max(proba)),
                            'probabilities': {
                                'Real': float(proba[0]),
                                'Fake': float(proba[1])
                            }
                        }
                    except Exception as e:
                        st.error(f"Baseline model error: {e}")
            
            if model_choice in ["BERT/DistilBERT", "Both"]:
                bert_model = load_bert_model()
                if bert_model:
                    try:
                        pred = bert_model.predict(pd.Series([job_text]))[0]
                        proba = bert_model.predict_proba(pd.Series([job_text]))[0]
                        results['bert'] = {
                            'prediction': 'Fake' if pred == 1 else 'Real',
                            'confidence': float(max(proba)),
                            'probabilities': {
                                'Real': float(proba[0]),
                                'Fake': float(proba[1])
                            }
                        }
                    except Exception as e:
                        st.error(f"BERT model error: {e}")
            
            # Impossible jobs detection
            impossible_result = st.session_state.impossible_detector.detect_impossible_requirements(job_text)
            results['impossible'] = impossible_result
        
        # Display results
        st.markdown("---")
        st.header("📊 Results")
        
        # Prediction results
        if results:
            cols = st.columns(len(results) if 'impossible' not in results else len(results) - 1)
            col_idx = 0
            
            if 'baseline' in results:
                with cols[col_idx]:
                    st.subheader("Baseline Model")
                    pred = results['baseline']['prediction']
                    conf = results['baseline']['confidence']
                    color = "🔴" if pred == "Fake" else "🟢"
                    st.metric("Prediction", f"{color} {pred}", f"{conf:.2%} confidence")
                    
                    # Probability chart
                    fig = px.bar(
                        x=list(results['baseline']['probabilities'].keys()),
                        y=list(results['baseline']['probabilities'].values()),
                        labels={'x': 'Class', 'y': 'Probability'},
                        title="Prediction Probabilities"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                col_idx += 1
            
            if 'bert' in results:
                with cols[col_idx]:
                    st.subheader("BERT Model")
                    pred = results['bert']['prediction']
                    conf = results['bert']['confidence']
                    color = "🔴" if pred == "Fake" else "🟢"
                    st.metric("Prediction", f"{color} {pred}", f"{conf:.2%} confidence")
                    
                    # Probability chart
                    fig = px.bar(
                        x=list(results['bert']['probabilities'].keys()),
                        y=list(results['bert']['probabilities'].values()),
                        labels={'x': 'Class', 'y': 'Probability'},
                        title="Prediction Probabilities"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                col_idx += 1
        
        # Impossible requirements
        if 'impossible' in results and results['impossible']['has_impossible_requirements']:
            st.markdown("---")
            st.header("⚠️ Impossible Requirements Detected")
            st.warning(f"Found {results['impossible']['impossible_count']} impossible requirement(s)!")
            
            for req in results['impossible']['impossible_requirements']:
                st.error(
                    f"**{req['technology'].title()}**: Requires {req['years_required']} years, "
                    f"but technology is only {req['technology_age']} years old!"
                )
        elif 'impossible' in results:
            st.info("✅ No impossible requirements detected.")


with tab2:
    st.header("Batch Analysis")
    st.markdown("Upload a CSV file with job postings to analyze multiple postings at once.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df)} job postings")
        st.dataframe(df.head())
        
        if st.button("Analyze All"):
            st.info("Batch analysis feature - to be implemented")


with tab3:
    st.header("Model Comparison")
    st.markdown("Compare performance metrics of different models.")
    
    st.info("Model comparison metrics will be displayed here after training.")
    
    # Placeholder for model comparison charts
    # This would show accuracy, precision, recall, F1-score comparison


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Fake Job Posting Detection System | NLP Final Project</div>",
    unsafe_allow_html=True
)

