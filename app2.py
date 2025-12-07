import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time
import os
import re

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Career Shield | AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        border-radius: 8px;
        height: 50px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL (WITH ERROR FIX) ---
@st.cache_resource
def load_model():
    # Path where your model is saved in Colab
    local_path = "/content/bert_local_save"
    
    # Check if folder exists first
    if not os.path.exists(local_path):
        st.error(f"❌ Error: Could not find model folder at '{local_path}'. Did you run the training step?")
        return None, None

    try:
        # Load Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        model = AutoModelForSequenceClassification.from_pretrained(local_path)
        return tokenizer, model
    except Exception as e:
        # If it fails, print the exact reason
        st.error(f"❌ Model Load Error: {e}")
        st.info("💡 Tip: If the error says 'safetensors', try running '!pip install safetensors' in a Colab cell.")
        return None, None

tokenizer, model = load_model()

# --- 3. LOGIC FUNCTIONS ---
def highlight_keywords(text):
    """Finds and highlights suspicious words"""
    keywords = ["wire transfer", "money order", "cashiers check", "immediate start", 
                "no experience", "payment processing", "gmail.com", "telegram", "whatsapp",
                "check cashing", "package forwarding"]
    
    found_flags = []
    processed_text = text
    
    for word in keywords:
        if word in text.lower():
            found_flags.append(word)
            # Visual highlight
            processed_text = processed_text.replace(word, f":red[**{word}**]")
            processed_text = processed_text.replace(word.title(), f":red[**{word.title()}**]")
            
    return processed_text, found_flags

# --- 4. SIDEBAR UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9565/9565655.png", width=80)
    st.title("Career Shield 🛡️")
    st.caption("AI-Powered Job Scams Detector")
    st.markdown("---")
    
    st.subheader("⚙️ System Status")
    if model:
        st.success("✅ AI Model Online")
        st.caption("Engine: DistilBERT-Uncased")
    else:
        st.error("❌ AI Model Offline")
        
    st.markdown("---")
    st.info("ℹ️ **About:** This tool uses a Fine-Tuned BERT model to analyze semantic patterns in job descriptions that indicate fraud.")

# --- 5. MAIN INTERFACE ---
st.markdown("<h1 style='text-align: center; color: #1565C0;'>🛡️ Career Shield Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Paste a job description below to detect potential fraud risks.</p>", unsafe_allow_html=True)
st.markdown("---")

col_left, col_right = st.columns([1.5, 1], gap="large")

with col_left:
    st.subheader("📝 Job Description Input")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    sample_text = ""
    
    if col_btn1.button("📋 Load Real Job"):
        sample_text = "Senior Data Analyst. Requires 5 years experience in SQL and Python. Health benefits and 401k included. Apply via company portal."
    if col_btn2.button("🚨 Load Fake Job"):
        sample_text = "URGENT!! Earn $2000/week from home. No experience needed. We send you a check for equipment. Text manager on Telegram immediately."
    if col_btn3.button("🗑️ Clear Text"):
        sample_text = ""

    job_text = st.text_area("Paste text here:", value=sample_text, height=350)
    analyze_btn = st.button("🔍 SCAN FOR FRAUD", type="primary")

with col_right:
    st.subheader("📊 Scan Results")
    
    if analyze_btn:
        if not job_text:
            st.warning("⚠️ Please enter text to analyze.")
        elif model is None:
            st.error("❌ Model not loaded. Cannot scan.")
        else:
            with st.spinner("🤖 Analyzing patterns..."):
                time.sleep(0.5)
                
                # AI Prediction
                inputs = tokenizer(job_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
                
                real_score = probs[0]
                fake_score = probs[1]
                
                # Highlight check
                highlighted_text, flags = highlight_keywords(job_text)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if fake_score > 0.5:
                    st.error("🚨 **VERDICT: HIGH RISK**")
                    st.metric(label="Fraud Probability", value=f"{fake_score:.1%}", delta="Suspicious")
                    st.progress(float(fake_score), text="Risk Level: CRITICAL")
                    
                    st.markdown("### 🚩 Red Flags Detected")
                    if flags:
                        for f in flags:
                            st.write(f"⚠️ Contains suspicious term: **'{f}'**")
                    else:
                        st.write("⚠️ *Language tone matches known scam patterns.*")
                        
                else:
                    st.success("✅ **VERDICT: LIKELY SAFE**")
                    st.metric(label="Authenticity Score", value=f"{real_score:.1%}", delta="Safe")
                    st.progress(float(1-fake_score), text="Trust Level: HIGH")
                    st.markdown("### 🛡️ Analysis")
                    st.write("No obvious fraud patterns detected.")
