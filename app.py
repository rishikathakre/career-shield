import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))
from impossible_jobs import ImpossibleJobsDetector
# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="Career Shield | AI Fraud Detection",
    page_icon="🛡️",
    layout="wide", # <--- This makes it full width!
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
    .highlight-red {
        background-color: #ffebee;
        color: #c62828;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #ef9a9a;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Path where BERT model files are saved
    model_path = "models"  # models/ folder in project root
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Model not found: {e}")
        st.info("💡 Make sure all model files are in the 'models/' folder.")
        return None, None

tokenizer, model = load_model()

# Initialize impossible jobs detector
impossible_detector = ImpossibleJobsDetector()

# --- 3. LOGIC FUNCTIONS ---
def calculate_scam_risk_score(text):
    """Calculate risk score based on multiple scam indicators"""
    text_lower = text.lower()
    risk_score = 0
    risk_factors = []
    
    # Payment-related red flags (high risk)
    payment_scams = {
        "wire transfer": 15,
        "money order": 15, 
        "cashiers check": 15,
        "cashier's check": 15,
        "check cashing": 20,
        "western union": 20,
        "payment processing": 12,
        "process payments": 12,
        "forward packages": 18,
        "package forwarding": 18,
    }
    
    # Communication red flags (medium-high risk)
    comm_flags = {
        "whatsapp": 12,
        "telegram": 12,
        "text message": 8,
        "gmail.com": 10,
        "yahoo.com": 8,
    }
    
    # Urgency/pressure tactics (medium risk)
    urgency_flags = {
        "immediate start": 10,
        "start immediately": 10,
        "act fast": 10,
        "limited positions": 8,
        "apply now": 5,
        "urgent": 8,
        "asap": 7,
    }
    
    # Unrealistic promises (high risk)
    promise_flags = {
        "no experience needed": 8,
        "no experience necessary": 8,
        "guaranteed income": 12,
        "easy money": 15,
        "work from home": 3,  # Lower score as legitimate jobs also use this
        "flexible hours": 2,
    }
    
    # Check all categories
    all_flags = {**payment_scams, **comm_flags, **urgency_flags, **promise_flags}
    
    for flag, score in all_flags.items():
        if flag in text_lower:
            risk_score += score
            risk_factors.append({
                'flag': flag,
                'severity': 'high' if score >= 15 else 'medium' if score >= 10 else 'low',
                'points': score
            })
    
    # Additional pattern checks
    if "$" in text and any(word in text_lower for word in ["week", "weekly", "per week"]):
        # High weekly pay might be suspicious
        import re
        amounts = re.findall(r'\$(\d+)', text)
        if amounts and any(int(amt) > 2000 for amt in amounts):
            risk_score += 8
            risk_factors.append({'flag': 'unusually high pay', 'severity': 'medium', 'points': 8})
    
    return risk_score, risk_factors

def highlight_keywords(text):
    """Finds and highlights suspicious words"""
    keywords = ["wire transfer", "money order", "cashiers check", "cashier's check", "immediate start",
                "no experience", "payment processing", "gmail.com", "telegram", "whatsapp",
                "check cashing", "package forwarding", "western union", "process payments"]

    found_flags = []
    processed_text = text

    for word in keywords:
        if word in text.lower():
            found_flags.append(word)
            # Simple highlight replacement (case insensitive visual)
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
    
    st.success("✅ Impossible Jobs Detector Active")
    st.caption("Checks for unrealistic tech experience")

    st.markdown("---")
    st.info("ℹ️ **About:** This tool uses a Fine-Tuned BERT model to analyze semantic patterns in job descriptions that indicate fraud. It also detects impossible requirements (e.g., 10 years of ChatGPT experience).")

# --- 5. MAIN INTERFACE ---
# Header
st.markdown("<h1 style='text-align: center; color: #1565C0;'>🛡️ Career Shield Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Paste a job description below to detect potential fraud risks.</p>", unsafe_allow_html=True)
st.markdown("---")

# Layout: 2 Columns
col_left, col_right = st.columns([1.5, 1], gap="large")

with col_left:
    st.subheader("📝 Job Description Input")

    # "Quick Load" Buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    sample_text = ""

    if col_btn1.button("📋 Load Real Job"):
        sample_text = """Senior Software Engineer - Backend

Microsoft Corporation
Redmond, WA (Hybrid)

About the Role:
We are seeking an experienced Backend Engineer to join our Cloud Services team. You will work on building scalable microservices that power Azure's infrastructure management tools.

Responsibilities:
- Design and implement RESTful APIs and microservices
- Collaborate with product managers and frontend teams
- Optimize database queries and improve system performance
- Participate in code reviews and mentor junior engineers
- Contribute to architectural decisions and technical documentation

Required Qualifications:
- Bachelor's degree in Computer Science or related field
- 5+ years of professional software development experience
- Strong proficiency in C#, .NET Core, and SQL Server
- Experience with cloud platforms (Azure, AWS, or GCP)
- Understanding of containerization (Docker) and orchestration (Kubernetes)
- Excellent problem-solving and communication skills

Preferred Qualifications:
- Master's degree in Computer Science
- Experience with message queues (RabbitMQ, Kafka)
- Knowledge of CI/CD pipelines and DevOps practices
- 3+ years with React or modern frontend frameworks

Benefits:
- Competitive salary and annual bonus
- Comprehensive health, dental, and vision coverage
- 401(k) with company matching
- Generous PTO and parental leave
- Professional development budget
- Stock purchase plan

Microsoft is an equal opportunity employer. All qualified applicants will receive consideration for employment.

To apply, visit careers.microsoft.com/us/en/job/1542876"""
    
    if col_btn2.button("🚨 Load Fake Job"):
        sample_text = """Payment Processing Coordinator - Work From Home

Global Financial Services Group
Remote Position - Immediate Start Available

Position Summary:
Join our growing team as a Payment Processing Coordinator. This is an excellent opportunity for someone looking to work from home with flexible hours. No experience necessary - we provide full training!

Primary Responsibilities:
- Process customer payments and handle money order transactions
- Coordinate check cashing operations for international clients
- Forward packages to our distribution center
- Manage wire transfer requests from overseas partners
- Maintain confidential client payment information

What We're Looking For:
- Must be 18 years or older
- Have an active checking account for payment processing
- Reliable computer and internet access
- Good communication skills
- Available for immediate start

Compensation & Benefits:
- Earn $2,800-$4,200 per month
- Weekly pay via direct deposit
- Flexible work schedule - work 2-3 hours daily
- Performance bonuses
- No experience needed - comprehensive training provided

Getting Started:
To facilitate your onboarding, we'll send you a cashier's check to cover initial equipment costs and software setup. Simply deposit the funds and use wire transfer to pay our verified technology vendor. You'll keep your first week's payment from these funds.

For fastest response, please contact our hiring coordinator on WhatsApp at +1-425-555-0198 or Telegram @hiring_coordinator. Email applications may experience delays.

Apply now - positions fill quickly! Send a brief message indicating your interest and availability to start immediately.

Contact: recruitment@globalfinservices-group.com"""
    
    if col_btn3.button("⚠️ Impossible Job"):
        sample_text = """Lead Machine Learning Engineer

Nexus AI Technologies
San Francisco, CA / Remote

Company Overview:
Nexus AI Technologies is a well-funded startup (Series B, $45M raised) building next-generation conversational AI solutions for enterprise clients including Fortune 500 companies.

Position Summary:
We're looking for an exceptional ML Engineer to lead our LLM research and deployment initiatives. This is a unique opportunity to work on cutting-edge AI systems at scale.

Core Responsibilities:
- Architect and deploy production LLM systems
- Fine-tune and optimize transformer models for specific use cases
- Lead technical design reviews and mentor junior engineers
- Collaborate with research team on model improvements
- Establish best practices for ML infrastructure

Required Experience:
- PhD or Master's in Computer Science, AI/ML, or equivalent experience
- 10+ years of hands-on experience with Large Language Models
- 9+ years working with transformer-based architectures
- 6+ years experience with ChatGPT integration and prompt engineering
- 4+ years hands-on with GPT-4 and Claude
- 12+ years of Python programming experience
- Deep expertise in PyTorch and TensorFlow
- Strong understanding of distributed systems and cloud infrastructure
- Track record of deploying ML models at scale

Preferred Qualifications:
- Publications in top-tier ML conferences (NeurIPS, ICML, ACL)
- 3+ years working with LLaMA models
- 7+ years with modern MLOps tools and practices
- Open-source contributions to ML frameworks

Technical Stack:
- Python, PyTorch, Transformers library
- Kubernetes, Docker, AWS/GCP
- Vector databases, Redis, PostgreSQL

Compensation Package:
- Base salary: $180,000 - $240,000
- Equity: 0.1% - 0.25%
- Full benefits including health, dental, vision
- 401(k) matching
- Unlimited PTO
- Learning & development budget

Our hiring process includes technical interviews, a take-home project, and team fit assessment. We value diverse perspectives and encourage applications from underrepresented groups in tech.

Apply at: careers.nexusai.io/ml-lead or email talent@nexusai.io with your resume and GitHub profile."""

    # Text Area
    job_text = st.text_area("Paste text here:", value=sample_text, height=350, placeholder="Example: 'Looking for administrative assistant...'")

    # Main Action Button
    analyze_btn = st.button("🔍 SCAN FOR FRAUD", type="primary")

with col_right:
    st.subheader("📊 Scan Results")

    if analyze_btn:
        if not job_text:
            st.warning("⚠️ Please enter text to analyze.")
        elif model is None:
            st.error("❌ Model not loaded.")
        else:
            with st.spinner("🤖 Analyzing patterns..."):
                time.sleep(1) # Visual delay for effect

                # AI Prediction
                inputs = tokenizer(job_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]

                real_score = probs[0]
                fake_score = probs[1]

                # Logic/Highlight check
                highlighted_text, flags = highlight_keywords(job_text)
                
                # Calculate rule-based risk score
                rule_risk_score, risk_factors = calculate_scam_risk_score(job_text)
                
                # Impossible jobs detection
                impossible_result = impossible_detector.detect_impossible_requirements(job_text)
                
                # Combine AI, rule-based scores, AND impossible requirements
                combined_confidence = fake_score
                
                # Impossible requirements are a MAJOR red flag
                if impossible_result['has_impossible_requirements']:
                    impossible_count = impossible_result['impossible_count']
                    # Each impossible requirement adds to suspicion
                    impossible_boost = min(0.3, impossible_count * 0.15)  # Up to 30% boost
                    combined_confidence = max(fake_score, fake_score + impossible_boost, 0.6)
                
                # Rule-based scam patterns
                if rule_risk_score > 30:  # High rule-based risk
                    combined_confidence = max(combined_confidence, 0.7)
                elif rule_risk_score > 15:  # Medium rule-based risk
                    combined_confidence = max(combined_confidence, 0.55)

                # --- SHOW RESULTS ---
                st.markdown("<br>", unsafe_allow_html=True)

                # Determine final verdict using hybrid approach
                is_fake = combined_confidence > 0.5

                if is_fake:
                    # FAKE RESULT
                    st.error("🚨 **VERDICT: HIGH RISK**")

                    # Big Metric
                    st.metric(label="Fraud Probability", value=f"{combined_confidence:.1%}", delta="Suspicious")

                    # Progress Bar
                    st.progress(float(combined_confidence), text="Risk Level: CRITICAL")

                    # Detection Method
                    st.markdown("### 🔍 Detection Method")
                    if rule_risk_score > 15:
                        st.write(f"🤖 **AI Model**: {fake_score:.1%} confidence")
                        st.write(f"📋 **Rule-Based**: {rule_risk_score} risk points")
                        st.caption("Hybrid detection: AI + Pattern matching")
                    else:
                        st.write(f"🤖 **AI Model**: {fake_score:.1%} confidence")
                        st.caption("Deep learning semantic analysis")

                    # Red Flags Section
                    st.markdown("### 🚩 Red Flags Detected")
                    if risk_factors:
                        st.write("**Pattern-Based Indicators:**")
                        high_risk = [r for r in risk_factors if r['severity'] == 'high']
                        medium_risk = [r for r in risk_factors if r['severity'] == 'medium']
                        
                        if high_risk:
                            for r in high_risk:
                                st.error(f"🔴 **{r['flag'].title()}** (High Risk)")
                        if medium_risk:
                            for r in medium_risk:
                                st.warning(f"🟡 **{r['flag'].title()}** (Medium Risk)")
                    elif flags:
                        for f in flags:
                            st.write(f"⚠️ Contains suspicious term: **'{f}'**")
                    else:
                        st.write("⚠️ *Language tone matches known scam patterns.*")
                    
                    # Impossible requirements
                    if impossible_result['has_impossible_requirements']:
                        st.markdown("---")
                        st.error(f"⚠️ **IMPOSSIBLE REQUIREMENTS DETECTED** ({impossible_result['impossible_count']})")
                        for req in impossible_result['impossible_requirements']:
                            st.write(f"❌ **{req['technology'].upper()}**: Requires {req['years_required']} years, but only existed for {req['technology_age']} years!")

                else:
                    # REAL RESULT
                    st.success("✅ **VERDICT: LIKELY SAFE**")

                    # Big Metric
                    st.metric(label="Authenticity Score", value=f"{real_score:.1%}", delta="Safe")

                    # Progress Bar
                    st.progress(float(1-fake_score), text="Trust Level: HIGH")

                    st.markdown("### 🛡️ Analysis")
                    st.write("No obvious fraud patterns detected. Standard professional language used.")
                    st.write(f"🤖 **AI Confidence**: {real_score:.1%}")
                    if rule_risk_score > 0:
                        st.write(f"📋 **Minor Flags**: {rule_risk_score} points (Below threshold)")
                    
                    # Check for impossible requirements even in safe jobs
                    if impossible_result['has_impossible_requirements']:
                        st.markdown("---")
                        st.warning(f"⚠️ **Warning: Impossible Requirements Found** ({impossible_result['impossible_count']})")
                        st.write("This job may be legitimate but has unrealistic experience requirements:")
                        for req in impossible_result['impossible_requirements']:
                            st.write(f"⚠️ **{req['technology'].upper()}**: Requires {req['years_required']} years, but only existed for {req['technology_age']} years!")
