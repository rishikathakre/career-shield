import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import requests
from bs4 import BeautifulSoup
import re

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
        background-color: #f5f7fa;
    }
    .stButton>button {
        width: 100%;
        font-weight: bold;
        border-radius: 10px;
        height: 50px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 700;
    }
    .highlight-red {
        background-color: #ffebee;
        color: #c62828;
        padding: 2px 6px;
        border-radius: 4px;
        border: 1px solid #ef9a9a;
        font-weight: 600;
    }
    /* Card styling */
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
    }
    /* Metric cards */
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    /* Text area styling - LARGER and MORE READABLE */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #cbd5e1;
        font-size: 17px !important;
        line-height: 1.8 !important;
        padding: 20px !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        min-height: 300px !important;
    }
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
    }
    
    /* Larger fonts throughout */
    .stMarkdown {
        font-size: 17px !important;
    }
    
    .stMarkdown p, .stMarkdown li {
        font-size: 17px !important;
        line-height: 1.7 !important;
    }
    
    /* Section headers much larger */
    h2 {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h3 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
    
    /* Buttons larger text */
    .stButton>button {
        font-size: 14px !important;
        height: 45px !important;
    }
    /* Progress bar */
    .stProgress > div > div {
        background-color: #ef4444;
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

# Initialize session state for job text
if 'job_text' not in st.session_state:
    st.session_state.job_text = ""
if 'sample_loaded' not in st.session_state:
    st.session_state.sample_loaded = False

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

# --- URL SCRAPER ---
def scrape_job_from_url(url: str) -> dict:
    """
    Extract job posting text from URL.
    Supports: Indeed, LinkedIn, Glassdoor, and generic pages
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Try to find job description based on common patterns
        job_text = ""
        
        # Indeed specific - try multiple selectors
        if 'indeed.com' in url:
            selectors = [
                {'id': 'jobDescriptionText'},
                {'class': 'jobsearch-jobDescriptionText'},
                {'class': re.compile('jobDescriptionText', re.I)}
            ]
            for selector in selectors:
                desc = soup.find('div', selector)
                if desc:
                    job_text = desc.get_text(separator='\n', strip=True)
                    break
        
        # LinkedIn specific - try multiple selectors
        elif 'linkedin.com' in url:
            selectors = [
                {'class': 'show-more-less-html__markup'},
                {'class': 'description__text'},
                {'class': re.compile('description', re.I)}
            ]
            for selector in selectors:
                desc = soup.find('div', selector)
                if desc:
                    job_text = desc.get_text(separator='\n', strip=True)
                    break
        
        # Glassdoor specific
        elif 'glassdoor.com' in url:
            selectors = [
                {'class': 'jobDescriptionContent'},
                {'class': re.compile('JobDetails_jobDescription', re.I)},
                {'class': re.compile('desc', re.I)}
            ]
            for selector in selectors:
                desc = soup.find('div', selector)
                if desc:
                    job_text = desc.get_text(separator='\n', strip=True)
                    break
        
        # Generic fallback - look for common patterns
        if not job_text:
            # Try common class names and IDs
            patterns = [
                'job-description', 'job_description', 'jobdescription',
                'description', 'job-details', 'job_details', 'jobdetails',
                'posting-description', 'posting_description', 'postingdescription',
                'job-content', 'job_content', 'jobcontent'
            ]
            
            for pattern in patterns:
                # Try as class
                desc = soup.find(['div', 'section', 'article'], {'class': re.compile(pattern, re.I)})
                if desc:
                    job_text = desc.get_text(separator='\n', strip=True)
                    break
                # Try as id
                desc = soup.find(['div', 'section', 'article'], {'id': re.compile(pattern, re.I)})
                if desc:
                    job_text = desc.get_text(separator='\n', strip=True)
                    break
            
            # If still nothing, try article or main tags
            if not job_text:
                for tag in ['article', 'main', 'section']:
                    desc = soup.find(tag)
                    if desc:
                        text = desc.get_text(separator='\n', strip=True)
                        if len(text) > 200:  # Only use if substantial content
                            job_text = text
                            break
            
            # Last resort: get all paragraphs
            if not job_text:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    job_text = '\n\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        
        # Clean up the text
        if job_text:
            # Remove excessive whitespace and newlines
            job_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', job_text)
            job_text = re.sub(r' +', ' ', job_text)
            job_text = job_text.strip()
            
            # Limit length but try to end at a sentence
            if len(job_text) > 5000:
                job_text = job_text[:5000]
                last_period = job_text.rfind('.')
                if last_period > 4000:
                    job_text = job_text[:last_period + 1]
        
        if not job_text or len(job_text) < 100:
            return {'success': False, 'error': 'Could not extract enough text from URL. The site may require JavaScript or login. Try copying the job description manually.'}
        
        return {'success': True, 'text': job_text}
    
    except requests.exceptions.Timeout:
        return {'success': False, 'error': 'Request timed out. The website took too long to respond.'}
    except requests.exceptions.SSLError:
        return {'success': False, 'error': 'SSL certificate verification failed. Try copying the description manually.'}
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Could not fetch URL: {str(e)}'}
    except Exception as e:
        return {'success': False, 'error': f'Error processing URL: {str(e)}'}

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
# Enhanced Header with better styling
st.markdown("""
    <div style='text-align: center; padding: 0.8rem 0 0.6rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h1 style='color: white; font-size: 2.8rem; font-weight: 900; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
            🛡️ Career Shield
        </h1>
        <p style='color: #e0e7ff; font-size: 1rem; margin-top: 0.3rem; font-weight: 500;'>
            AI-Powered Job Fraud Detection System
        </p>
    </div>
""", unsafe_allow_html=True)

# ==== SECTION 1: INPUT AREA (FULL WIDTH) ====
st.markdown("## 📝 Job Description Input")

# Quick Load Buttons (4 across - full width)
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    if st.button("📋 Load Real Job"):
        st.session_state.job_text = """Senior Software Engineer - Backend

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

with col_btn2:
    if st.button("🚨 Load Fake Job"):
        st.session_state.job_text = """Payment Processing Coordinator - Work From Home

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

with col_btn3:
    if st.button("⚠️ Impossible Job"):
        st.session_state.job_text = """Lead Machine Learning Engineer

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

with col_btn4:
    if st.button("🗑️ Clear"):
        st.session_state.job_text = ""

# Text Area (FULL WIDTH)
job_text = st.text_area("Paste job description here:", value=st.session_state.job_text, height=300, 
                        placeholder="Paste the complete job description including company name, requirements, salary, benefits...", 
                        key="job_text_input")

# Update session state when user types
if job_text != st.session_state.job_text:
    st.session_state.job_text = job_text

# ==== URL SCRAPER SECTION (PROMINENT) ====
st.markdown("### 🔗 Fetch from URL (Experimental)")
st.caption("⚠️ Works only with simple HTML sites. Most modern job boards (LinkedIn, Indeed, Workday) require manual copy-paste.")

url_col1, url_col2 = st.columns([3, 1])

with url_col1:
    job_url = st.text_input("Job Posting URL:", placeholder="https://careers.company.com/job/12345", label_visibility="collapsed")

with url_col2:
    fetch_btn = st.button("📥 Fetch Job Description", use_container_width=True)

if fetch_btn:
    if not job_url or len(job_url.strip()) == 0:
        st.warning("⚠️ Please enter a URL first.")
    else:
        with st.spinner("Fetching job description..."):
            scrape_result = scrape_job_from_url(job_url.strip())
            if scrape_result['success']:
                st.success(f"✅ Job description fetched successfully! ({len(scrape_result['text'])} characters)")
                st.session_state.job_text = scrape_result['text']
                # Update the text area with fetched content
                st.rerun()
            else:
                st.error(f"❌ {scrape_result['error']}")
                st.info("💡 **Why this happens:** Modern job sites (LinkedIn, Indeed, Workday, etc.) use JavaScript to load content, which our scraper can't access.\n\n**✅ Solution:** Copy the job description directly from the webpage and paste it above. This gives you full control and better accuracy!")

# Scan Button (FULL WIDTH)
analyze_btn = st.button("🔍 SCAN FOR FRAUD", type="primary", use_container_width=True)

st.markdown("---")

# ==== SECTION 2: RESULTS AREA (FULL WIDTH) ====
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
            # Hybrid approach: AI leads, but strong evidence from patterns can override
            
            # Start with AI confidence
            ai_confidence = fake_score
            
            # Calculate pattern confidence (normalize rule score to 0-1 scale)
            # Rule scores typically range 0-100+, normalize to confidence
            pattern_confidence = min(rule_risk_score / 100.0, 0.95) if rule_risk_score > 0 else 0
            
            # Calculate impossible requirements confidence
            impossible_confidence = 0
            if impossible_result['has_impossible_requirements']:
                # Each impossible requirement strongly suggests fraud
                impossible_confidence = min(0.6 + (impossible_result['impossible_count'] * 0.15), 0.95)
            
            # Intelligent combination: use the highest confidence among all detectors
            # This ensures if ANY detector is confident, we catch it
            combined_confidence = max(ai_confidence, pattern_confidence, impossible_confidence)
            
            # Store for display
            rule_boost = pattern_confidence
            impossible_boost = impossible_confidence

            # --- SHOW RESULTS ---
            st.markdown("<br>", unsafe_allow_html=True)

            # Determine final verdict using hybrid approach
            is_fake = combined_confidence > 0.5

            if is_fake:
                # FAKE RESULT
                st.error("🚨 **VERDICT: HIGH RISK**")

                # Metrics in columns
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric(label="Fraud Probability", value=f"{combined_confidence:.1%}", delta="Suspicious", delta_color="inverse")
                with met_col2:
                    risk_level = "CRITICAL" if combined_confidence > 0.7 else "HIGH" if combined_confidence > 0.5 else "MEDIUM"
                    st.metric(label="Risk Level", value=risk_level, delta=f"{int(combined_confidence * 100)} points")

                    # Progress Bar
                    st.progress(float(combined_confidence), text=f"Risk Score: {combined_confidence:.1%}")

                    # Detection Method
                    st.markdown("### 🔍 Detection Method")
                    
                    # Create 2 columns for better space utilization
                    det_col1, det_col2 = st.columns([1, 1])
                    
                    with det_col1:
                        st.write(f"🤖 **AI Model**: {fake_score:.1%} confidence")
                        st.write(f"📋 **Pattern Analysis**: {rule_risk_score} risk points")
                        if impossible_result['has_impossible_requirements']:
                            st.write(f"⚠️ **Impossible Requirements**: {impossible_result['impossible_count']} found")
                        
                        st.caption(f"Combined Score: {combined_confidence:.1%} (AI: {fake_score:.1%} + Patterns: +{(combined_confidence - fake_score):.1%})")
                    
                    with det_col2:
                        # Visual Risk Breakdown - Pie chart with DISTINCT colors
                        st.markdown("#### Risk Sources")
                        
                        # Create data for visualization based on actual contributions
                        breakdown_data = {
                            'AI Model': fake_score * 100,
                        }
                        
                        # Only add if they actually contributed
                        if rule_risk_score > 0:
                            # Show rule-based contribution (normalized)
                            breakdown_data['Scam Patterns'] = min(rule_boost * 100, 30) if 'rule_boost' in locals() else min(rule_risk_score * 0.5, 30)
                        
                        if impossible_result['has_impossible_requirements']:
                            breakdown_data['Impossible Req'] = impossible_boost * 100 if 'impossible_boost' in locals() else impossible_result['impossible_count'] * 10
                        
                        # Pie chart for risk sources with DISTINCT colors
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=list(breakdown_data.keys()),
                            values=list(breakdown_data.values()),
                            hole=0.3,
                            marker=dict(colors=['#667eea', '#f59f00', '#e74c3c']),  # Blue, Orange, Red - clearly different!
                            textfont=dict(size=16)
                        )])
                        fig_pie.update_layout(
                            height=280,
                            showlegend=True,
                            margin=dict(l=10, r=10, t=10, b=10),
                            font=dict(size=14)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
                        
                        # Bar chart for risk factors
                        if risk_factors:
                            st.markdown("#### Top Risk Factors")
                            factor_names = [r['flag'].title() for r in risk_factors[:6]]  # Top 6
                            factor_scores = [r['points'] for r in risk_factors[:6]]
                            factor_colors = ['#c92a2a' if r['severity'] == 'high' else '#f59f00' if r['severity'] == 'medium' else '#fab005' for r in risk_factors[:6]]
                            
                            fig_bar = go.Figure(data=[go.Bar(
                                x=factor_scores,
                                y=factor_names,
                                orientation='h',
                                marker=dict(color=factor_colors),
                                text=factor_scores,
                                textposition='auto',
                                textfont=dict(size=15)
                            )])
                            fig_bar.update_layout(
                                xaxis_title="Risk Points",
                                height=max(250, len(factor_names) * 50),
                                showlegend=False,
                                margin=dict(l=10, r=10, t=10, b=40),
                                font=dict(size=14)
                            )
                            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

                    # Red Flags Section
                    st.markdown("### 🚩 Red Flags Detected")
                    
                    # Use expander for details
                    with st.expander("📋 View Detailed Risk Factors", expanded=True):
                        if risk_factors:
                            st.write("**Pattern-Based Indicators:**")
                            high_risk = [r for r in risk_factors if r['severity'] == 'high']
                            medium_risk = [r for r in risk_factors if r['severity'] == 'medium']
                            
                            if high_risk:
                                st.markdown("**🔴 High Risk Factors:**")
                                for r in high_risk:
                                    st.error(f"• **{r['flag'].title()}** ({r['points']} points)", icon="🚨")
                            if medium_risk:
                                st.markdown("**🟡 Medium Risk Factors:**")
                                for r in medium_risk:
                                    st.warning(f"• **{r['flag'].title()}** ({r['points']} points)", icon="⚠️")
                        elif flags:
                            for f in flags:
                                st.write(f"⚠️ Contains suspicious term: **'{f}'**")
                        else:
                            st.write("⚠️ *Language tone matches known scam patterns.*")
                    
                    # Impossible requirements
                    if impossible_result['has_impossible_requirements']:
                        st.markdown("---")
                        st.error(f"⚠️ **IMPOSSIBLE REQUIREMENTS DETECTED** ({impossible_result['impossible_count']})", icon="🚫")
                        with st.expander("🔍 View Impossible Requirements", expanded=True):
                            for req in impossible_result['impossible_requirements']:
                                col_imp1, col_imp2 = st.columns([3, 1])
                                with col_imp1:
                                    st.write(f"**{req['technology'].upper()}**")
                                    st.caption(f"Requires {req['years_required']} years experience")
                                with col_imp2:
                                    st.metric("Tech Age", f"{req['technology_age']}y", delta=f"-{req['years_required'] - req['technology_age']}y", delta_color="inverse")

            else:
                # REAL RESULT
                st.success("✅ **VERDICT: LIKELY SAFE**")

                # Metrics in columns
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric(label="Authenticity Score", value=f"{real_score:.1%}", delta="Safe", delta_color="normal")
                with met_col2:
                    trust_level = "HIGH" if real_score > 0.9 else "MEDIUM" if real_score > 0.7 else "LOW"
                    st.metric(label="Trust Level", value=trust_level, delta=f"{int(real_score * 100)} points")

                # Progress Bar
                st.progress(float(1-fake_score), text=f"Safety Score: {real_score:.1%}")

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
