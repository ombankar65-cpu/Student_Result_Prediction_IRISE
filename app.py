import streamlit as st
import pickle
import numpy as np
import time

# Optional celebration effect (falls back gracefully if not installed)
try:
    from streamlit_confetti import confetti
    HAS_CONFETTI = True
except ImportError:
    HAS_CONFETTI = False

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS & ANIMATIONS ---
st.markdown("""
    <style>
    /* Main Background & Font Styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Card Container Styling */
    .css-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    /* Custom Header */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Custom Input Labels */
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    /* Predict Button Styling & Pulse Animation */
    div.stButton > button:first-child {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        padding: 14px 28px;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
        background: linear-gradient(90deg, #4f46e5 0%, #9333ea 100%);
    }

    /* Success Metric Card */
    .metric-card {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #34d399;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD PICKLE MODEL ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pkl`: {e}")
    st.stop()

# --- APP HEADER ---
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter subject marks below to evaluate predicted outcome via KNN Model</div>', unsafe_allow_html=True)

# --- INPUT FORM AREA ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    hindi = st.number_input("📚 Hindi", min_value=0, max_value=100, value=75, step=1)
    english = st.number_input("📖 English", min_value=0, max_value=100, value=80, step=1)
    science = st.number_input("🔬 Science", min_value=0, max_value=100, value=70, step=1)
    maths = st.number_input("📐 Maths", min_value=0, max_value=100, value=85, step=1)

with col2:
    history = st.number_input("🏛️ History", min_value=0, max_value=100, value=65, step=1)
    geography = st.number_input("🌍 Geography", min_value=0, max_value=100, value=72, step=1)
    
    # Calculate Total automatically
    total_calculated = hindi + english + science + maths + history + geography
    st.text_input("📊 Total Marks (Auto-calculated)", value=str(total_calculated), disabled=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION & EFFECTS ---
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🚀 Predict Outcome")

if predict_btn:
    # Trigger native snow/balloons visual effects
    st.balloons()
    if HAS_CONFETTI:
        confetti()

    # Progress bar effect
    with st.spinner("Analyzing input features..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)
        
        # Prepare feature vector matching model's expected 7 features
        # Features: ['Hindi', 'English', 'Science', 'Maths', 'History', 'Geography', 'Total']
        features = np.array([[hindi, english, science, maths, history, geography, total_calculated]])
        
        prediction = model.predict(features)[0]
        
        # Display probabilities if available
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]

    # Display Results
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([1, 1])
    
    with res_col1:
        st.markdown(f"""
            <div class="metric-card">
                <div style="color: #94a3b8; font-size: 0.9rem;">PREDICTED CLASS</div>
                <div class="metric-value">{prediction}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown(f"""
            <div class="metric-card" style="border-color: rgba(99, 102, 241, 0.3); background: rgba(99, 102, 241, 0.1);">
                <div style="color: #94a3b8; font-size: 0.9rem;">TOTAL MARKS</div>
                <div class="metric-value" style="color: #818cf8;">{total_calculated} / 600</div>
            </div>
        """, unsafe_allow_html=True)

    if proba is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("**Prediction Confidence:**")
        classes = model.classes_
        for cls, p in zip(classes, proba):
            st.progress(float(p), text=f"Class {cls}: {p*100:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)
