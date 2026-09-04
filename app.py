import streamlit as st
import pickle
import numpy as np
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS WITH SHADOWS & VERTICAL CARD DESIGN ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Elevated Card Style with Shadows */
    .css-card {
        background: #1e293b;
        border-radius: 16px;
        padding: 28px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
    }

    /* Main Header */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 24px;
    }

    /* Input Fields Styling */
    .stNumberInput div[data-baseweb="input"] {
        background-color: #0f172a !important;
        border-color: #475569 !important;
        color: #f8fafc !important;
        border-radius: 8px;
    }

    /* Predict Button Styling */
    div.stButton > button:first-child {
        width: 100%;
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        color: #ffffff;
        border: none;
        padding: 16px 28px;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.5);
    }
    
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.6);
        background: linear-gradient(90deg, #4338ca 0%, #6d28d9 100%);
    }

    /* Output Results Box */
    .result-card {
        background: #064e3b;
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
    }
    
    .result-value {
        font-size: 2rem;
        font-weight: 800;
        color: #34d399;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file (`model.pkl`): {e}")
    st.stop()

# --- HEADER SECTION ---
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter subject marks below to generate model prediction</div>', unsafe_allow_html=True)

# --- VERTICAL FORM SECTION ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)

hindi = st.number_input("📚 Hindi", min_value=0, max_value=100, value=75, step=1)
english = st.number_input("📖 English", min_value=0, max_value=100, value=80, step=1)
science = st.number_input("🔬 Science", min_value=0, max_value=100, value=70, step=1)
maths = st.number_input("📐 Maths", min_value=0, max_value=100, value=85, step=1)
history = st.number_input("🏛️ History", min_value=0, max_value=100, value=65, step=1)
geography = st.number_input("🌍 Geography", min_value=0, max_value=100, value=72, step=1)

# Auto-calculate total
total_calculated = hindi + english + science + maths + history + geography
st.text_input("📊 Total Marks (Auto-calculated)", value=f"{total_calculated} / 600", disabled=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- ACTION & PREDICTION SECTION ---
if st.button("🚀 Predict Outcome"):
    st.balloons()

    with st.spinner("Processing prediction..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.003)
            progress_bar.progress(i + 1)

        # Build feature vector: ['Hindi', 'English', 'Science', 'Maths', 'History', 'Geography', 'Total']
        features = np.array([[hindi, english, science, maths, history, geography, total_calculated]])
        prediction = model.predict(features)[0]

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]

    # Display Result Card
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="result-card">
            <div style="color: #a7f3d0; font-size: 0.9rem; font-weight: 600;">PREDICTED OUTCOME</div>
            <div class="result-value">{prediction}</div>
        </div>
    """, unsafe_allow_html=True)

    if proba is not None:
        st.write("")
        st.write("**Prediction Confidence:**")
        classes = model.classes_
        for cls, p in zip(classes, proba):
            st.progress(float(p), text=f"Class {cls}: {p*100:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)
