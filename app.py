import streamlit as st
import pickle
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Student Result Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- LIGHT THEME & SHADOW CSS ---
st.markdown("""
    <style>
    /* White Background */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Clean White Card with Drop Shadow */
    .css-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 28px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -4px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* Title Styling */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        color: #1e293b;
        margin-bottom: 4px;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 24px;
    }

    /* Predict Button */
    div.stButton > button:first-child {
        width: 100%;
        background-color: #2563eb;
        color: #ffffff;
        border: none;
        padding: 14px 20px;
        font-size: 1.05rem;
        font-weight: 600;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    
    div.stButton > button:first-child:hover {
        background-color: #1d4ed8;
    }

    /* Result Cards */
    .result-pass {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        color: #15803d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }

    .result-fail {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        color: #b91c1c;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }

    .result-text {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD PICKLE MODEL ---
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading `model.pkl`: {e}")
    st.stop()

# --- HEADER ---
st.markdown('<div class="main-title">🎓 Student Performance Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter subject marks to calculate result outcome</div>', unsafe_allow_html=True)

# --- INPUT FORM ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)

hindi = st.number_input("📚 Hindi", min_value=0, max_value=100, value=75, step=1)
english = st.number_input("📖 English", min_value=0, max_value=100, value=80, step=1)
science = st.number_input("🔬 Science", min_value=0, max_value=100, value=70, step=1)
maths = st.number_input("📐 Maths", min_value=0, max_value=100, value=85, step=1)
history = st.number_input("🏛️ History", min_value=0, max_value=100, value=65, step=1)
geography = st.number_input("🌍 Geography", min_value=0, max_value=100, value=72, step=1)

# Auto-calculate total
total_calculated = hindi + english + science + maths + history + geography
st.text_input("📊 Total Marks (Out of 600)", value=str(total_calculated), disabled=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- PREDICTION AND OUTPUT ---
if st.button("Predict Result"):
    # Feature array matching 7 input features: [Hindi, English, Science, Maths, History, Geography, Total]
    features = np.array([[hindi, english, science, maths, history, geography, total_calculated]])
    
    # Raw prediction from model (0 or 1)
    raw_pred = model.predict(features)[0]

    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    
    # Map 1/0 (or truthy values) to PASS / FAIL
    if raw_pred == 1 or str(raw_pred).strip().lower() in ['1', 'pass', 'p']:
        st.balloons()
        st.markdown("""
            <div class="result-pass">
                <div style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">Final Status</div>
                <div class="result-text">PASS 🎉</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="result-fail">
                <div style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase;">Final Status</div>
                <div class="result-text">FAIL ❌</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
