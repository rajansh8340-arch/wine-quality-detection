import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page Configuration
st.set_page_config(
    page_title="Wine Quality AI Sommelier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Wine Burgundy & Gold Modern Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,800;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #72092c 0%, #b5179e 50%, #f72585 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .card {
        background: linear-gradient(145deg, #ffffff, #fdfbfb);
        border: 1px solid #ebd4dc;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 10px 25px -5px rgba(114, 9, 44, 0.07);
        margin-bottom: 20px;
    }

    .card-dark {
        background: linear-gradient(135deg, #2b0b18 0%, #4a0e28 100%);
        color: #ffffff !important;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 12px 30px rgba(74, 14, 40, 0.25);
        margin-bottom: 20px;
    }

    .metric-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .badge-good {
        background: #e6f9ed;
        color: #0e8a3a;
        border: 1px solid #a8e5bc;
    }

    .badge-bad {
        background: #fff0f3;
        color: #c9184a;
        border: 1px solid #ffccd5;
    }

    .good-result {
        border-left: 6px solid #2ec4b6;
        background: linear-gradient(90deg, rgba(46,196,182,0.12) 0%, rgba(255,255,255,0) 100%);
        padding: 20px;
        border-radius: 12px;
    }

    .bad-result {
        border-left: 6px solid #e63946;
        background: linear-gradient(90deg, rgba(230,57,70,0.12) 0%, rgba(255,255,255,0) 100%);
        padding: 20px;
        border-radius: 12px;
    }

    /* Style button */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #72092c 0%, #a01a4e 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        border: none;
        padding: 12px 28px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(114, 9, 44, 0.3);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #50061e 0%, #72092c 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(114, 9, 44, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Load Model & Dataset
@st.cache_resource
def load_resources():
    curr_dir = os.path.dirname(__file__)
    model_path = os.path.join(curr_dir, 'model.joblib')
    csv_path = os.path.join(curr_dir, 'winequality-red.csv')
    
    model = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    return model, df

try:
    model, df = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# Header Section
st.markdown('<div class="main-title">🍷 Wine Quality AI Sommelier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict red wine grade and explore chemical profiles powered by Random Forest Machine Learning</div>', unsafe_allow_html=True)

# Sidebar with Presets & Quick Actions
with st.sidebar:
    st.markdown("### 🧪 Preset Wine Samples")
    preset = st.selectbox(
        "Choose a sample profile to auto-fill:",
        ["Custom Manual Input", "🌟 Award-Winning Reserve (High Quality)", "🍷 Standard Daily Red (Average)", "📉 Low-Grade Table Wine"]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Benchmark")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.metric("Total Samples", f"{len(df):,}")
    with col_sb2:
        high_qual_pct = (df['quality'] >= 7).mean() * 100
        st.metric("High Quality", f"{high_qual_pct:.1f}%")
        
    st.info("💡 **Criteria:** Wines rated 7 or higher on the sensory scale are classified as **Good Quality**.")

# Default preset values
defaults = {
    "fixed_acidity": 8.3,
    "volatile_acidity": 0.52,
    "citric_acid": 0.27,
    "residual_sugar": 2.5,
    "chlorides": 0.08,
    "free_sulfur_dioxide": 14.0,
    "total_sulfur_dioxide": 46.0,
    "density": 0.9968,
    "pH": 3.31,
    "sulphates": 0.65,
    "alcohol": 10.4
}

if preset == "🌟 Award-Winning Reserve (High Quality)":
    defaults = {
        "fixed_acidity": 9.1,
        "volatile_acidity": 0.32,
        "citric_acid": 0.50,
        "residual_sugar": 2.2,
        "chlorides": 0.058,
        "free_sulfur_dioxide": 18.0,
        "total_sulfur_dioxide": 36.0,
        "density": 0.9952,
        "pH": 3.24,
        "sulphates": 0.88,
        "alcohol": 12.8
    }
elif preset == "🍷 Standard Daily Red (Average)":
    defaults = {
        "fixed_acidity": 7.8,
        "volatile_acidity": 0.60,
        "citric_acid": 0.18,
        "residual_sugar": 2.4,
        "chlorides": 0.085,
        "free_sulfur_dioxide": 15.0,
        "total_sulfur_dioxide": 55.0,
        "density": 0.9972,
        "pH": 3.35,
        "sulphates": 0.60,
        "alcohol": 9.8
    }
elif preset == "📉 Low-Grade Table Wine":
    defaults = {
        "fixed_acidity": 7.0,
        "volatile_acidity": 0.88,
        "citric_acid": 0.04,
        "residual_sugar": 3.6,
        "chlorides": 0.115,
        "free_sulfur_dioxide": 8.0,
        "total_sulfur_dioxide": 85.0,
        "density": 0.9984,
        "pH": 3.52,
        "sulphates": 0.48,
        "alcohol": 9.2
    }

# Input Controls Layout
st.markdown("### ⚙️ Chemical Composition Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🍋 Acidity & pH")
    fixed_acidity = st.slider(
        "Fixed Acidity (g/dm³)", 
        min_value=4.0, max_value=16.0, value=float(defaults["fixed_acidity"]), step=0.1,
        help="Tartaric acid: vital for the structure and crispness of the wine."
    )
    volatile_acidity = st.slider(
        "Volatile Acidity (g/dm³)", 
        min_value=0.1, max_value=1.6, value=float(defaults["volatile_acidity"]), step=0.01,
        help="Acetic acid: high levels cause an unpleasant vinegar-like taste."
    )
    citric_acid = st.slider(
        "Citric Acid (g/dm³)", 
        min_value=0.0, max_value=1.0, value=float(defaults["citric_acid"]), step=0.01,
        help="Adds freshness and flavor complexity in small amounts."
    )
    pH = st.slider(
        "pH Level", 
        min_value=2.7, max_value=4.1, value=float(defaults["pH"]), step=0.01,
        help="Describes how acidic or basic the wine is on a scale from 0 to 14."
    )

with col2:
    st.markdown("#### 🍬 Sugars, Salts & Density")
    residual_sugar = st.slider(
        "Residual Sugar (g/dm³)", 
        min_value=0.5, max_value=15.5, value=float(defaults["residual_sugar"]), step=0.1,
        help="Amount of sugar remaining after fermentation stops."
    )
    chlorides = st.slider(
        "Chlorides (g/dm³)", 
        min_value=0.01, max_value=0.62, value=float(defaults["chlorides"]), step=0.001,
        help="The amount of salt in the wine."
    )
    density = st.slider(
        "Density (g/cm³)", 
        min_value=0.9900, max_value=1.0040, value=float(defaults["density"]), step=0.0001, format="%.4f",
        help="Depends on the percent of alcohol and sugar content."
    )

with col3:
    st.markdown("#### 🔬 Sulfur, Sulphates & Alcohol")
    free_sulfur_dioxide = st.slider(
        "Free Sulfur Dioxide (mg/dm³)", 
        min_value=1.0, max_value=72.0, value=float(defaults["free_sulfur_dioxide"]), step=1.0,
        help="Prevents microbial growth and oxidation of wine."
    )
    total_sulfur_dioxide = st.slider(
        "Total Sulfur Dioxide (mg/dm³)", 
        min_value=6.0, max_value=290.0, value=float(defaults["total_sulfur_dioxide"]), step=1.0,
        help="Free + bound forms of SO2. Becomes evident in aroma if too high."
    )
    sulphates = st.slider(
        "Sulphates (g/dm³)", 
        min_value=0.3, max_value=2.0, value=float(defaults["sulphates"]), step=0.01,
        help="Antimicrobial and antioxidant additive (potassium sulphate)."
    )
    alcohol = st.slider(
        "Alcohol Content (% vol)", 
        min_value=8.0, max_value=15.0, value=float(defaults["alcohol"]), step=0.1,
        help="Percent alcohol content of the wine."
    )

# Prepare dataframe for prediction
feature_dict = {
    'fixed acidity': fixed_acidity,
    'volatile acidity': volatile_acidity,
    'citric acid': citric_acid,
    'residual sugar': residual_sugar,
    'chlorides': chlorides,
    'free sulfur dioxide': free_sulfur_dioxide,
    'total sulfur dioxide': total_sulfur_dioxide,
    'density': density,
    'pH': pH,
    'sulphates': sulphates,
    'alcohol': alcohol
}
input_df = pd.DataFrame([feature_dict])

st.markdown("---")

# Predict Section
col_pred, col_viz = st.columns([1, 1.2])

with col_pred:
    st.markdown("### 🎯 Sommelier Verdict")
    
    # Get prediction and probabilities
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    confidence = proba[pred] * 100
    
    if pred == 1:
        st.markdown(f"""
        <div class="good-result">
            <h2 style="color: #2ec4b6; margin-top:0;">🍷 Excellent / Premium Wine</h2>
            <p style="font-size: 1.1rem; margin-bottom: 8px;"><strong>Status:</strong> High Quality (Score 7 - 10)</p>
            <p style="font-size: 1.05rem; color: #444;">The chemical balance features balanced acidity, high alcohol content, and low volatile acidity typical of well-crafted wines.</p>
            <div class="metric-badge badge-good">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="bad-result">
            <h2 style="color: #e63946; margin-top:0;">🍇 Standard / Table Wine</h2>
            <p style="font-size: 1.1rem; margin-bottom: 8px;"><strong>Status:</strong> Average to Below Average (Score 3 - 6)</p>
            <p style="font-size: 1.05rem; color: #444;">This wine falls short of reserve tier, often due to high volatile acidity, lower alcohol concentration, or suboptimal sulphate balance.</p>
            <div class="metric-badge badge-bad">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("**Model Probabilities:**")
    st.progress(float(proba[1]))
    st.caption(f"Good Quality Probability: **{proba[1]*100:.1f}%** | Standard Quality Probability: **{proba[0]*100:.1f}%**")

with col_viz:
    st.markdown("### 📈 Key Attributes vs. Dataset Averages")
    
    # Compare with high-quality benchmark
    good_wines = df[df['quality'] >= 7]
    all_wines = df
    
    comparison_data = pd.DataFrame({
        "Attribute": ["Alcohol (% vol)", "Sulphates (g/dm³)", "Citric Acid (g/dm³)", "Volatile Acidity (g/dm³)"],
        "Current Input": [alcohol, sulphates, citric_acid, volatile_acidity],
        "High Quality Avg": [good_wines['alcohol'].mean(), good_wines['sulphates'].mean(), good_wines['citric acid'].mean(), good_wines['volatile acidity'].mean()],
        "Overall Dataset Avg": [all_wines['alcohol'].mean(), all_wines['sulphates'].mean(), all_wines['citric acid'].mean(), all_wines['volatile acidity'].mean()]
    })
    
    st.dataframe(
        comparison_data.style.format({
            "Current Input": "{:.2f}",
            "High Quality Avg": "{:.2f}",
            "Overall Dataset Avg": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("""
    > **Key Insights:**
    > - **Alcohol:** Strongest positive correlation with quality (~11.5%+ typical in top wines).
    > - **Volatile Acidity:** Lower is substantially better (< 0.40 g/dm³ prevents vinegar flaw).
    > - **Sulphates:** Act as antioxidant and preserve wine freshness.
    """)
