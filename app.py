import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD CSS
# ==========================================

def load_css():
    try:
        with open(
            "assets/styles.css",
            encoding="utf-8"
        ) as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

    except:
        pass

load_css()

# ==========================================
# HEADER
# ==========================================

st.markdown(
    """
    <div class='main-title'>
        💳 Financial Fraud Detection Dashboard
    </div>

    <div class='sub-title'>
        AI-Powered Fraud Analytics & Risk Monitoring System
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# KPI SECTION
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Datasets",
        "3"
    )

with col2:
    st.metric(
        "ML Models",
        "3"
    )

with col3:
    st.metric(
        "Analytics Pages",
        "6"
    )

with col4:
    st.metric(
        "Prediction",
        "Real-Time"
    )

st.divider()

# ==========================================
# OVERVIEW
# ==========================================

st.subheader("📌 Project Overview")

st.markdown("""
This project uses **Machine Learning** and **Data Analytics**
to identify fraudulent financial transactions and monitor risk.

### Features

- Multi Dataset Support
- Data Preprocessing
- Exploratory Data Analysis
- Logistic Regression
- Decision Tree
- Random Forest
- ROC Curve Analysis
- Confusion Matrix Analysis
- Fraud Prediction
- Interactive Dashboard
""")

st.divider()

# ==========================================
# DATASETS
# ==========================================

st.subheader("📂 Available Datasets")

c1, c2, c3 = st.columns(3)

with c1:
    st.info(
        """
        💳 Credit Card Fraud Dataset

        Detect fraudulent card transactions.
        """
    )

with c2:
    st.info(
        """
        💰 Online Payment Fraud Dataset

        Analyze digital payment fraud.
        """
    )

with c3:
    st.info(
        """
        🏦 Bank Transaction Fraud Dataset

        Monitor suspicious banking activity.
        """
    )

st.divider()

# ==========================================
# WORKFLOW
# ==========================================

st.subheader("⚙️ System Workflow")

st.code(
"""
Datasets
    │
    ▼
Data Collection
    │
    ▼
Data Preprocessing
    │
    ▼
Feature Engineering
    │
    ▼
Model Training
(Logistic Regression,
 Decision Tree,
 Random Forest)
    │
    ▼
Model Evaluation
    │
    ▼
Fraud Prediction
    │
    ▼
Dashboard Reporting
""",
language="text"
)

st.divider()

# ==========================================
# NAVIGATION
# ==========================================

st.subheader("🧭 Dashboard Navigation")

st.markdown("""
### Pages Available

🏠 Home

📊 Exploratory Data Analysis

🤖 Model Training

📈 Model Comparison

🚨 Fraud Prediction

📑 Reports
""")

st.success(
    "Use the left sidebar to navigate through dashboard pages."
)

st.divider()

# ==========================================
# TECHNOLOGY STACK
# ==========================================

st.subheader("🛠 Technology Stack")

tech1, tech2, tech3, tech4 = st.columns(4)

with tech1:
    st.metric(
        "Frontend",
        "Streamlit"
    )

with tech2:
    st.metric(
        "ML",
        "Scikit-Learn"
    )

with tech3:
    st.metric(
        "Data",
        "Pandas"
    )

with tech4:
    st.metric(
        "Visualization",
        "Plotly"
    )

st.divider()

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div style='text-align:center;color:gray;'>
        Financial Fraud Detection Dashboard |
        Machine Learning & Data Analytics Project
    </div>
    """,
    unsafe_allow_html=True
)