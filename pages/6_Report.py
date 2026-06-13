import streamlit as st
import pandas as pd

st.title("📑 Final Report")

results = pd.read_csv(
    "reports/model_results.csv"
)

st.subheader(
    "Model Results"
)

st.dataframe(
    results,
    use_container_width=True
)

best_model = results.loc[
    results["ROC AUC"].idxmax()
]

st.success(
    f"""
Best Model:
{best_model['Model']}

ROC AUC:
{best_model['ROC AUC']}
"""
)

st.subheader(
    "Project Summary"
)

st.markdown(
"""
### Objectives

- Detect fraudulent transactions
- Compare ML models
- Visualize fraud patterns
- Generate actionable insights

### Models

- Logistic Regression
- Decision Tree
- Random Forest

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Outcome

Real-time Financial Fraud Detection Dashboard
using Streamlit.
"""
)