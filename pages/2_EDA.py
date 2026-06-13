import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import load_dataset

st.set_page_config(
    page_title="EDA",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis")

# Dataset Selection
dataset_name = st.selectbox(
    "Select Dataset",
    [
        "Dataset 1",
        "Dataset 2",
        "Dataset 3"
    ]
)

try:

    df, target = load_dataset(dataset_name)

    st.success(f"Dataset Loaded Successfully")

    # Show detected target
    st.info(f"Detected Target Column: {target}")

    # Dataset Preview
    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # Dataset Information

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    if target in df.columns:

        try:
            fraud_cases = int(df[target].sum())

        except:
            fraud_cases = df[target].value_counts().max()

    else:

        fraud_cases = 0

    col3.metric(
        "Fraud Cases",
        fraud_cases
    )

    st.divider()

    # Column List

    st.subheader("Columns")

    st.write(df.columns.tolist())

    # Missing Values

    st.subheader("Missing Values")

    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.divider()

    # Fraud Distribution

    if target in df.columns:

        st.subheader("Fraud Distribution")

        fraud_count = (
            df[target]
            .value_counts()
            .reset_index()
        )

        fraud_count.columns = [
            "Class",
            "Count"
        ]

        fig = px.pie(
            fraud_count,
            names="Class",
            values="Count",
            hole=0.4,
            title="Fraud vs Non-Fraud"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Numeric Columns

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    )

    # Correlation Heatmap

    if len(numeric_cols.columns) > 1:

        st.subheader(
            "Correlation Heatmap"
        )

        fig2, ax = plt.subplots(
            figsize=(10, 6)
        )

        sns.heatmap(
            numeric_cols.corr(),
            annot=False,
            cmap="coolwarm",
            ax=ax
        )

        st.pyplot(fig2)

    # Transaction Amount Analysis

    amount_cols = [

        col for col in df.columns

        if "amount" in col.lower()
    ]

    if len(amount_cols) > 0:

        amount_col = amount_cols[0]

        st.subheader(
            "Transaction Amount Distribution"
        )

        fig3 = px.histogram(
            df,
            x=amount_col,
            nbins=50,
            title=amount_col
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # Numeric Feature Distribution

    st.subheader(
        "Numeric Feature Distribution"
    )

    numeric_features = numeric_cols.columns.tolist()

    if len(numeric_features) > 0:

        selected_feature = st.selectbox(
            "Select Feature",
            numeric_features
        )

        fig4 = px.box(
            df,
            y=selected_feature,
            title=f"{selected_feature} Distribution"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

except Exception as e:

    st.error(
        f"Error Loading Dataset: {e}"
    )

    st.write(
        "Available Columns (if dataset loaded):"
    )

    try:
        st.write(df.columns.tolist())
    except:
        pass