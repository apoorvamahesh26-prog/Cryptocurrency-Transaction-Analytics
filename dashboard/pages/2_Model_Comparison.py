
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = BASE_DIR / "reports" / "ml" / "final_model_comparison.csv"

st.title("📊 Machine Learning Model Comparison")
st.write(
    "Compare Random Forest and XGBoost using their final test-set results."
)
st.divider()

if not REPORT_PATH.exists():
    st.error(f"Comparison report not found: {REPORT_PATH}")
    st.stop()

data = pd.read_csv(REPORT_PATH)

required_columns = [
    "Model", "Accuracy", "Precision", "Recall",
    "F1-score", "ROC-AUC", "PR-AUC"
]

missing = [col for col in required_columns if col not in data.columns]

if missing:
    st.error(f"Missing columns in report: {missing}")
    st.stop()

# Convert metrics to percentages for display
metric_columns = [
    "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"
]

st.subheader("Final Test Performance")

metric_names = [
    ("Accuracy", "Accuracy"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
    ("F1-score", "F1 Score"),
    ("ROC-AUC", "ROC-AUC"),
    ("PR-AUC", "PR-AUC")
]

for row_index, model_row in data.iterrows():
    st.markdown(f"**{model_row['Model']}**")
    cols = st.columns(3)

    for i, (column, label) in enumerate(metric_names):
        cols[i % 3].metric(
            label,
            f"{model_row[column] * 100:.2f}%"
        )

st.divider()

st.subheader("Performance Metrics Comparison")

chart_data = data.set_index("Model")[metric_columns] * 100

st.bar_chart(chart_data, y=metric_columns)

st.divider()

st.subheader("Confusion Matrix Comparison")

confusion_columns = [
    "True Negatives",
    "False Positives",
    "False Negatives",
    "True Positives"
]

if all(col in data.columns for col in confusion_columns):

    confusion_data = data.set_index("Model")[confusion_columns]

    st.dataframe(confusion_data, width="stretch")

    fig, ax = plt.subplots(figsize=(10, 4))
    confusion_data.plot(kind="bar", ax=ax)

    ax.set_title("Confusion Matrix Values")
    ax.set_ylabel("Number of Transactions")
    ax.set_xlabel("Machine Learning Model")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Prediction Result")

    st.pyplot(fig)
    plt.close(fig)

else:
    st.info("Confusion matrix columns are not available in the CSV.")

st.divider()

st.subheader("Model Evaluation Summary")

rf = data[data["Model"] == "Random Forest"]
xgb = data[data["Model"] == "XGBoost"]

if not rf.empty and not xgb.empty:

    rf = rf.iloc[0]
    xgb = xgb.iloc[0]

    st.write(
        f"""
        - **Random Forest F1-score:** {rf['F1-score'] * 100:.2f}%
        - **XGBoost F1-score:** {xgb['F1-score'] * 100:.2f}%
        - **Random Forest Precision:** {rf['Precision'] * 100:.2f}%
        - **XGBoost Recall:** {xgb['Recall'] * 100:.2f}%
        """
    )

    st.info(
        "Random Forest achieved the higher F1-score and precision "
        "on this final test set. XGBoost detected more illicit transactions "
        "with fewer false negatives. Both models are affected by temporal "
        "distribution shifts, so these results should not be interpreted "
        "as guaranteed future performance."
    )

st.caption(
    "Evaluation uses the chronological test period, time steps 35–49. "
    "The test results were inspected during model development and are "
    "therefore retrospective rather than an untouched independent evaluation."
)
