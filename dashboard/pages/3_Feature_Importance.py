
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Feature Importance",
    page_icon="🎯",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[2]

IMPORTANCE_PATH = (
    BASE_DIR
    / "reports"
    / "ml"
    / "random_forest_feature_importance.csv"
)

st.title("🎯 Feature Importance Analysis")

st.write(
    "Explore which transaction features contribute most "
    "to the Random Forest classification model."
)

st.divider()


# --------------------------------------------------
# CHECK FILE
# --------------------------------------------------

if not IMPORTANCE_PATH.exists():
    st.error(f"Feature importance file not found: {IMPORTANCE_PATH}")
    st.stop()


# --------------------------------------------------
# LOAD FEATURE IMPORTANCE
# --------------------------------------------------

@st.cache_data
def load_importance():

    df = pd.read_csv(IMPORTANCE_PATH)

    # Identify feature and importance columns
    columns = {str(col).strip().lower(): col for col in df.columns}

    feature_col = next(
        (columns[c] for c in ["feature", "features"] if c in columns),
        None
    )

    importance_col = next(
        (
            columns[c]
            for c in ["importance", "feature_importance", "importance_score"]
            if c in columns
        ),
        None
    )

    if feature_col is None or importance_col is None:
        raise ValueError(
            "Could not identify the Feature and Importance columns. "
            f"Available columns: {list(df.columns)}"
        )

    result = df[[feature_col, importance_col]].copy()
    result.columns = ["Feature", "Importance"]

    result["Importance"] = pd.to_numeric(
        result["Importance"], errors="coerce"
    )

    result = result.dropna()
    result = result.sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)

    return result


try:
    importance_df = load_importance()

except Exception as error:
    st.error(f"Unable to load feature importance: {error}")
    st.stop()


if importance_df.empty:
    st.warning("No feature importance data is available.")
    st.stop()


# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

top_feature = importance_df.iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Features",
    f"{len(importance_df):,}"
)

col2.metric(
    "Most Important Feature",
    str(top_feature["Feature"])
)

col3.metric(
    "Highest Importance",
    f"{top_feature['Importance']:.4f}"
)

st.divider()


# --------------------------------------------------
# TOP N FEATURE SELECTION
# --------------------------------------------------

st.subheader("Top Features")

top_n = st.slider(
    "Select number of features",
    min_value=5,
    max_value=min(30, len(importance_df)),
    value=min(15, len(importance_df))
)

top_features = importance_df.head(top_n).copy()


# --------------------------------------------------
# FEATURE IMPORTANCE CHART
# --------------------------------------------------

st.subheader(f"Top {top_n} Important Features")

chart_df = top_features.sort_values(
    "Importance", ascending=True
)

fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(
    chart_df["Feature"],
    chart_df["Importance"]
)

ax.set_xlabel("Importance Score")
ax.set_ylabel("Transaction Features")
ax.set_title("Random Forest Feature Importance")

ax.grid(axis="x", alpha=0.25)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# FEATURE TABLE
# --------------------------------------------------

st.divider()

st.subheader("Feature Importance Details")

display_df = importance_df.copy()
display_df.index = range(1, len(display_df) + 1)
display_df.index.name = "Rank"

st.dataframe(
    display_df,
    width="stretch"
)


# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

csv_data = importance_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Feature Importance CSV",
    data=csv_data,
    file_name="feature_importance.csv",
    mime="text/csv"
)


# --------------------------------------------------
# PROJECT INTERPRETATION
# --------------------------------------------------

st.divider()

st.subheader("Project Interpretation")

st.write(
    f"The feature with the highest Random Forest importance "
    f"is **{top_feature['Feature']}**, with an importance score "
    f"of **{top_feature['Importance']:.4f}**."
)

st.info(
    "Feature importance measures how much a feature contributes "
    "to the Random Forest model's split-based decisions. It does "
    "not prove that the feature causes illicit activity."
)

st.caption(
    "This analysis uses the saved Random Forest feature importance report."
)
