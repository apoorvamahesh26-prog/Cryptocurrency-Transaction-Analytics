
from pathlib import Path

import pandas as pd
import streamlit as st


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Cryptocurrency Transaction Analytics",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
FEATURES_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "elliptic"
    / "elliptic_txs_features.csv"
)

CLASSES_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "elliptic"
    / "elliptic_txs_classes.csv"
)


# ==========================================================
# 2. CUSTOM CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
    }

    .sub-title {
        font-size: 15px;
        color: #a0aec0;
    }

    [data-testid="stMetric"] {
        background-color: #1a2233;
        border: 1px solid #26344d;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #a0aec0;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    h1, h2, h3 {
        color: #ffffff;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

@st.cache_data(show_spinner=False)
def load_data():

    # ----------------------------------------------
    # Load transaction features
    # ----------------------------------------------

    features = pd.read_csv(
        FEATURES_PATH,
        header=None
    )

    # Remove header row only if one exists
    if str(features.iloc[0, 0]).strip().lower() == "txid":
        features = features.iloc[1:].reset_index(drop=True)

    feature_columns = (
        ["txId", "time_step"]
        + [f"feature_{i}" for i in range(1, 166)]
    )

    if features.shape[1] != len(feature_columns):
        raise ValueError(
            f"Features file should contain 167 columns, "
            f"but found {features.shape[1]}."
        )

    features.columns = feature_columns

    # ----------------------------------------------
    # Load transaction classes
    # ----------------------------------------------

    classes = pd.read_csv(
        CLASSES_PATH,
        header=None
    )

    # Remove header row only if one exists
    if str(classes.iloc[0, 0]).strip().lower() == "txid":
        classes = classes.iloc[1:].reset_index(drop=True)

    if classes.shape[1] != 2:
        raise ValueError(
            f"Classes file should contain 2 columns, "
            f"but found {classes.shape[1]}."
        )

    classes.columns = ["txId", "class"]

    # ----------------------------------------------
    # Convert transaction IDs
    # ----------------------------------------------

    features["txId"] = pd.to_numeric(
        features["txId"],
        errors="raise"
    )

    classes["txId"] = pd.to_numeric(
        classes["txId"],
        errors="raise"
    )

    # ----------------------------------------------
    # Merge features and classes
    # ----------------------------------------------

    data = features.merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one"
    )

    # ----------------------------------------------
    # Normalize transaction labels
    # ----------------------------------------------

    class_values = (
        data["class"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\.0$", "", regex=True)
    )

    data["Transaction Type"] = class_values.map({
        "1": "Illicit",
        "2": "Licit",
        "unknown": "Unknown"
    }).fillna("Unknown")

    return data


# ==========================================================
# 4. DASHBOARD HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">'
    '₿ Cryptocurrency Transaction Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Big Data Analytics and Machine Learning '
    'for Illicit Cryptocurrency Transaction Detection'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================================
# 5. SIDEBAR
# ==========================================================

with st.sidebar:

    st.title("₿ Crypto Analytics")

    st.caption("Transaction Intelligence Platform")

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select Section",
        [
            "Dashboard Overview",
            "Transaction Analysis"
        ]
    )

    st.divider()

    st.caption("Dataset: Elliptic Bitcoin Transaction Dataset")

    st.caption("Total Time Steps: 49")


# ==========================================================
# 6. DATASET VALIDATION AND LOADING
# ==========================================================

if not FEATURES_PATH.exists():
    st.error(f"Feature dataset not found: {FEATURES_PATH}")
    st.stop()

if not CLASSES_PATH.exists():
    st.error(f"Class dataset not found: {CLASSES_PATH}")
    st.stop()

try:

    with st.spinner("Loading cryptocurrency transaction dataset..."):
        data = load_data()

except Exception as error:

    st.error(f"Unable to load dataset: {error}")
    st.stop()


st.success("Cryptocurrency transaction dataset loaded successfully!")


# ==========================================================
# 7. CALCULATE TRANSACTION STATISTICS
# ==========================================================

total_transactions = len(data)

licit_transactions = (
    data["Transaction Type"] == "Licit"
).sum()

illicit_transactions = (
    data["Transaction Type"] == "Illicit"
).sum()

unknown_transactions = (
    data["Transaction Type"] == "Unknown"
).sum()

illicit_percentage = (
    illicit_transactions / total_transactions * 100
    if total_transactions > 0 else 0
)

licit_percentage = (
    licit_transactions / total_transactions * 100
    if total_transactions > 0 else 0
)


# ==========================================================
# 8. DASHBOARD OVERVIEW
# ==========================================================

if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    st.write(
        "Monitor cryptocurrency transaction activity "
        "and explore licit, illicit and unknown transactions."
    )

    # ----------------------------------------------
    # KPI CARDS
    # ----------------------------------------------

    st.subheader("Transaction Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        f"{total_transactions:,}"
    )

    col2.metric(
        "Licit Transactions",
        f"{licit_transactions:,}"
    )

    col3.metric(
        "Illicit Transactions",
        f"{illicit_transactions:,}",
        f"{illicit_percentage:.2f}% of total"
    )

    col4.metric(
        "Unknown Transactions",
        f"{unknown_transactions:,}"
    )

    st.divider()

    # ----------------------------------------------
    # CLASS DISTRIBUTION
    # ----------------------------------------------

    st.subheader("Transaction Class Distribution")

    class_counts = (
        data["Transaction Type"]
        .value_counts()
        .reindex(
            ["Licit", "Illicit", "Unknown"],
            fill_value=0
        )
    )

    class_percentages = (
        class_counts / total_transactions * 100
    ).round(2)

    left, right = st.columns(2)

    with left:

        st.markdown("#### Transaction Count")

        st.bar_chart(class_counts)

    with right:

        st.markdown("#### Transaction Percentage")

        percentage_df = pd.DataFrame({
            "Transaction Type": class_percentages.index,
            "Percentage": class_percentages.values
        })

        st.dataframe(
            percentage_df,
            hide_index=True,
            use_container_width=True
        )

    st.divider()

    # ----------------------------------------------
    # TIME STEP ANALYSIS
    # ----------------------------------------------

    st.subheader("Transaction Activity Over Time")

    time_data = (
        data.groupby(["time_step", "Transaction Type"])
        .size()
        .unstack(fill_value=0)
    )

    time_data = time_data.reindex(
        columns=["Licit", "Illicit", "Unknown"],
        fill_value=0
    )

    st.line_chart(time_data)

    st.caption(
        "Transaction distribution across the 49 time steps."
    )

    st.divider()

    # ----------------------------------------------
    # DATASET INFORMATION
    # ----------------------------------------------

    st.subheader("Dataset Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    info_col1.metric(
        "Number of Features",
        "165"
    )

    info_col2.metric(
        "Number of Time Steps",
        data["time_step"].nunique()
    )

    info_col3.metric(
        "Total Dataset Columns",
        data.shape[1]
    )


# ==========================================================
# 9. TRANSACTION ANALYSIS
# ==========================================================

elif page == "Transaction Analysis":

    st.header("Transaction Analysis")

    st.write(
        "Explore transaction records using the transaction "
        "classification filter."
    )

    # ----------------------------------------------
    # FILTER
    # ----------------------------------------------

    selected_type = st.selectbox(
        "Select Transaction Type",
        [
            "All",
            "Licit",
            "Illicit",
            "Unknown"
        ]
    )

    selected_time = st.selectbox(
        "Select Time Step",
        ["All"] + sorted(
            data["time_step"].unique().tolist()
        )
    )

    filtered_data = data.copy()

    if selected_type != "All":

        filtered_data = filtered_data[
            filtered_data["Transaction Type"] == selected_type
        ]

    if selected_time != "All":

        filtered_data = filtered_data[
            filtered_data["time_step"] == selected_time
        ]

    st.divider()

    # ----------------------------------------------
    # FILTERED STATISTICS
    # ----------------------------------------------

    st.metric(
        "Matching Transactions",
        f"{len(filtered_data):,}"
    )

    # ----------------------------------------------
    # TRANSACTION TABLE
    # ----------------------------------------------

    st.subheader("Transaction Records")

    display_columns = (
        ["txId", "time_step", "Transaction Type"]
        + [f"feature_{i}" for i in range(1, 11)]
    )

    st.dataframe(
        filtered_data[display_columns].head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing up to 100 records. "
        "The original dataset remains unchanged."
    )

    # ----------------------------------------------
    # DOWNLOAD FILTERED DATA
    # ----------------------------------------------

    csv_data = filtered_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Filtered Transactions CSV",
        data=csv_data,
        file_name="filtered_crypto_transactions.csv",
        mime="text/csv"
    )


# ==========================================================
# 10. FOOTER
# ==========================================================

st.divider()

st.caption(
    "Cryptocurrency Transaction Analytics | "
    "Big Data Analytics Project | Elliptic Dataset"
)
