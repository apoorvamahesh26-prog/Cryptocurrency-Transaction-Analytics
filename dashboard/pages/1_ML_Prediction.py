
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ML Prediction",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURES_PATH = BASE_DIR / "data" / "raw" / "elliptic" / "elliptic_txs_features.csv"
CLASSES_PATH = BASE_DIR / "data" / "raw" / "elliptic" / "elliptic_txs_classes.csv"

RF_MODEL_PATH = BASE_DIR / "models" / "random_forest_model.joblib"
XGB_MODEL_PATH = BASE_DIR / "models" / "xgboost_final_model.joblib"

FEATURE_COLUMNS = [f"feature_{i}" for i in range(1, 166)]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_transaction_data():

    features = pd.read_csv(FEATURES_PATH, header=None)

    if str(features.iloc[0, 0]).strip().lower() == "txid":
        features = features.iloc[1:].reset_index(drop=True)

    if features.shape[1] != 167:
        raise ValueError(
            f"Expected 167 feature columns, found {features.shape[1]}."
        )

    features.columns = ["txId", "time_step"] + FEATURE_COLUMNS

    classes = pd.read_csv(CLASSES_PATH, header=None)

    if str(classes.iloc[0, 0]).strip().lower() == "txid":
        classes = classes.iloc[1:].reset_index(drop=True)

    if classes.shape[1] != 2:
        raise ValueError(
            f"Expected 2 class columns, found {classes.shape[1]}."
        )

    classes.columns = ["txId", "class"]

    features["txId"] = pd.to_numeric(features["txId"])
    features["time_step"] = pd.to_numeric(features["time_step"])

    classes["txId"] = pd.to_numeric(classes["txId"])
    classes["class"] = classes["class"].astype(str).str.strip().str.lower()

    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )

    return features, classes


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():

    rf_model = joblib.load(RF_MODEL_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)

    return rf_model, xgb_model


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🤖 Cryptocurrency Transaction Prediction")

st.write(
    "Use trained Machine Learning models to analyse an "
    "unknown cryptocurrency transaction."
)

st.divider()


# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------

required_files = [
    FEATURES_PATH,
    CLASSES_PATH,
    RF_MODEL_PATH,
    XGB_MODEL_PATH
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("Required project files are missing:")

    for file in missing_files:
        st.write(file)

    st.stop()


# --------------------------------------------------
# LOAD DATA AND MODELS
# --------------------------------------------------

try:
    with st.spinner("Loading transaction data and ML models..."):
        features, classes = load_transaction_data()
        rf_model, xgb_model = load_models()

except Exception as error:
    st.error(f"Unable to load data or models: {error}")
    st.stop()


# --------------------------------------------------
# FIND UNKNOWN TRANSACTIONS
# --------------------------------------------------

unknown_ids = set(
    classes.loc[classes["class"] == "unknown", "txId"].tolist()
)

unknown_features = features[
    features["txId"].isin(unknown_ids)
]

if unknown_features.empty:
    st.warning("No unknown transactions were found.")
    st.stop()


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Available Unknown Transactions", f"{len(unknown_features):,}")
col2.metric("Input Features", "165")
col3.metric("Available ML Models", "2")

st.divider()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("Select Transaction")

default_transaction_id = str(
    int(unknown_features.iloc[0]["txId"])
)

transaction_id = st.text_input(
    "Enter an Unknown Transaction ID",
    value=default_transaction_id,
    help="Enter a transaction ID from the unknown transactions dataset."
)

model_choice = st.selectbox(
    "Choose Machine Learning Model",
    ["Random Forest", "XGBoost"]
)

st.caption(
    "You can find transaction IDs in the Transaction Analysis page "
    "of the main dashboard."
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("Predict Transaction", type="primary", width="stretch"):

    try:
        selected_id = int(transaction_id.strip())

        matching_rows = unknown_features[
            unknown_features["txId"] == selected_id
        ]

        if matching_rows.empty:
            st.error(
                "Transaction ID not found among unknown transactions. "
                "Please enter a valid unknown transaction ID."
            )
            st.stop()

        selected_transaction = matching_rows.iloc[[0]]

        X = selected_transaction[FEATURE_COLUMNS]

        if X.isnull().any().any():
            st.error("This transaction contains missing feature values.")
            st.stop()

        if model_choice == "Random Forest":
            model = rf_model
        else:
            model = xgb_model

        if model.n_features_in_ != len(FEATURE_COLUMNS):
            st.error("Model feature count does not match the dataset.")
            st.stop()

        prediction = int(model.predict(X)[0])

        probabilities = model.predict_proba(X)[0]

        positive_index = list(model.classes_).index(1)
        illicit_probability = float(probabilities[positive_index])

        licit_probability = 1.0 - illicit_probability

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Transaction ID",
            str(selected_id)
        )

        result_col2.metric(
            "Time Step",
            int(selected_transaction.iloc[0]["time_step"])
        )

        if prediction == 1:

            st.error("⚠️ Prediction: POTENTIALLY ILLICIT")

        else:

            st.success("✅ Prediction: LICIT")

        prob1, prob2 = st.columns(2)

        prob1.metric(
            "Licit Probability",
            f"{licit_probability * 100:.2f}%"
        )

        prob2.metric(
            "Illicit Probability",
            f"{illicit_probability * 100:.2f}%"
        )

        st.write("### Prediction Probability")

        probability_data = pd.DataFrame({
            "Class": ["Licit", "Illicit"],
            "Probability": [
                licit_probability,
                illicit_probability
            ]
        })

        st.bar_chart(
            probability_data.set_index("Class"),
            y="Probability"
        )

        st.info(
            f"Model used: {model_choice} | "
            f"Decision threshold: 0.50"
        )

        st.warning(
            "This is an experimental model prediction, not a confirmed "
            "fraud determination. The dataset shows significant temporal "
            "distribution shifts, so predictions may be unreliable for "
            "some later time periods."
        )

    except ValueError:
        st.error("Please enter a valid numeric transaction ID.")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
