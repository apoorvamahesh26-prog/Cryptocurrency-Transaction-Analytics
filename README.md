
# Cryptocurrency Transaction Analytics

A Big Data Analytics and Machine Learning project for analysing Bitcoin transactions and identifying potentially illicit transactions using PySpark, Random Forest, XGBoost, and Streamlit.

## Project Overview

Cryptocurrency Transaction Analytics is a data-driven project that analyses large-scale Bitcoin transaction data using Big Data technologies and Machine Learning.

The project uses the Elliptic Bitcoin Transaction Dataset to perform data quality analysis, exploratory data analysis, preprocessing, transaction classification, model evaluation, and temporal distribution shift analysis.

An interactive Streamlit dashboard presents the analytical results and model predictions.

## Objectives

- Analyse large-scale cryptocurrency transaction data.
- Perform data quality checking and preprocessing.
- Explore transaction patterns across time steps.
- Identify potentially illicit transactions using Machine Learning.
- Compare Random Forest and XGBoost models.
- Analyse feature importance and temporal distribution shifts.
- Present analytical results through an interactive dashboard.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PySpark | Big Data processing |
| Pandas | Data manipulation |
| PyArrow | Parquet file handling |
| Scikit-learn | Machine Learning and evaluation |
| XGBoost | Gradient boosting classification |
| Matplotlib | Data visualization |
| Streamlit | Interactive dashboard |
| Joblib | Model serialization |

## Dataset

**Dataset:** Elliptic Bitcoin Transaction Dataset

The dataset contains:

- 203,769 transactions
- 234,355 transaction edges
- 165 transaction features
- 49 time steps

### Class Distribution

| Class | Transactions |
|---|---:|
| Licit | 42,019 |
| Illicit | 4,545 |
| Unknown | 157,205 |
| **Total** | **203,769** |

Unknown transactions are excluded from labelled model training and evaluation.

**Dataset setup:** The raw dataset is not included in this GitHub repository. Download the Elliptic dataset separately and place its three CSV files in:

`data/raw/elliptic/`

Required files:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

## Project Workflow

1. Data collection
2. Data quality checking
3. Exploratory Data Analysis
4. Data preprocessing using PySpark
5. Feature validation
6. Chronological train-test splitting
7. Machine Learning model training
8. Model comparison and evaluation
9. Temporal drift analysis
10. Feature importance analysis
11. Interactive dashboard development

## Machine Learning Models

### 1. Random Forest

- Number of estimators: 200
- Class weighting: `balanced_subsample`
- Random state: 42

### 2. XGBoost

- Number of estimators: 300
- Maximum depth: 7
- Learning rate: 0.05
- Subsample: 0.8
- Column sampling: 0.8

## Model Performance Comparison

Evaluation on the chronological test set (time steps 35–49).

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 98.04% | 97.53% |
| Precision | 97.49% | 86.36% |
| Recall | 71.65% | 73.68% |
| F1-score | 82.60% | 79.52% |
| ROC-AUC | 93.42% | 93.33% |
| PR-AUC | 79.68% | 80.06% |

**Observation:** Random Forest achieved higher precision and F1-score, while XGBoost achieved slightly higher recall and PR-AUC on the final test set.

These are retrospective results from the project evaluation, not evidence of reliable future-period performance.

## Dashboard Features

The Streamlit dashboard provides:

1. Dashboard Overview
2. Transaction Analysis
3. Machine Learning Prediction
4. Model Comparison
5. Feature Importance

The dashboard uses trained Random Forest and XGBoost models to support transaction classification and visualization.

## Project Structure

```text
Cryptocurrency-Transaction-Analytics/
│
├── dashboard/
│   ├── app.py
│   └── pages/
│       ├── 1_ML_Prediction.py
│       ├── 2_Model_Comparison.py
│       └── 3_Feature_Importance.py
│
├── data/
│   ├── raw/
│   │   └── elliptic/           # Download separately
│   └── processed/              # Generated locally
│
├── models/
│   ├── random_forest_model.joblib
│   └── xgboost_final_model.joblib
│
├── reports/
│   ├── eda/
│   └── ml/
│
├── requirements.txt
├── preprocessing.py
├── train_model.py
├── model_comparison.py
├── exploratory_analysis.py
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/apoorvamahesh26-prog/Cryptocurrency-Transaction-Analytics.git
```

### 2. Navigate to the Project Folder

```bash
cd Cryptocurrency-Transaction-Analytics
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure the Dataset

Download the Elliptic Bitcoin Transaction Dataset and place the three required CSV files inside:

```text
data/raw/elliptic/
```

The raw dataset is required for the dashboard's transaction prediction functionality.

## Run the Dashboard

From the project root directory, execute:

```bash
streamlit run dashboard/app.py
```

Open the local URL displayed in your terminal:

```text
http://localhost:8501
```

## Important Limitations

- A large proportion of transactions in the dataset have unknown labels.
- Chronological splitting is used to simulate prediction on later time steps.
- Significant temporal distribution shifts were observed, particularly in later test time steps.
- Model performance can deteriorate substantially under temporal distribution shift.
- Test results were inspected during model development and should be treated as retrospective results rather than an untouched independent evaluation.
- Predictions are experimental and should not be treated as confirmed fraud determinations.

## Future Enhancements

- Real-time cryptocurrency transaction analysis.
- Advanced temporal drift detection.
- More robust model retraining strategies.
- Interactive transaction network visualization.
- Model monitoring and performance tracking.

## Conclusion

This project demonstrates how Big Data processing and Machine Learning can be combined to analyse cryptocurrency transactions and identify potentially illicit activity.

It also highlights the importance of chronological validation, model evaluation, and temporal distribution shift analysis when developing transaction classification systems.

## Author

**Apoorva Mahesh**

GitHub: [apoorvamahesh26-prog](https://github.com/apoorvamahesh26-prog)

**Project:** Cryptocurrency Transaction Analytics
