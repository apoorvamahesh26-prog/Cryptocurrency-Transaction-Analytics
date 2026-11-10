
# Cryptocurrency Transaction Analytics

## Project Overview

Cryptocurrency Transaction Analytics is a Big Data Analytics and Machine Learning project that analyses cryptocurrency transactions and identifies potentially illicit transactions.

The project uses the Elliptic Bitcoin Transaction Dataset and combines PySpark, Python, Machine Learning, and Streamlit to perform data processing, exploratory analysis, classification, and interactive visualization.

## Objectives

- Analyse large-scale cryptocurrency transaction data.
- Perform data quality checking and preprocessing.
- Explore transaction patterns across time steps.
- Identify potentially illicit transactions using Machine Learning.
- Compare Random Forest and XGBoost models.
- Analyse feature importance and temporal distribution shifts.
- Present results through an interactive Streamlit dashboard.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PySpark | Big Data processing |
| Pandas | Data manipulation |
| PyArrow | Parquet file handling |
| Scikit-learn | Random Forest and evaluation |
| XGBoost | Gradient boosting classification |
| Matplotlib | Data visualization |
| Streamlit | Interactive dashboard |
| Joblib | Model saving and loading |

## Dataset

Dataset: Elliptic Bitcoin Transaction Dataset.

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
| Total | 203,769 |

Unknown transactions are not used as labelled training examples.

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

### Random Forest

- Number of estimators: 200
- Class weighting: balanced_subsample
- Random state: 42

### XGBoost

- Number of estimators: 300
- Maximum depth: 7
- Learning rate: 0.05
- Subsample: 0.8
- Column sampling: 0.8

## Final Model Comparison

Evaluation on the chronological test set (time steps 35–49):

| Metric | Random Forest | XGBoost |
|---|---:|---:|
| Accuracy | 98.04% | 97.53% |
| Precision | 97.49% | 86.36% |
| Recall | 71.65% | 73.68% |
| F1-score | 82.60% | 79.52% |
| ROC-AUC | 93.42% | 93.33% |
| PR-AUC | 79.68% | 80.06% |

Random Forest achieved a higher F1-score and precision on the final test set. XGBoost detected more illicit transactions.

## Dashboard Features

The Streamlit dashboard includes:

1. Dashboard Overview
2. Transaction Analysis
3. ML Prediction
4. Model Comparison
5. Feature Importance

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Open the project folder:

```bash
cd Cryptocurrency_Transaction_Analytics
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## Important Limitations

- The dataset contains a large proportion of unknown transactions.
- Model evaluation uses a chronological split to simulate future transaction prediction.
- Significant temporal distribution shifts were observed, particularly in later test time steps.
- The final test results were inspected during model development and should be treated as retrospective results, not an untouched independent evaluation.
- Predictions are experimental and should not be treated as confirmed fraud determinations.

## Future Enhancements

- Real-time cryptocurrency transaction analysis.
- Advanced temporal drift detection.
- More robust model retraining strategies.
- Interactive transaction network visualization.
- Model monitoring and performance tracking.

## Conclusion

This project demonstrates how Big Data processing and Machine Learning can be combined to analyse cryptocurrency transactions and identify potentially illicit activity. It also highlights the importance of temporal validation and distribution-shift analysis when building transaction classification systems.
