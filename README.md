# Spatial-Temporal Traffic Accident Risk Assessment in Chicago

## Overview

This project estimates the relative risk of traffic accidents on road segments in Chicago using traffic conditions, temporal patterns, and spatial information.

The objective is to rank road segment-time combinations by accident risk and identify traffic conditions associated with elevated accident likelihood. The final solution includes:

* Data collection and preprocessing
* Spatial and temporal feature engineering
* Machine Learning model training and evaluation
* Interactive risk assessment interface built with Streamlit
* Road segment visualization using Folium maps

The project covers the period from **May 2025 to April 2026**.

---

## Dataset Description

### Traffic Accident Dataset

Historical traffic crash records for Chicago:

| Metric  | Value                 |
| ------- | --------------------- |
| Records | 110,393               |
| Period  | May 2025 – April 2026 |

### Traffic Dataset

Hourly traffic measurements collected on monitored road segments:

| Metric        | Value      |
| ------------- | ---------- |
| Records       | 32,355,160 |
| Road Segments | 1,046      |

### Final Aggregated Dataset

After spatial matching, temporal aggregation, feature engineering, and creation of a complete road segment-time grid:

| Metric          | Value                        |
| --------------- | ---------------------------- |
| Records         | 2,108,736                    |
| Target Variable | Accident occurrence (binary) |

Each row represents a unique combination of:

* Road Segment
* Hour
* Day of Week
* Month

with associated traffic characteristics and accident label.

---

## Project Structure

```text
project/
│
├── .streamlit/
│
├── app/
│   ├── streamlit_app.py
│   └── predictor.py
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   └── catboost_model.cbm
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   ├── utils/
│   └── visualization/
│
├── requirements.txt
└── README.md
```

---

## Feature Engineering

### Traffic Features

* Average Speed
* Free Flow Speed
* Congestion Level

### Spatial Features

* Segment Latitude
* Segment Longitude
* Segment ID

### Temporal Features

Cyclic encoding:

* Hour (sin/cos)
* Day of Week (sin/cos)
* Month (sin/cos)

Additional categorical variables:

* Hour
* Day of Week
* Month

---

## Machine Learning Models

The following models were trained and evaluated:

### Logistic Regression

Two configurations:

1. Numeric features only
2. Numeric features + Segment ID

---

### Multi-Layer Perceptron (MLP)

Two configurations:

1. Numeric features only
2. Numeric features + Segment ID

Oversampling was applied using RandomOverSampler to address class imbalance.

---

### CatBoost

CatBoost was selected as the final production model because it achieved the best overall performance and naturally handles categorical features.

Categorical features:

* segment_id
* hour
* day_of_week
* month

---

## Model Evaluation

The following metrics were used:

* ROC-AUC
* PR-AUC
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Threshold optimization was performed to maximize F1-score and improve classification performance on the highly imbalanced accident dataset.

---

## Visualization

The project includes:

* Folium Map of tracked road segments and crashes before data preprocessing
* Folium Map of tracked road segments and crashes after data processing
* Temporal analysis of accidents
* ROC Curves
* Feature Importance Analysis
* Confusion Matrices
* Interactive Folium Maps of top 100 roads with the highest risk

---

## Interactive User Interface

An interactive Streamlit application allows users to:

* Select a road segment
* Choose hour, day of week, and month
* Visualize historical traffic conditions
* Calculate accident risk score
* Visualize the selected road segment on a map
* Display accident risk level (Low / Medium / High)

The displayed score represents a model-based accident risk score rather than a calibrated probability of an accident. The score is used to compare relative risk levels across different road segments and traffic conditions.
Road segments are visualized using Folium and GeoSpatial data.

---

## Technologies and Libraries

### Core Data Science

* NumPy
* Pandas
* Scikit-Learn
* Imbalanced-Learn

### Geospatial Analysis

* GeoPandas
* Shapely
* Folium
* Branca

### Machine Learning

* CatBoost

### Visualization

* Matplotlib
* Seaborn

### User Interface

* Streamlit
* Streamlit-Folium

---

## Requirements

```bash
numpy~=2.4.4
pandas~=3.0.2
scikit-learn~=1.8.0
imbalanced-learn~=0.14.2
geopandas~=1.1.3
folium~=0.20.0

shapely~=2.1.2
branca~=0.8.2
catboost~=1.2.10
streamlit~=1.58.0
streamlit-folium~=0.27.2
seaborn~=0.13.2
matplotlib~=3.10.9
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Yuliya-Ptlsk/DS-ML-Traffic-Accident-Risk-Assessment-in-Chicago.git
cd DS-ML-Traffic-Accident-Risk-Assessment-in-Chicago
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Best Model if there is no catboost_model.cbm

Run training pipeline located in:

```text
src/pipelines
```

Example:

```bash
python src/pipelines/train_catboost.py
```

The trained model will be saved to:

```text
models/catboost_model.cbm
```

### 4. Launch Interactive Application

From the project root directory:

```bash
streamlit run app/streamlit_app.py
```

The application will open automatically in your browser.

---

## Future Improvements

Potential extensions include:

* Real-time traffic integration
* Weather data integration
* Multi-class accident severity prediction
* Deployment to cloud infrastructure
* Automated retraining pipeline
* REST API for external applications

---

## Author

Traffic Accident Risk Assessment Project

Data Science / Machine Learning Final Project