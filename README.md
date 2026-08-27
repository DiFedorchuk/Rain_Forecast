# 🌦️ Australian Rain Forecast

### Predicting Next-Day Rainfall with Machine Learning

An end-to-end Machine Learning classification project that predicts whether it will rain tomorrow in Australia based on current weather observations.

The project covers the complete ML lifecycle, including exploratory data analysis, data preprocessing, model training, validation, evaluation, and deployment through an interactive Streamlit application.

## 🔗 Live Application

Try the deployed model and generate a rainfall prediction directly in your browser:

### [Launch Australian Rain Forecast App](https://austlianrainforecast.streamlit.app/)

---

## 🎯 What This Project Demonstrates

This project showcases an end-to-end Machine Learning workflow:

- Exploratory Data Analysis
- Data cleaning and preprocessing
- Feature engineering
- Binary classification modeling
- Reusable preprocessing and inference pipelines
- Model validation across multiple dataset splits
- Evaluation using Accuracy, F1 Score, and ROC AUC
- Production-oriented Python project structure
- Interactive deployment with Streamlit

The final solution allows users to enter weather observations and receive a next-day rainfall prediction through a web interface.

---

## 🔍 Machine Learning Workflow

### 1. Exploratory Data Analysis

Exploratory analysis was performed to understand the dataset, identify data quality issues, examine feature distributions, and explore relationships between weather observations and next-day rainfall.

The analysis included:

- Dataset structure inspection
- Target distribution analysis
- Missing-value analysis
- Numerical feature distributions
- Categorical feature exploration
- Feature relationship analysis
- Identification of potentially useful predictive signals

The full analysis and experimentation process is available in `EDA.ipynb`.

### 2. Data Preprocessing

The raw weather data was transformed into a format suitable for Machine Learning.

The preprocessing workflow includes:

- Handling missing values
- Processing numerical features
- Encoding categorical variables
- Scaling features where required
- Applying consistent transformations across training and inference
- Preparing data for binary classification

Preprocessing logic is separated into reusable Python modules to ensure consistency between model training and application inference.

### 3. Model Training

A Logistic Regression classifier was trained to predict the binary target:

- `Yes`: Rain is expected tomorrow
- `No`: Rain is not expected tomorrow

Logistic Regression provides a strong and interpretable baseline for binary classification while also producing probability-based predictions.

### 4. Model Validation

The prepared dataset was divided into training, validation, and test sets.

This approach makes it possible to:

- Train the model on the training dataset
- Compare model behavior on unseen validation data
- Evaluate final performance on the independent test dataset
- Identify potential overfitting
- Verify model stability across different data splits

### 5. Model Deployment

The trained model and preprocessing pipeline were saved as a reusable artifact and integrated into a Streamlit application.

The deployed application:

- Collects weather observations through an interactive interface
- Applies the same preprocessing steps used during training
- Generates a next-day rainfall prediction
- Uses the saved model artifact without requiring retraining
- Provides accessible browser-based model inference

---

## 📊 Model Performance

The saved Logistic Regression model was evaluated on the training, validation, and test splits prepared by the project pipeline.

| Dataset | Rows | Accuracy | F1 Score, Rain = Yes | ROC AUC |
|:---|---:|---:|---:|---:|
| Train | 84,471 | 0.8500 | 0.5993 | 0.8741 |
| Validation | 28,158 | 0.8503 | 0.5985 | 0.8718 |
| Test | 28,158 | 0.8518 | 0.6083 | 0.8718 |

### Performance Interpretation

- The model achieves approximately **85% accuracy** across all three dataset splits.
- The test F1 score for the positive rainfall class reaches **0.6083**.
- The test ROC AUC of **0.8718** indicates strong discrimination between rain and no-rain observations.
- Similar validation and test results indicate stable performance on unseen data.
- The small differences between training, validation, and test metrics suggest that the model does not show significant overfitting.

Because the target is a binary classification problem, Accuracy is considered together with F1 Score and ROC AUC rather than being interpreted as the only measure of model quality.

---

## 📈 ROC AUC Analysis

The Receiver Operating Characteristic curve illustrates the model's ability to distinguish between positive and negative classes across different classification thresholds.

<p align="center">
  image/roc_auc_curve.png
</p>

The dashed diagonal line represents the expected performance of a random classifier. Curves positioned closer to the upper-left corner indicate better classification performance.

### ROC AUC by Dataset

| Dataset | ROC AUC |
|:---|---:|
| Train | 0.874 |
| Validation | 0.872 |
| Test | 0.872 |

The nearly identical ROC AUC values across the three dataset splits indicate that the model maintains consistent discrimination performance on unseen data.

---

## 🛠️ Technology Stack

### Data Processing and Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Data Visualization

- Matplotlib
- Seaborn

### Application and Deployment

- Streamlit
- Streamlit Community Cloud

### Development Environment

- Jupyter Notebook
- Git
- GitHub

---

## 📂 Project Structure

```text
.
├── .streamlit/
│   └── config.toml
├── image/
│   └── roc_curve.png
├── model/
│   └── aussie_rain.joblib
├── rain_predictor/
│   ├── data loading
│   ├── preprocessing
│   ├── modeling
│   └── UI helpers
├── app.py
├── train.py
├── EDA.ipynb
├── requirements.txt
├── README.md
└── .gitignore
