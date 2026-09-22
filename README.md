# 🛡️ ClaimRisk AI — Vehicle Insurance Fraud Detection System

![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue)
![Framework](https://img.shields.io/badge/Streamlit-1.30+-red)
![ML Library](https://img.shields.io/badge/Scikit--Learn-1.3+-orange)
![Visualization](https://img.shields.io/badge/Plotly-5.18+-teal)
![License](https://img.shields.io/badge/License-Academic-green)

An enterprise-ready **InsurTech Machine Learning Platform** developed for real-time fraud detection, claim risk scoring, batch file auditing, and multi-model benchmark evaluation.

---

## 🌟 Key Capabilities & Features

### 1. 🔮 Live Fraud Prediction & Batch Audit Engine
- **Single Claim Scoring**: Live probability risk index (0–100%), color-coded risk tiers (*Low Risk*, *Moderate Audit*, *Critical Fraud Alert*), multi-model consensus, multi-factor radar profile, and SIU Action Checklist.
- **Batch CSV Audit Engine**: Upload any claims dataset to score hundreds of claims simultaneously, categorize risk tiers, and download audited reports.
- **Sensitivity Threshold Tuning**: Interactively adjust decision thresholds (20%–80%) to balance Precision vs Recall according to risk tolerance.

### 2. 📁 Dataset Explorer
- Full inspection of the 11,988-record insurance claim dataset.
- Real-time search, multi-column filters, descriptive 5-number statistics, missing value audit, data dictionary, and CSV export.

### 3. 📊 Exploratory Data Analysis (EDA)
- Light-themed interactive Plotly visualizations for fraud class balance, driver demographics (Age, Income), claim financials (Vehicle Price vs Claim Amount), and Pearson correlation heatmaps.

### 4. ⚖️ 5 Supervised ML Algorithms Benchmark
- Benchmarks **Logistic Regression**, **Decision Tree**, **KNN**, **AdaBoost**, and **Random Forest**.
- Filter/toggle models to compare head-to-head metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Specificity, 5-Fold Cross Validation Accuracy, ROC Curves, PR Curves, Confusion Matrices, and Top 15 Feature Importances.

### 5. 🧩 Unsupervised Clustering & Anomaly Profiling
- **K-Means** & **K-Medoids (PAM)** cluster segmentation.
- 2D & 3D PCA cluster scatter projections, Elbow method inertia curve, Silhouette scores (\(k=2..6\)), and fraud rate risk profiling per cluster.

---

## 📁 Repository Structure

```
FrontEnd_python/
│
├── app.py                      # Main Streamlit Web Application
├── train_models.py             # Model training & evaluation pipeline
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Containerization specification
├── Procfile                    # Render/Heroku process configuration
├── vehicle_insurance.csv       # Standardized cleaned dataset
├── vehicle_insurance_raw.csv   # Raw dataset with natural units
│
├── .streamlit/
│   └── config.toml             # Custom Light Blue Theme & server settings
│
└── model/
    ├── logistic_model.pkl      # Logistic Regression Model
    ├── decision_tree.pkl       # Decision Tree Classifier
    ├── knn_model.pkl           # K-Nearest Neighbors Classifier
    ├── adaboost_model.pkl      # AdaBoost Classifier
    ├── random_forest.pkl       # Random Forest Classifier
    ├── kmeans_model.pkl        # K-Means Clustering Model
    ├── kmedoids_model.pkl      # K-Medoids (PAM) Clustering Model
    ├── evaluation_metrics.pkl  # Precomputed benchmark metrics & curves
    └── clustering_summary.pkl  # PCA coordinates & cluster risk profiles
```

---

## 🚀 Quick Start Guide

### 1. Local Setup
```bash
# Clone or navigate to the repository
cd FrontEnd_python

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain all models
python train_models.py

# Launch the Streamlit Web Application
streamlit run app.py
```
👉 Open your browser at **`http://localhost:8501`**

### 2. Docker Setup
```bash
# Build Docker image
docker build -t vehicle-insurance-fraud .

# Run Docker container
docker run -p 8501:8501 vehicle-insurance-fraud
```

---

## 👥 Academic Credits
- **Project**: Vehicle Insurance Fraud Detection System
- **Semester**: Sem-5 Machine Learning Lab Project
- **Tech Stack**: Streamlit, Scikit-Learn, Plotly, Pandas, NumPy, Joblib
