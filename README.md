# Mtech-Project-file

## IoT Intrusion Detection System

This repository contains a complete machine learning pipeline for intrusion detection in IoT networks using the **TON_IoT** and **BoT-IoT** datasets.

Implemented capabilities:

- Binary classification: **Normal vs Attack**
- Multi-class classification for 8 attack classes:
  - DDoS
  - Spoofing
  - Port Scanning
  - Sinkhole
  - Man-in-the-Middle
  - Botnet
  - Ransomware
  - Physical Tampering
- Attack categorization into:
  - Physical Attack
  - Network Layer Attack
  - Malware/Application Attack
- Supervised model comparison:
  - Random Forest
  - SVM
  - KNN
  - Decision Tree
  - XGBoost
  - Logistic Regression
  - Artificial Neural Network
- Unsupervised anomaly detection:
  - Isolation Forest
  - Autoencoder
- LSTM-based sequential attack prediction
- Feature scaling, feature selection, SMOTE, cross-validation, GridSearchCV
- Metrics + confusion matrices + model comparison graphs

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn tensorflow xgboost
```

## Usage

```bash
python iot_intrusion_detection_system.py \
  --ton-iot /path/to/ton_iot.csv \
  --bot-iot /path/to/bot_iot.csv \
  --output-dir outputs \
  --label-col label \
  --attack-col attack_type \
  --timestamp-col timestamp
```

Notes:
- If column names differ, pass the correct values using `--label-col`, `--attack-col`, and `--timestamp-col`.
- The script attempts automatic column inference if these are not provided.

## Output artifacts

The script generates:

- `metrics_report.json`
- `classification_reports.txt`
- Confusion matrix plots for each model
- Model comparison charts for each task
- LSTM confusion matrix plot
