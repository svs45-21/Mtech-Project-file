"""Complete ML-based Intrusion Detection System for IoT networks.

This script supports:
- Binary classification (Normal vs Attack)
- Multi-class classification (8 attack classes)
- Attack category classification (3 categories)
- Supervised model comparison with CV + GridSearchCV
- Unsupervised anomaly detection (Isolation Forest + Autoencoder)
- LSTM-based future attack prediction from sequential traffic
- Metrics, confusion matrices, ROC curves, and comparison graphs

Dependencies:
    pandas, numpy, scikit-learn, imbalanced-learn, xgboost,
    matplotlib, seaborn, tensorflow
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from xgboost import XGBClassifier


RANDOM_STATE = 42


ATTACK_CLASS_MAP = {
    # DDoS
    "ddos": "DDoS",
    "dos": "DDoS",
    "udp flood": "DDoS",
    "tcp flood": "DDoS",
    # Spoofing
    "spoofing": "Spoofing",
    "arp spoof": "Spoofing",
    "ip spoof": "Spoofing",
    # Port Scanning
    "scan": "Port Scanning",
    "port scan": "Port Scanning",
    "service scan": "Port Scanning",
    # Sinkhole
    "sinkhole": "Sinkhole",
    # MITM
    "mitm": "Man-in-the-Middle",
    "man in the middle": "Man-in-the-Middle",
    # Botnet
    "botnet": "Botnet",
    "bot": "Botnet",
    # Ransomware
    "ransomware": "Ransomware",
    # Physical Tampering
    "physical": "Physical Tampering",
    "tampering": "Physical Tampering",
}

CATEGORY_MAP = {
    "Physical Tampering": "Physical Attack",
    "DDoS": "Network Layer Attack",
    "Spoofing": "Network Layer Attack",
    "Port Scanning": "Network Layer Attack",
    "Sinkhole": "Network Layer Attack",
    "Man-in-the-Middle": "Network Layer Attack",
    "Botnet": "Malware/Application Attack",
    "Ransomware": "Malware/Application Attack",
}

TARGET_ATTACK_CLASSES = [
    "DDoS",
    "Spoofing",
    "Port Scanning",
    "Sinkhole",
    "Man-in-the-Middle",
    "Botnet",
    "Ransomware",
    "Physical Tampering",
]


@dataclass
class DatasetConfig:
    file_path: str
    label_column: str


def normalize_attack_name(raw_label: str) -> str:
    raw = str(raw_label).strip().lower()
    if raw in {"normal", "benign", "0", "none"}:
        return "Normal"
    for key, canonical in ATTACK_CLASS_MAP.items():
        if key in raw:
            return canonical
    return "Botnet" if "malware" in raw else "DDoS"


def load_and_merge_datasets(configs: List[DatasetConfig]) -> pd.DataFrame:
    dfs = []
    for cfg in configs:
        df = pd.read_csv(cfg.file_path)
        if cfg.label_column not in df.columns:
            raise ValueError(f"Label column '{cfg.label_column}' not found in {cfg.file_path}")

        df = df.copy()
        df["raw_label"] = df[cfg.label_column]
        df["attack_class"] = df["raw_label"].map(normalize_attack_name)
        df["binary_label"] = np.where(df["attack_class"] == "Normal", 0, 1)
        df["attack_category"] = df["attack_class"].map(CATEGORY_MAP).fillna("Network Layer Attack")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates().reset_index(drop=True)
    return merged


def split_features_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    target_cols = {"raw_label", "attack_class", "binary_label", "attack_category"}
    feature_cols = [c for c in df.columns if c not in target_cols]
    X = df[feature_cols]
    y_binary = df["binary_label"]
    y_multiclass = df["attack_class"]
    y_category = df["attack_category"]
    return X, y_binary, y_multiclass, y_category


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def evaluate_predictions(y_true, y_pred, y_score=None, average="binary") -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score, multi_class="ovr" if average != "binary" else "raise")
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def train_supervised_models(
    X_train, y_train, X_test, y_test, task_name: str, output_dir: Path, is_binary: bool
) -> pd.DataFrame:
    models = {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 20]},
        ),
        "SVM": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0], "model__kernel": ["rbf", "linear"]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 11], "model__weights": ["uniform", "distance"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"model__max_depth": [None, 20], "model__min_samples_split": [2, 10]},
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=200,
                learning_rate=0.05,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            {"model__max_depth": [4, 8], "model__subsample": [0.8, 1.0]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
    }

    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, (estimator, param_grid) in models.items():
        pipe = Pipeline([
            ("selector", SelectKBest(mutual_info_classif, k=min(50, X_train.shape[1]))),
            ("model", estimator),
        ])

        gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_

        cv_score = cross_val_score(clone(best_model), X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1).mean()
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None

        avg = "binary" if is_binary else "weighted"
        score_input = y_proba[:, 1] if (is_binary and y_proba is not None) else y_proba
        metrics = evaluate_predictions(y_test, y_pred, score_input, average=avg)
        metrics["model"] = name
        metrics["cv_f1_weighted"] = cv_score
        metrics["best_params"] = gs.best_params_
        results.append(metrics)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - {task_name} - {name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"cm_{task_name}_{name.replace(' ', '_')}.png", dpi=150)
        plt.close()

    ann = train_ann_model(X_train, y_train, X_test, y_test, is_binary=is_binary)
    ann["model"] = "Artificial Neural Network"
    ann["cv_f1_weighted"] = np.nan
    ann["best_params"] = {"epochs": 25, "batch_size": 256}
    results.append(ann)

    return pd.DataFrame(results).sort_values("f1", ascending=False)


def train_ann_model(X_train, y_train, X_test, y_test, is_binary: bool) -> Dict[str, float]:
    n_features = X_train.shape[1]
    model = Sequential([
        Dense(128, activation="relu", input_shape=(n_features,)),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1 if is_binary else len(np.unique(y_train)), activation="sigmoid" if is_binary else "softmax"),
    ])

    if is_binary:
        y_train_fit, y_test_fit = y_train.values, y_test.values
        model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    else:
        enc = LabelEncoder()
        y_train_enc = enc.fit_transform(y_train)
        y_test_enc = enc.transform(y_test)
        y_train_fit = pd.get_dummies(y_train_enc).values
        y_test_fit = pd.get_dummies(y_test_enc).reindex(columns=range(len(enc.classes_)), fill_value=0).values
        model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train,
        y_train_fit,
        validation_split=0.2,
        epochs=25,
        batch_size=256,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )

    proba = model.predict(X_test, verbose=0)
    if is_binary:
        pred = (proba.ravel() > 0.5).astype(int)
        return evaluate_predictions(y_test, pred, proba.ravel(), average="binary")

    pred = np.argmax(proba, axis=1)
    y_true = np.argmax(y_test_fit, axis=1)
    return evaluate_predictions(y_true, pred, proba, average="weighted")


def run_isolation_forest(X_train, X_test, y_test_binary) -> Dict[str, float]:
    iso = IsolationForest(n_estimators=300, contamination=0.2, random_state=RANDOM_STATE)
    iso.fit(X_train)
    preds = iso.predict(X_test)
    preds = np.where(preds == -1, 1, 0)
    return evaluate_predictions(y_test_binary, preds, average="binary")


def build_autoencoder(input_dim: int) -> Model:
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu")(inp)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(input_dim, activation="linear")(x)
    autoencoder = Model(inp, out)
    autoencoder.compile(optimizer=Adam(1e-3), loss="mse")
    return autoencoder


def run_autoencoder(X_train, y_train_binary, X_test, y_test_binary) -> Dict[str, float]:
    X_train_normal = X_train[y_train_binary == 0]
    if len(X_train_normal) == 0:
        X_train_normal = X_train
    ae = build_autoencoder(X_train.shape[1])
    ae.fit(
        X_train_normal,
        X_train_normal,
        validation_split=0.2,
        epochs=30,
        batch_size=256,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )

    recon = ae.predict(X_test, verbose=0)
    mse = np.mean(np.square(X_test - recon), axis=1)
    threshold = np.percentile(mse, 80)
    preds = (mse > threshold).astype(int)
    return evaluate_predictions(y_test_binary, preds, mse, average="binary")


def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


def run_lstm_forecasting(X_train, y_train, X_test, y_test, output_dir: Path) -> Dict[str, float]:
    seq_len = 10
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)

    lstm_model = Sequential([
        LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    lstm_model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

    lstm_model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=0,
    )

    proba = lstm_model.predict(X_test_seq, verbose=0).ravel()
    pred = (proba > 0.5).astype(int)
    metrics = evaluate_predictions(y_test_seq, pred, proba, average="binary")

    fpr, tpr, _ = roc_curve(y_test_seq, proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"LSTM (AUC={metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("LSTM ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lstm_roc.png", dpi=150)
    plt.close()

    return metrics


def plot_model_comparison(results: pd.DataFrame, task_name: str, output_dir: Path) -> None:
    plt.figure(figsize=(12, 6))
    melted = results.melt(id_vars=["model"], value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"])
    sns.barplot(data=melted, x="model", y="value", hue="variable")
    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.title(f"Model Comparison - {task_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"comparison_{task_name}.png", dpi=150)
    plt.close()


def preprocess_data(X: pd.DataFrame, y_binary: pd.Series):
    preprocessor = build_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    if not isinstance(X_processed, np.ndarray):
        X_processed = X_processed.toarray()

    selector = SelectKBest(mutual_info_classif, k=min(80, X_processed.shape[1]))
    X_selected = selector.fit_transform(X_processed, y_binary)
    return X_selected


def run_pipeline(ton_iot_path: str, bot_iot_path: str, ton_label_col: str, bot_label_col: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_merge_datasets([
        DatasetConfig(ton_iot_path, ton_label_col),
        DatasetConfig(bot_iot_path, bot_label_col),
    ])

    # Keep only required 8-class + normal for the multiclass task consistency.
    df = df[df["attack_class"].isin(TARGET_ATTACK_CLASSES + ["Normal"])].reset_index(drop=True)

    X, y_binary, y_multiclass, y_category = split_features_targets(df)
    X_final = preprocess_data(X, y_binary)

    X_train, X_test, y_bin_train, y_bin_test = train_test_split(
        X_final, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
    )

    sm = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_bin_train_sm = sm.fit_resample(X_train, y_bin_train)

    binary_results = train_supervised_models(
        X_train_sm, y_bin_train_sm, X_test, y_bin_test, "binary", out, is_binary=True
    )
    binary_results.to_csv(out / "binary_model_results.csv", index=False)
    plot_model_comparison(binary_results, "binary", out)

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_final, y_multiclass, test_size=0.2, random_state=RANDOM_STATE, stratify=y_multiclass
    )
    X_train_m_sm, y_train_m_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_m, y_train_m)

    multiclass_results = train_supervised_models(
        X_train_m_sm, y_train_m_sm, X_test_m, y_test_m, "multiclass", out, is_binary=False
    )
    multiclass_results.to_csv(out / "multiclass_model_results.csv", index=False)
    plot_model_comparison(multiclass_results, "multiclass", out)

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_final, y_category, test_size=0.2, random_state=RANDOM_STATE, stratify=y_category
    )
    X_train_c_sm, y_train_c_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_c, y_train_c)

    category_results = train_supervised_models(
        X_train_c_sm, y_train_c_sm, X_test_c, y_test_c, "category", out, is_binary=False
    )
    category_results.to_csv(out / "category_model_results.csv", index=False)
    plot_model_comparison(category_results, "category", out)

    iso_metrics = run_isolation_forest(X_train_sm, X_test, y_bin_test)
    ae_metrics = run_autoencoder(X_train_sm, y_bin_train_sm.values, X_test, y_bin_test)
    unsup_df = pd.DataFrame([
        {"model": "Isolation Forest", **iso_metrics},
        {"model": "Autoencoder", **ae_metrics},
    ])
    unsup_df.to_csv(out / "unsupervised_results.csv", index=False)

    lstm_metrics = run_lstm_forecasting(X_train_sm, y_bin_train_sm.values, X_test, y_bin_test.values, out)

    summary = {
        "binary_top_model": binary_results.iloc[0].to_dict(),
        "multiclass_top_model": multiclass_results.iloc[0].to_dict(),
        "category_top_model": category_results.iloc[0].to_dict(),
        "unsupervised": unsup_df.to_dict(orient="records"),
        "lstm": lstm_metrics,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print("Pipeline complete. Results saved to:", out.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML-based IoT Intrusion Detection System")
    parser.add_argument("--ton-iot", required=True, help="Path to TON_IoT CSV")
    parser.add_argument("--bot-iot", required=True, help="Path to BoT-IoT CSV")
    parser.add_argument("--ton-label", default="label", help="Label column in TON_IoT")
    parser.add_argument("--bot-label", default="attack", help="Label column in BoT-IoT")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save all outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        ton_iot_path=args.ton_iot,
        bot_iot_path=args.bot_iot,
        ton_label_col=args.ton_label,
        bot_label_col=args.bot_label,
        output_dir=args.output_dir,
    )
