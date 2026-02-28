#!/usr/bin/env python3
"""
Complete ML-based IoT Intrusion Detection System using TON_IoT and BoT-IoT datasets.

Implements:
- Binary classification (Normal vs Attack)
- Multi-class classification (8 attack classes)
- Attack category classification (3 categories)
- Supervised model comparison (RF, SVM, KNN, DT, XGBoost, LR, ANN)
- Unsupervised anomaly detection (Isolation Forest, Autoencoder)
- Feature scaling, feature selection, SMOTE, cross-validation, GridSearchCV
- Evaluation metrics and confusion matrices
- LSTM sequence model for future attack prediction
- Model comparison graphs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError as exc:
    raise ImportError("Please install imbalanced-learn: pip install imbalanced-learn") from exc

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


RANDOM_STATE = 42

ATTACK_CLASSES = [
    "DDoS",
    "Spoofing",
    "Port Scanning",
    "Sinkhole",
    "Man-in-the-Middle",
    "Botnet",
    "Ransomware",
    "Physical Tampering",
]

ATTACK_TO_CATEGORY = {
    "DDoS": "Network Layer Attack",
    "Spoofing": "Network Layer Attack",
    "Port Scanning": "Network Layer Attack",
    "Sinkhole": "Network Layer Attack",
    "Man-in-the-Middle": "Network Layer Attack",
    "Botnet": "Malware/Application Attack",
    "Ransomware": "Malware/Application Attack",
    "Physical Tampering": "Physical Attack",
    "Normal": "Normal",
}


@dataclass
class TaskData:
    X: pd.DataFrame
    y: pd.Series
    name: str


class IoTIntrusionDetectionSystem:
    def __init__(
        self,
        ton_iot_path: Path,
        bot_iot_path: Path,
        output_dir: Path,
        label_col: Optional[str] = None,
        attack_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        k_features: int = 25,
    ) -> None:
        self.ton_iot_path = ton_iot_path
        self.bot_iot_path = bot_iot_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_col = label_col
        self.attack_col = attack_col
        self.timestamp_col = timestamp_col
        self.k_features = k_features

        self.best_models: Dict[str, object] = {}
        self.model_results: Dict[str, Dict[str, float]] = {}

    def load_and_merge_datasets(self) -> pd.DataFrame:
        ton_df = pd.read_csv(self.ton_iot_path)
        bot_df = pd.read_csv(self.bot_iot_path)

        common_cols = sorted(set(ton_df.columns).intersection(bot_df.columns))
        if len(common_cols) < 3:
            raise ValueError("TON_IoT and BoT-IoT datasets have insufficient overlapping columns.")

        merged = pd.concat([ton_df[common_cols], bot_df[common_cols]], ignore_index=True)
        merged.columns = [c.strip() for c in merged.columns]

        # infer label/attack columns if not provided
        self.label_col = self.label_col or self._find_col(merged, ["label", "attack", "is_attack", "target"])
        self.attack_col = self.attack_col or self._find_col(merged, ["attack_type", "attack", "type", "class"])

        if self.label_col is None:
            raise ValueError("Could not infer binary label column. Please pass --label-col.")
        if self.attack_col is None:
            raise ValueError("Could not infer attack class column. Please pass --attack-col.")

        if self.timestamp_col is None:
            self.timestamp_col = self._find_col(merged, ["timestamp", "time", "ts", "datetime"])

        return merged

    @staticmethod
    def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    @staticmethod
    def normalize_attack_name(name: str) -> str:
        n = str(name).strip().lower()
        mapping = {
            "ddos": "DDoS",
            "spoofing": "Spoofing",
            "portscan": "Port Scanning",
            "port scanning": "Port Scanning",
            "scan": "Port Scanning",
            "sinkhole": "Sinkhole",
            "mitm": "Man-in-the-Middle",
            "man-in-the-middle": "Man-in-the-Middle",
            "man in the middle": "Man-in-the-Middle",
            "botnet": "Botnet",
            "ransomware": "Ransomware",
            "physical tampering": "Physical Tampering",
            "tampering": "Physical Tampering",
            "normal": "Normal",
            "benign": "Normal",
        }
        return mapping.get(n, str(name))

    def prepare_tasks(self, df: pd.DataFrame) -> Dict[str, TaskData]:
        data = df.copy()
        data[self.attack_col] = data[self.attack_col].apply(self.normalize_attack_name)

        # binary label creation
        if data[self.label_col].dtype == object:
            data["binary_label"] = data[self.label_col].astype(str).str.lower().map(
                {"normal": 0, "benign": 0, "0": 0, "attack": 1, "malicious": 1, "1": 1}
            )
            data["binary_label"] = data["binary_label"].fillna((data[self.attack_col] != "Normal").astype(int))
        else:
            data["binary_label"] = (data[self.label_col].astype(float) > 0).astype(int)

        # multiclass and categories
        data = data[data[self.attack_col].isin(ATTACK_CLASSES + ["Normal"])].copy()
        data["attack_category"] = data[self.attack_col].map(ATTACK_TO_CATEGORY).fillna("Malware/Application Attack")

        feature_drop = {self.label_col, self.attack_col, "binary_label", "attack_category"}
        if self.timestamp_col:
            feature_drop.add(self.timestamp_col)

        X = data[[c for c in data.columns if c not in feature_drop]]

        return {
            "binary": TaskData(X=X, y=data["binary_label"], name="binary"),
            "multiclass": TaskData(X=X, y=data[self.attack_col], name="multiclass"),
            "category": TaskData(X=X, y=data["attack_category"], name="category"),
        }

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, num_cols),
                ("cat", categorical_pipeline, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return preprocessor

    def supervised_model_search(self, task: TaskData) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            task.X,
            task.y,
            test_size=0.2,
            stratify=task.y,
            random_state=RANDOM_STATE,
        )

        preprocessor = self.build_preprocessor(task.X)
        is_multiclass = len(pd.Series(task.y).unique()) > 2

        models = {
            "RandomForest": (
                RandomForestClassifier(random_state=RANDOM_STATE),
                {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 20]},
            ),
            "SVM": (
                SVC(probability=True, random_state=RANDOM_STATE),
                {"clf__C": [1, 10], "clf__kernel": ["rbf", "linear"]},
            ),
            "KNN": (
                KNeighborsClassifier(),
                {"clf__n_neighbors": [5, 11], "clf__weights": ["uniform", "distance"]},
            ),
            "DecisionTree": (
                DecisionTreeClassifier(random_state=RANDOM_STATE),
                {"clf__max_depth": [None, 15, 30], "clf__min_samples_split": [2, 10]},
            ),
            "LogisticRegression": (
                LogisticRegression(max_iter=400, random_state=RANDOM_STATE),
                {"clf__C": [0.1, 1, 10], "clf__solver": ["lbfgs"]},
            ),
        }

        if XGBClassifier is not None:
            models["XGBoost"] = (
                XGBClassifier(
                    eval_metric="mlogloss" if is_multiclass else "logloss",
                    random_state=RANDOM_STATE,
                    n_estimators=120,
                ),
                {"clf__max_depth": [4, 6], "clf__learning_rate": [0.05, 0.1]},
            )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = "f1_macro" if is_multiclass else "f1"

        task_results = {}

        for model_name, (model, grid) in models.items():
            pipe = ImbPipeline(
                steps=[
                    ("prep", preprocessor),
                    ("selector", SelectKBest(mutual_info_classif, k=min(self.k_features, task.X.shape[1]))),
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("clf", model),
                ]
            )
            search = GridSearchCV(pipe, param_grid=grid, cv=cv, scoring=scoring, n_jobs=-1)
            search.fit(X_train, y_train)

            y_pred = search.predict(X_test)
            y_prob = self._predict_proba_safe(search, X_test)
            metrics = self.compute_metrics(y_test, y_pred, y_prob)
            task_results[model_name] = metrics

            self._save_confusion_matrix(y_test, y_pred, f"{task.name}_{model_name}_cm.png")
            self.best_models[f"{task.name}_{model_name}"] = search.best_estimator_

        ann_metrics = self._train_ann(task, X_train, y_train, X_test, y_test)
        task_results["ANN"] = ann_metrics

        self.model_results[task.name] = task_results
        self._plot_model_comparison(task.name, task_results)

    @staticmethod
    def _predict_proba_safe(model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None

    def compute_metrics(self, y_true, y_pred, y_prob=None) -> Dict[str, float]:
        is_multiclass = len(np.unique(y_true)) > 2
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        try:
            if y_prob is not None:
                if is_multiclass:
                    classes = np.unique(y_true)
                    y_true_bin = label_binarize(y_true, classes=classes)
                    results["roc_auc"] = roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="weighted")
                else:
                    positive_scores = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
                    results["roc_auc"] = roc_auc_score(y_true, positive_scores)
            else:
                results["roc_auc"] = np.nan
        except Exception:
            results["roc_auc"] = np.nan
        return results

    def _save_confusion_matrix(self, y_true, y_pred, filename: str) -> None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(7, 5))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(filename.replace("_", " "))
        fig.tight_layout()
        fig.savefig(self.output_dir / filename)
        plt.close(fig)

    def _train_ann(self, task: TaskData, X_train, y_train, X_test, y_test) -> Dict[str, float]:
        prep = self.build_preprocessor(task.X)
        X_train_p = prep.fit_transform(X_train)
        X_test_p = prep.transform(X_test)

        # manual feature selection
        selector = SelectKBest(mutual_info_classif, k=min(self.k_features, X_train_p.shape[1]))
        X_train_s = selector.fit_transform(X_train_p, y_train)
        X_test_s = selector.transform(X_test_p)

        # handle imbalance by class weights
        y_series = pd.Series(y_train)
        class_counts = y_series.value_counts().to_dict()
        total = len(y_series)
        class_weights = {cls: total / (len(class_counts) * count) for cls, count in class_counts.items()}

        is_multiclass = len(np.unique(y_train)) > 2

        if is_multiclass:
            encoder = LabelEncoder()
            y_train_enc = encoder.fit_transform(y_train)
            y_test_enc = encoder.transform(y_test)
            y_train_cat = to_categorical(y_train_enc)
            output_units = y_train_cat.shape[1]
            output_activation = "softmax"
            loss = "categorical_crossentropy"
        else:
            y_train_enc = np.asarray(y_train).astype(int)
            y_test_enc = np.asarray(y_test).astype(int)
            y_train_cat = y_train_enc
            output_units = 1
            output_activation = "sigmoid"
            loss = "binary_crossentropy"

        model = Sequential([
            Input(shape=(X_train_s.shape[1],)),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(output_units, activation=output_activation),
        ])
        model.compile(optimizer=Adam(1e-3), loss=loss, metrics=["accuracy"])

        model.fit(
            X_train_s,
            y_train_cat,
            validation_split=0.2,
            epochs=20,
            batch_size=256,
            verbose=0,
            class_weight=class_weights,
        )

        probs = model.predict(X_test_s, verbose=0)
        if is_multiclass:
            y_pred_idx = np.argmax(probs, axis=1)
            y_pred = encoder.inverse_transform(y_pred_idx)
            y_prob = probs
            y_true_eval = y_test
        else:
            y_pred = (probs.ravel() >= 0.5).astype(int)
            y_prob = np.column_stack([1 - probs.ravel(), probs.ravel()])
            y_true_eval = y_test_enc

        metrics = self.compute_metrics(y_true_eval, y_pred, y_prob)
        self._save_confusion_matrix(y_true_eval, y_pred, f"{task.name}_ANN_cm.png")
        return metrics

    def unsupervised_anomaly_detection(self, task: TaskData) -> Dict[str, Dict[str, float]]:
        X_train, X_test, y_train, y_test = train_test_split(
            task.X,
            task.y,
            test_size=0.2,
            stratify=task.y,
            random_state=RANDOM_STATE,
        )

        prep = self.build_preprocessor(task.X)
        X_train_p = prep.fit_transform(X_train)
        X_test_p = prep.transform(X_test)

        y_train_bin = (pd.Series(y_train).astype(str) != "0").astype(int)
        y_test_bin = (pd.Series(y_test).astype(str) != "0").astype(int)
        if task.name != "binary":
            y_train_bin = (pd.Series(y_train).astype(str).str.lower() != "normal").astype(int)
            y_test_bin = (pd.Series(y_test).astype(str).str.lower() != "normal").astype(int)

        # Isolation Forest
        iso = IsolationForest(contamination=0.15, random_state=RANDOM_STATE)
        iso.fit(X_train_p)
        iso_pred = (iso.predict(X_test_p) == -1).astype(int)

        # Autoencoder
        ae_input_dim = X_train_p.shape[1]
        encoder = Sequential([
            Input(shape=(ae_input_dim,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
        ])
        decoder = Sequential([
            Input(shape=(16,)),
            Dense(32, activation="relu"),
            Dense(64, activation="relu"),
            Dense(ae_input_dim, activation="linear"),
        ])
        inp = Input(shape=(ae_input_dim,))
        encoded = encoder(inp)
        decoded = decoder(encoded)
        autoencoder = Model(inp, decoded)
        autoencoder.compile(optimizer="adam", loss="mse")

        normal_idx = np.where(y_train_bin.values == 0)[0]
        X_train_normal = X_train_p[normal_idx]

        autoencoder.fit(X_train_normal, X_train_normal, epochs=20, batch_size=256, verbose=0, validation_split=0.1)
        recon = autoencoder.predict(X_test_p, verbose=0)
        mse = np.mean(np.square(X_test_p - recon), axis=1)
        threshold = np.percentile(mse, 85)
        ae_pred = (mse > threshold).astype(int)

        results = {
            "IsolationForest": self.compute_metrics(y_test_bin, iso_pred),
            "Autoencoder": self.compute_metrics(y_test_bin, ae_pred),
        }

        self._save_confusion_matrix(y_test_bin, iso_pred, f"{task.name}_IsolationForest_cm.png")
        self._save_confusion_matrix(y_test_bin, ae_pred, f"{task.name}_Autoencoder_cm.png")
        return results

    def lstm_attack_prediction(self, task: TaskData, sequence_length: int = 20) -> Dict[str, float]:
        data = task.X.copy()
        y = task.y.copy()

        if self.timestamp_col and self.timestamp_col in data.columns:
            data = data.sort_values(self.timestamp_col)
            y = y.loc[data.index]

        prep = self.build_preprocessor(data)
        X_proc = prep.fit_transform(data)

        y_bin = (pd.Series(y).astype(str).str.lower() != "normal").astype(int).values
        if task.name == "binary":
            y_bin = np.asarray(y).astype(int)

        sequences, labels = [], []
        for i in range(len(X_proc) - sequence_length):
            sequences.append(X_proc[i : i + sequence_length])
            labels.append(y_bin[i + sequence_length])

        X_seq = np.asarray(sequences)
        y_seq = np.asarray(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X_seq,
            y_seq,
            test_size=0.2,
            stratify=y_seq,
            random_state=RANDOM_STATE,
        )

        model = Sequential([
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model.fit(X_train, y_train, epochs=15, batch_size=128, validation_split=0.2, verbose=0)

        y_prob = model.predict(X_test, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = self.compute_metrics(y_test, y_pred, np.column_stack([1 - y_prob, y_prob]))
        self._save_confusion_matrix(y_test, y_pred, "lstm_attack_prediction_cm.png")
        return metrics

    def _plot_model_comparison(self, task_name: str, task_results: Dict[str, Dict[str, float]]) -> None:
        df = pd.DataFrame(task_results).T
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        fig, ax = plt.subplots(figsize=(12, 6))
        df[metrics].plot(kind="bar", ax=ax)
        ax.set_title(f"Model Comparison - {task_name}")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"{task_name}_model_comparison.png")
        plt.close(fig)

    def save_reports(self, unsup_results: Dict[str, Dict[str, float]], lstm_metrics: Dict[str, float]) -> None:
        report = {
            "supervised": self.model_results,
            "unsupervised": unsup_results,
            "lstm": lstm_metrics,
        }
        with open(self.output_dir / "metrics_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        with open(self.output_dir / "classification_reports.txt", "w", encoding="utf-8") as f:
            for task_name, results in self.model_results.items():
                f.write(f"\n=== {task_name.upper()} ===\n")
                for model_name, m in results.items():
                    f.write(f"{model_name}: {m}\n")
            f.write("\n=== UNSUPERVISED ===\n")
            for model_name, m in unsup_results.items():
                f.write(f"{model_name}: {m}\n")
            f.write("\n=== LSTM ===\n")
            f.write(str(lstm_metrics))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IoT IDS using TON_IoT and BoT-IoT datasets")
    parser.add_argument("--ton-iot", required=True, type=Path, help="Path to TON_IoT CSV")
    parser.add_argument("--bot-iot", required=True, type=Path, help="Path to BoT-IoT CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--label-col", type=str, default=None, help="Label column name")
    parser.add_argument("--attack-col", type=str, default=None, help="Attack class column name")
    parser.add_argument("--timestamp-col", type=str, default=None, help="Timestamp column name")
    parser.add_argument("--k-features", type=int, default=25, help="Number of features for SelectKBest")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    system = IoTIntrusionDetectionSystem(
        ton_iot_path=args.ton_iot,
        bot_iot_path=args.bot_iot,
        output_dir=args.output_dir,
        label_col=args.label_col,
        attack_col=args.attack_col,
        timestamp_col=args.timestamp_col,
        k_features=args.k_features,
    )

    data = system.load_and_merge_datasets()
    tasks = system.prepare_tasks(data)

    for task in tasks.values():
        system.supervised_model_search(task)

    unsup_results = system.unsupervised_anomaly_detection(tasks["binary"])
    lstm_metrics = system.lstm_attack_prediction(tasks["binary"])

    system.save_reports(unsup_results, lstm_metrics)
    print(f"Completed. Outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
