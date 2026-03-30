#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_CANDIDATES = [
    "duration",
    "id.orig_p",
    "id.resp_p",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
    "orig_ip_bytes",
    "resp_ip_bytes",
    "missed_bytes",
]

CATEGORICAL_CANDIDATES = [
    "proto",
    "service",
    "conn_state",
    "history",
]

LABEL_CANDIDATES = [
    "label",
    "Label",
    "detailed-label",
    "detailed_label",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 IoT-23 baseline pipeline")
    parser.add_argument("--data", required=True, help="Path to IoT-23 labeled flow file")
    parser.add_argument("--output-dir", default="outputs", help="Directory for generated outputs")
    parser.add_argument("--sample-size", type=int, default=50000, help="Optional row cap for faster runs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def read_iot23_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t", low_memory=False)

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    if any(line.startswith("#fields") for line in lines):
        return read_zeek_log(lines)

    return pd.read_csv(path, sep=None, engine="python", low_memory=False)


def read_zeek_log(lines: List[str]) -> pd.DataFrame:
    fields: List[str] = []
    records: List[List[str]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#fields"):
            fields = stripped.split("\t")[1:]
            if fields and "label" in fields[-1] and "detailed-label" in fields[-1]:
                expanded = re.split(r"\s{2,}", fields[-1].strip(), maxsplit=2)
                if len(expanded) == 3:
                    fields = fields[:-1] + expanded
            continue
        if stripped.startswith("#"):
            continue
        if fields:
            parts = stripped.split("\t")
            if len(parts) == len(fields) - 2 and parts:
                expanded = re.split(r"\s{2,}", parts[-1].strip(), maxsplit=2)
                if len(expanded) == 3:
                    parts = parts[:-1] + expanded
            records.append(parts)

    if not fields:
        raise ValueError("Could not detect Zeek #fields header in the dataset.")

    return pd.DataFrame(records, columns=fields)


def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({"-": np.nan, "(empty)": np.nan, "": np.nan})


def find_label_column(df: pd.DataFrame) -> str:
    for column in LABEL_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"No label column found. Expected one of: {LABEL_CANDIDATES}")


def label_to_binary(value: object) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if not text:
        return None
    if "benign" in text:
        return 0
    if any(token in text for token in ["malicious", "attack", "c&c", "cc", "botnet", "spam", "dos", "scan"]):
        return 1
    return None


def safe_ip_flag(value: object, check_private: bool) -> float:
    if pd.isna(value):
        return np.nan
    try:
        ip_obj = ipaddress.ip_address(str(value))
        return float(ip_obj.is_private if check_private else ip_obj.is_global)
    except ValueError:
        return np.nan


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    label_column = find_label_column(df)
    df = df.copy()
    df["target"] = df[label_column].map(label_to_binary)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    present_numeric = [col for col in NUMERIC_CANDIDATES if col in df.columns]
    present_categorical = [col for col in CATEGORICAL_CANDIDATES if col in df.columns]

    features = df[present_numeric + present_categorical].copy()

    for column in present_numeric:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    if "id.orig_h" in df.columns:
        features["orig_is_private"] = df["id.orig_h"].map(lambda value: safe_ip_flag(value, check_private=True))
    if "id.resp_h" in df.columns:
        features["resp_is_private"] = df["id.resp_h"].map(lambda value: safe_ip_flag(value, check_private=True))
    if "id.resp_h" in df.columns:
        features["resp_is_public"] = df["id.resp_h"].map(lambda value: safe_ip_flag(value, check_private=False))

    if {"orig_bytes", "resp_bytes"}.issubset(features.columns):
        features["total_bytes"] = features["orig_bytes"].fillna(0) + features["resp_bytes"].fillna(0)
        features["byte_ratio"] = features["orig_bytes"] / (features["resp_bytes"] + 1)
    if {"orig_pkts", "resp_pkts"}.issubset(features.columns):
        features["total_pkts"] = features["orig_pkts"].fillna(0) + features["resp_pkts"].fillna(0)
        features["pkt_ratio"] = features["orig_pkts"] / (features["resp_pkts"] + 1)
    if {"duration", "orig_bytes", "resp_bytes"}.issubset(features.columns):
        features["bytes_per_second"] = (
            (features["orig_bytes"].fillna(0) + features["resp_bytes"].fillna(0))
            / (features["duration"].replace(0, np.nan) + 1e-6)
        )

    return features, df["target"]


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def get_models(random_state: int) -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, Pipeline]:
    results = []
    fitted_models: Dict[str, Pipeline] = {}

    for model_name, estimator in get_models(random_state).items():
        preprocessor, _, _ = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
            }
        )

        matrix = confusion_matrix(y_test, predictions)
        plot_confusion_matrix(matrix, model_name, output_dir / f"confusion_matrix_{model_name}.png")
        fitted_models[model_name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    best_model_name = metrics_df.iloc[0]["model"]
    return metrics_df, pd.Series({"best_model": best_model_name}), fitted_models[best_model_name]


def plot_confusion_matrix(matrix: np.ndarray, model_name: str, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def prepare_clustering_matrix(X: pd.DataFrame) -> np.ndarray:
    numeric_X = X.select_dtypes(include=[np.number]).copy()
    numeric_X = numeric_X.replace([np.inf, -np.inf], np.nan)
    numeric_X = numeric_X.fillna(numeric_X.median(numeric_only=True))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_X)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(scaled, -10.0, 10.0)


def summarize_cluster_labels(
    method_name: str,
    labels: np.ndarray,
    matrix: np.ndarray,
    y: pd.Series,
) -> Dict[str, float | str]:
    unique_labels = sorted(set(labels))
    non_noise_labels = [label for label in unique_labels if label != -1]
    cluster_count = len(non_noise_labels)
    noise_fraction = float(np.mean(labels == -1)) if -1 in unique_labels else 0.0

    if cluster_count >= 2:
        valid_mask = labels != -1 if -1 in unique_labels else np.ones(len(labels), dtype=bool)
        silhouette = float(silhouette_score(matrix[valid_mask], labels[valid_mask]))
    else:
        silhouette = np.nan

    summary: Dict[str, float | str] = {
        "method": method_name,
        "cluster_count": cluster_count,
        "noise_fraction": noise_fraction,
        "silhouette_score": silhouette,
    }

    for label in non_noise_labels[:4]:
        mask = labels == label
        summary[f"cluster_{label}_size"] = int(np.sum(mask))
        summary[f"cluster_{label}_malicious_rate"] = float(y[mask].mean()) if np.any(mask) else np.nan

    return summary


def plot_cluster_projection(
    coords: np.ndarray,
    labels: np.ndarray,
    y: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    cluster_df = pd.DataFrame(
        {
            "pc1": coords[:, 0],
            "pc2": coords[:, 1],
            "cluster": labels.astype(str),
            "target": y.map({0: "Benign", 1: "Malicious"}),
        }
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=cluster_df, x="pc1", y="pc2", hue="cluster", style="target", alpha=0.65, s=30)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_clustering(X: pd.DataFrame, y: pd.Series, output_dir: Path, random_state: int) -> pd.DataFrame:
    clustering_limit = 10000
    if len(X) > clustering_limit:
        sampled = X.assign(target=y).sample(n=clustering_limit, random_state=random_state, replace=False)
        y = sampled.pop("target")
        X = sampled

    matrix = prepare_clustering_matrix(X)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(matrix)

    clustering_runs = {
        "kmeans": KMeans(n_clusters=2, random_state=random_state, n_init=10).fit_predict(matrix),
        "agglomerative": AgglomerativeClustering(n_clusters=2).fit_predict(matrix),
        "dbscan": DBSCAN(eps=0.9, min_samples=20).fit_predict(matrix),
    }

    summaries = []
    for method_name, labels in clustering_runs.items():
        summaries.append(summarize_cluster_labels(method_name, labels, matrix, y))
        plot_cluster_projection(
            coords,
            labels,
            y,
            f"{method_name.replace('_', ' ').title()} Clustering on IoT-23 (PCA Projection)",
            output_dir / f"clustering_{method_name}_pca.png",
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "clustering_summary.csv", index=False)
    return summary_df


def extract_feature_importance(best_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    return importance_df


def select_reduced_columns(importance_df: pd.DataFrame, top_n: int = 10) -> List[str]:
    selected = []
    for feature in importance_df["feature"]:
        if "__" in feature:
            raw_name = feature.split("__", 1)[1]
        else:
            raw_name = feature

        if "_" in raw_name and raw_name not in NUMERIC_CANDIDATES:
            maybe_base = raw_name.split("_", 1)[0]
            if maybe_base in CATEGORICAL_CANDIDATES:
                selected.append(maybe_base)
                continue

        selected.append(raw_name)

    deduped = []
    for item in selected:
        if item not in deduped:
            deduped.append(item)
    return deduped[:top_n]


def evaluate_reduced_features(
    X: pd.DataFrame,
    y: pd.Series,
    selected_columns: Iterable[str],
    output_dir: Path,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    reduced_cols = [col for col in selected_columns if col in X.columns]
    reduced_X = X[reduced_cols].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        reduced_X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor, _, _ = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=250,
                random_state=random_state,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=1,
            )),
        ]
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    reduced_df = pd.DataFrame(
        [
            {
                "selected_feature_count": len(reduced_cols),
                "selected_features": ", ".join(reduced_cols),
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1_score": f1_score(y_test, predictions, zero_division=0),
            }
        ]
    )
    reduced_df.to_csv(output_dir / "reduced_feature_metrics.csv", index=False)
    return reduced_df


def write_summary(
    output_dir: Path,
    dataset_path: Path,
    row_count: int,
    metrics_df: pd.DataFrame,
    clustering_summary_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    reduced_df: pd.DataFrame,
) -> None:
    top_features = importance_df.head(8)["feature"].tolist()
    best_row = metrics_df.sort_values(by="f1_score", ascending=False).iloc[0]
    removed_candidates = importance_df.tail(5)["feature"].tolist()

    best_clustering = clustering_summary_df.sort_values(
        by=["silhouette_score", "cluster_count"],
        ascending=[False, False],
        na_position="last",
    ).iloc[0]

    lines = [
        f"Dataset used: {dataset_path}",
        f"Samples used after cleaning: {row_count}",
        "",
        "Baseline classification summary:",
        (
            f"Best model: {best_row['model']} with accuracy={best_row['accuracy']:.4f}, "
            f"precision={best_row['precision']:.4f}, recall={best_row['recall']:.4f}, "
            f"f1={best_row['f1_score']:.4f}."
        ),
        "",
        "Clustering summary:",
        (
            f"Best clustering method: {best_clustering['method']} with "
            f"silhouette score={best_clustering['silhouette_score']:.4f}, "
            f"cluster_count={int(best_clustering['cluster_count'])}, "
            f"noise_fraction={best_clustering['noise_fraction']:.4f}."
        ),
        "",
        "Most useful signals:",
        ", ".join(top_features),
        "",
        "Signals removed or deprioritized:",
        ", ".join(removed_candidates),
        "",
        "Reduced feature evaluation:",
        (
            f"Using {int(reduced_df.iloc[0]['selected_feature_count'])} selected features, "
            f"Random Forest reached accuracy={reduced_df.iloc[0]['accuracy']:.4f} "
            f"and f1={reduced_df.iloc[0]['f1_score']:.4f}."
        ),
    ]

    (output_dir / "phase2_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = normalize_missing_values(read_iot23_file(dataset_path))
    features_df, target = build_feature_frame(raw_df)

    if args.sample_size and len(features_df) > args.sample_size:
        sampled = features_df.assign(target=target).sample(
            n=args.sample_size,
            random_state=args.random_state,
            replace=False,
        )
        target = sampled.pop("target")
        features_df = sampled

    features_df.head(200).to_csv(output_dir / "features_preview.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        features_df,
        target,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=target,
    )

    metrics_df, best_model_info, best_pipeline = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        output_dir,
        args.random_state,
    )
    metrics_df.to_csv(output_dir / "classification_metrics.csv", index=False)

    clustering_summary_df = run_clustering(features_df, target, output_dir, args.random_state)

    best_pipeline.fit(X_train, y_train)
    importance_df = extract_feature_importance(best_pipeline, X_train, y_train)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    selected_columns = select_reduced_columns(importance_df, top_n=10)
    reduced_df = evaluate_reduced_features(
        features_df,
        target,
        selected_columns,
        output_dir,
        args.test_size,
        args.random_state,
    )

    metadata = {
        "dataset": str(dataset_path),
        "rows_after_cleaning": int(len(features_df)),
        "best_model": best_model_info["best_model"],
        "selected_columns": selected_columns,
        "best_clustering_method": clustering_summary_df.sort_values(
            by=["silhouette_score", "cluster_count"],
            ascending=[False, False],
            na_position="last",
        ).iloc[0]["method"],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_summary(
        output_dir,
        dataset_path,
        len(features_df),
        metrics_df,
        clustering_summary_df,
        importance_df,
        reduced_df,
    )

    print(f"Phase 2 pipeline completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
