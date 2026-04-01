#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from phase2_iot23 import (
    build_feature_frame,
    build_preprocessor,
    extract_feature_importance,
    normalize_missing_values,
    read_iot23_file,
)


SCENARIO_METADATA: Dict[str, Dict[str, str]] = {
    "CTU-Honeypot-Capture-4-1.conn.log.labeled": {
        "scenario": "CTU-Honeypot-Capture-4-1",
        "family": "Philips-HUE",
        "capture_type": "benign",
        "device_group": "benign_device",
    },
    "CTU-Honeypot-Capture-5-1.conn.log.labeled": {
        "scenario": "CTU-Honeypot-Capture-5-1",
        "family": "Amazon-Echo",
        "capture_type": "benign",
        "device_group": "benign_device",
    },
    "CTU-Honeypot-Capture-7-1-Somfy-01.conn.log.labeled": {
        "scenario": "CTU-Honeypot-Capture-7-1-Somfy-01",
        "family": "Somfy-Doorlock",
        "capture_type": "benign",
        "device_group": "benign_device",
    },
    "CTU-IoT-Malware-Capture-1-1.conn.log.labeled": {
        "scenario": "CTU-IoT-Malware-Capture-1-1",
        "family": "Hide-and-Seek",
        "capture_type": "malicious",
        "device_group": "malware_family",
    },
    "CTU-IoT-Malware-Capture-3-1.conn.log.labeled": {
        "scenario": "CTU-IoT-Malware-Capture-3-1",
        "family": "Muhstik",
        "capture_type": "malicious",
        "device_group": "malware_family",
    },
    "CTU-IoT-Malware-Capture-8-1.conn.log.labeled": {
        "scenario": "CTU-IoT-Malware-Capture-8-1",
        "family": "Hakai",
        "capture_type": "malicious",
        "device_group": "malware_family",
    },
    "CTU-IoT-Malware-Capture-20-1.conn.log.labeled": {
        "scenario": "CTU-IoT-Malware-Capture-20-1",
        "family": "Torii",
        "capture_type": "malicious",
        "device_group": "malware_family",
    },
    "CTU-IoT-Malware-Capture-34-1.conn.log.labeled": {
        "scenario": "CTU-IoT-Malware-Capture-34-1",
        "family": "Mirai",
        "capture_type": "malicious",
        "device_group": "malware_family",
    },
    "CTU-IoT-Malware-Capture-60-1.conn.log.labeled": {
        "scenario": "CTU-IoT-Malware-Capture-60-1",
        "family": "Gagfyt",
        "capture_type": "malicious",
        "device_group": "malware_family",
    },
}

TRAIN_ONLY_SCENARIOS = {"CTU-IoT-Malware-Capture-20-1"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliable scenario-based IoT-23 evaluation")
    parser.add_argument("--data-dir", default="data/iot23_multi/raw", help="Directory of scenario conn.log.labeled files")
    parser.add_argument("--output-dir", default="outputs/reliable_iot23", help="Directory for reliable evaluation outputs")
    parser.add_argument(
        "--max-per-class-per-scenario",
        type=int,
        default=3000,
        help="Cap per binary class inside each scenario to reduce imbalance and runtime",
    )
    parser.add_argument("--num-splits", type=int, default=5, help="Number of unique scenario holdout splits")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_scenarios(data_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    expected_files = sorted(SCENARIO_METADATA.keys())

    for file_name in expected_files:
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Expected scenario file missing: {path}")

        metadata = SCENARIO_METADATA[file_name]
        df = normalize_missing_values(read_iot23_file(path))
        df["scenario"] = metadata["scenario"]
        df["family"] = metadata["family"]
        df["capture_type"] = metadata["capture_type"]
        df["source_file"] = file_name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def build_feature_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    features, target = build_feature_frame(df)
    metadata = df.loc[features.index, ["scenario", "family", "capture_type", "source_file"]].copy()
    metadata["target"] = target.values
    metadata["binary_label"] = metadata["target"].map({0: "Benign", 1: "Malicious"})
    return features.reset_index(drop=True), target.reset_index(drop=True), metadata.reset_index(drop=True)


def sample_per_scenario_and_label(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    max_per_class_per_scenario: int,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    combined = pd.concat([metadata.reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    combined["target"] = y.values

    sampled_parts = []
    for (_, target_value), group in combined.groupby(["scenario", "target"], sort=False):
        n = min(len(group), max_per_class_per_scenario)
        sampled_parts.append(group.sample(n=n, random_state=random_state, replace=False))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    meta_cols = ["scenario", "family", "capture_type", "source_file", "target", "binary_label"]
    sampled_metadata = sampled[meta_cols].copy()
    sampled_y = sampled_metadata.pop("target")
    sampled_X = sampled.drop(columns=meta_cols)
    return sampled_X, sampled_y, sampled_metadata


def build_split_definitions(metadata: pd.DataFrame, num_splits: int, random_state: int) -> List[Dict[str, List[str]]]:
    benign_scenarios = sorted(metadata.loc[metadata["capture_type"] == "benign", "scenario"].unique().tolist())
    malicious_scenarios = sorted(
        scenario
        for scenario in metadata.loc[metadata["capture_type"] == "malicious", "scenario"].unique().tolist()
        if scenario not in TRAIN_ONLY_SCENARIOS
    )

    rng = np.random.default_rng(random_state)
    splits: List[Dict[str, List[str]]] = []
    seen_signatures = set()

    while len(splits) < num_splits:
        test_benign = rng.choice(benign_scenarios, size=1, replace=False).tolist()
        test_malicious = rng.choice(malicious_scenarios, size=2, replace=False).tolist()
        test_scenarios = sorted(test_benign + test_malicious)
        signature = tuple(test_scenarios)
        if signature in seen_signatures:
            continue

        train_scenarios = sorted(set(metadata["scenario"].unique()) - set(test_scenarios))
        splits.append(
            {
                "split_id": len(splits),
                "train_scenarios": train_scenarios,
                "test_scenarios": test_scenarios,
            }
        )
        seen_signatures.add(signature)

    return splits


def get_models(random_state: int) -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced_subsample",
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=1,
            class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def balance_training_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    combined = pd.concat([meta_train.reset_index(drop=True), X_train.reset_index(drop=True)], axis=1)
    combined["target"] = y_train.values

    min_count = int(combined["target"].value_counts().min())
    balanced_parts = []
    for target_value, group in combined.groupby("target", sort=False):
        balanced_parts.append(group.sample(n=min_count, random_state=random_state, replace=False))

    balanced = pd.concat(balanced_parts, ignore_index=True).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    meta_cols = ["scenario", "family", "capture_type", "source_file", "binary_label"]
    balanced_meta = balanced[meta_cols].copy()
    balanced_y = balanced["target"].copy()
    balanced_X = balanced.drop(columns=meta_cols + ["target"])
    return balanced_X, balanced_y, balanced_meta


def score_predictions(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_score),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_split(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    split_definition: Dict[str, List[str]],
    output_dir: Path,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    test_scenarios = set(split_definition["test_scenarios"])
    train_mask = ~metadata["scenario"].isin(test_scenarios)
    test_mask = metadata["scenario"].isin(test_scenarios)

    X_train = X.loc[train_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    meta_train = metadata.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)
    meta_test = metadata.loc[test_mask].reset_index(drop=True)

    X_train_bal, y_train_bal, meta_train_bal = balance_training_split(
        X_train,
        y_train,
        meta_train,
        random_state + int(split_definition["split_id"]),
    )

    fitted_models: Dict[str, Pipeline] = {}
    results = []

    for model_name, estimator in get_models(random_state).items():
        preprocessor, _, _ = build_preprocessor(X_train_bal)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train_bal, y_train_bal)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]
        scores = score_predictions(y_test, y_pred, y_score)
        scores.update(
            {
                "split_id": split_definition["split_id"],
                "model": model_name,
                "train_rows": len(X_train_bal),
                "test_rows": len(X_test),
            }
        )
        results.append(scores)
        fitted_models[model_name] = pipeline

        if split_definition["split_id"] == 0:
            plot_confusion_matrix(
                np.array([[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]]),
                f"{model_name} (split 0)",
                output_dir / f"confusion_matrix_split0_{model_name}.png",
            )

    split_inventory = {
        "split_id": split_definition["split_id"],
        "train_scenarios": split_definition["train_scenarios"],
        "test_scenarios": split_definition["test_scenarios"],
        "train_benign": int((y_train == 0).sum()),
        "train_malicious": int((y_train == 1).sum()),
        "train_benign_balanced": int((y_train_bal == 0).sum()),
        "train_malicious_balanced": int((y_train_bal == 1).sum()),
        "test_benign": int((y_test == 0).sum()),
        "test_malicious": int((y_test == 1).sum()),
        "scenario_overlap_count": len(set(meta_train["scenario"]) & set(meta_test["scenario"])),
    }
    append_jsonl(output_dir / "split_inventory.jsonl", split_inventory)
    return pd.DataFrame(results), fitted_models


def plot_confusion_matrix(matrix: np.ndarray, title: str, output_path: Path) -> None:
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def write_leakage_report(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    splits: List[Dict[str, List[str]]],
    output_dir: Path,
) -> None:
    split_zero = splits[0]
    test_scenarios = set(split_zero["test_scenarios"])
    train_mask = ~metadata["scenario"].isin(test_scenarios)
    test_mask = metadata["scenario"].isin(test_scenarios)

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    metadata_train = metadata.loc[train_mask].reset_index(drop=True)
    metadata_test = metadata.loc[test_mask].reset_index(drop=True)

    # Exact row overlaps should be rare, but this check helps surface leakage-like duplicates.
    train_hash = pd.util.hash_pandas_object(X_train.fillna("NA").astype(str), index=False)
    test_hash = pd.util.hash_pandas_object(X_test.fillna("NA").astype(str), index=False)
    overlap_hashes = set(train_hash.tolist()) & set(test_hash.tolist())

    report = {
        "feature_columns": X.columns.tolist(),
        "excluded_leakage_columns": ["label", "detailed-label", "scenario", "family", "uid", "ts", "source_file"],
        "train_test_scenario_overlap": sorted(set(metadata_train["scenario"]) & set(metadata_test["scenario"])),
        "exact_feature_row_overlap_count": len(overlap_hashes),
        "train_rows_split0": int(len(X_train)),
        "test_rows_split0": int(len(X_test)),
        "train_target_distribution_split0": {
            "benign": int((y.loc[train_mask] == 0).sum()),
            "malicious": int((y.loc[train_mask] == 1).sum()),
        },
        "test_target_distribution_split0": {
            "benign": int((y.loc[test_mask] == 0).sum()),
            "malicious": int((y.loc[test_mask] == 1).sum()),
        },
    }

    (output_dir / "leakage_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def plot_class_distribution(metadata: pd.DataFrame, output_dir: Path) -> None:
    scenario_counts = (
        metadata.groupby(["scenario", "binary_label"])
        .size()
        .reset_index(name="rows")
        .sort_values(by=["scenario", "binary_label"])
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=scenario_counts, x="scenario", y="rows", hue="binary_label")
    plt.xticks(rotation=45, ha="right")
    plt.title("Sampled IoT-23 Scenario Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "scenario_class_distribution.png", dpi=200)
    plt.close()


def plot_metrics_summary(summary_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary_df[["model", "macro_f1_mean", "macro_f1_std"]].copy()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="model", y="macro_f1_mean", color="#4C78A8")
    plt.errorbar(
        x=np.arange(len(plot_df)),
        y=plot_df["macro_f1_mean"],
        yerr=plot_df["macro_f1_std"],
        fmt="none",
        ecolor="black",
        capsize=5,
    )
    plt.ylim(0, 1.0)
    plt.title("Scenario-Based Macro-F1 by Model")
    plt.tight_layout()
    plt.savefig(output_dir / "model_macro_f1_summary.png", dpi=200)
    plt.close()


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "accuracy": ["mean", "std"],
        "precision": ["mean", "std"],
        "recall": ["mean", "std"],
        "f1_score": ["mean", "std"],
        "macro_f1": ["mean", "std"],
        "balanced_accuracy": ["mean", "std"],
        "pr_auc": ["mean", "std"],
        "mcc": ["mean", "std"],
        "false_positive_rate": ["mean", "std"],
    }
    summary = metrics_df.groupby("model").agg(aggregations)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(by="macro_f1_mean", ascending=False)
    return summary


def write_inventory(metadata: pd.DataFrame, output_dir: Path) -> None:
    inventory = (
        metadata.groupby(["scenario", "family", "capture_type", "binary_label"])
        .size()
        .reset_index(name="rows")
        .pivot_table(
            index=["scenario", "family", "capture_type"],
            columns="binary_label",
            values="rows",
            fill_value=0,
        )
        .reset_index()
    )
    inventory.columns.name = None
    if "Benign" not in inventory.columns:
        inventory["Benign"] = 0
    if "Malicious" not in inventory.columns:
        inventory["Malicious"] = 0
    inventory["total_rows"] = inventory["Benign"] + inventory["Malicious"]
    inventory = inventory.sort_values(by=["capture_type", "scenario"])
    inventory.to_csv(output_dir / "scenario_inventory.csv", index=False)

    class_balance = pd.DataFrame(
        [
            {
                "benign_rows": int((metadata["binary_label"] == "Benign").sum()),
                "malicious_rows": int((metadata["binary_label"] == "Malicious").sum()),
            }
        ]
    )
    class_balance.to_csv(output_dir / "class_balance.csv", index=False)


def write_summary(
    output_dir: Path,
    summary_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    splits: List[Dict[str, List[str]]],
) -> None:
    best_row = summary_df.iloc[0]
    benign_total = int(inventory_df["Benign"].sum()) if "Benign" in inventory_df.columns else 0
    malicious_total = int(inventory_df["Malicious"].sum()) if "Malicious" in inventory_df.columns else 0
    top_features = feature_importance_df.head(10)["feature"].tolist()

    lines = [
        "Reliable IoT-23 scenario-based evaluation",
        "",
        f"Scenarios used: {len(inventory_df)}",
        f"Sampled benign rows: {benign_total}",
        f"Sampled malicious rows: {malicious_total}",
        "",
        (
            f"Best model by macro-F1: {best_row['model']} "
            f"(macro-F1={best_row['macro_f1_mean']:.4f} +/- {best_row['macro_f1_std']:.4f}, "
            f"balanced_accuracy={best_row['balanced_accuracy_mean']:.4f} +/- {best_row['balanced_accuracy_std']:.4f}, "
            f"PR-AUC={best_row['pr_auc_mean']:.4f} +/- {best_row['pr_auc_std']:.4f}, "
            f"MCC={best_row['mcc_mean']:.4f} +/- {best_row['mcc_std']:.4f})."
        ),
        "",
        "Scenario-based test splits:",
    ]

    for split in splits:
        lines.append(
            f"Split {split['split_id']}: test={', '.join(split['test_scenarios'])}; "
            f"train={', '.join(split['train_scenarios'])}"
        )

    lines.extend(
        [
            "",
            "Top signals from the best model on split 0:",
            ", ".join(top_features),
            "",
            "Leakage controls:",
            "Scenario-based holdout was used, label columns were excluded from features, and split 0 had zero scenario overlap between train and test.",
        ]
    )

    (output_dir / "reliable_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_inventory_path = output_dir / "split_inventory.jsonl"
    if split_inventory_path.exists():
        split_inventory_path.unlink()

    raw_df = load_scenarios(data_dir)
    X, y, metadata = build_feature_dataset(raw_df)
    sampled_X, sampled_y, sampled_metadata = sample_per_scenario_and_label(
        X,
        y,
        metadata,
        args.max_per_class_per_scenario,
        args.random_state,
    )

    sampled_features_preview = pd.concat(
        [sampled_metadata.reset_index(drop=True), sampled_X.reset_index(drop=True)],
        axis=1,
    )
    sampled_features_preview.head(300).to_csv(output_dir / "features_preview.csv", index=False)

    write_inventory(sampled_metadata, output_dir)
    plot_class_distribution(sampled_metadata, output_dir)

    split_definitions = build_split_definitions(sampled_metadata, args.num_splits, args.random_state)
    metrics_parts = []
    first_split_models: Dict[str, Pipeline] = {}

    for split_definition in split_definitions:
        split_metrics_df, fitted_models = evaluate_split(
            sampled_X,
            sampled_y,
            sampled_metadata,
            split_definition,
            output_dir,
            args.random_state,
        )
        metrics_parts.append(split_metrics_df)
        if split_definition["split_id"] == 0:
            first_split_models = fitted_models

    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    metrics_df.to_csv(output_dir / "split_metrics.csv", index=False)

    summary_df = summarize_metrics(metrics_df)
    summary_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    plot_metrics_summary(summary_df, output_dir)

    write_leakage_report(sampled_X, sampled_y, sampled_metadata, split_definitions, output_dir)

    best_model_name = summary_df.iloc[0]["model"]
    split_zero = split_definitions[0]
    train_mask = ~sampled_metadata["scenario"].isin(set(split_zero["test_scenarios"]))
    X_train0 = sampled_X.loc[train_mask].reset_index(drop=True)
    y_train0 = sampled_y.loc[train_mask].reset_index(drop=True)
    meta_train0 = sampled_metadata.loc[train_mask].reset_index(drop=True)
    X_train0_bal, y_train0_bal, _ = balance_training_split(X_train0, y_train0, meta_train0, args.random_state)
    best_model = first_split_models[best_model_name]
    best_model.fit(X_train0_bal, y_train0_bal)
    feature_importance_df = extract_feature_importance(best_model, X_train0_bal, y_train0_bal)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    inventory_df = pd.read_csv(output_dir / "scenario_inventory.csv")
    write_summary(output_dir, summary_df, inventory_df, feature_importance_df, split_definitions)

    run_metadata = {
        "data_dir": str(data_dir),
        "num_scenarios": int(sampled_metadata["scenario"].nunique()),
        "num_splits": args.num_splits,
        "max_per_class_per_scenario": args.max_per_class_per_scenario,
        "best_model": best_model_name,
        "test_scenarios_per_split": [split["test_scenarios"] for split in split_definitions],
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    print(f"Reliable scenario-based evaluation completed. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
