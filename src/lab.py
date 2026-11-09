"""CPU-conscious text classification utilities for Project 13."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_ROWS = 20_000
MIN_ROWS = 12
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


@dataclass
class SplitData:
    train_texts: list[str]
    validation_texts: list[str]
    test_texts: list[str]
    train_labels: list[str]
    validation_labels: list[str]
    test_labels: list[str]
    train_indices: list[int]
    validation_indices: list[int]
    test_indices: list[int]


def load_sample_dataset() -> pd.DataFrame:
    """Load the bundled, safe, hand-written sentiment example."""
    path = Path(__file__).resolve().parents[1] / "dataset.csv"
    return pd.read_csv(path)


def parse_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the 10 MB upload safety limit.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        from io import BytesIO
        return pd.read_csv(BytesIO(file_bytes), nrows=MAX_ROWS + 1)
    if suffix in {".jsonl", ".ndjson"}:
        from io import BytesIO
        return pd.read_json(BytesIO(file_bytes), lines=True, nrows=MAX_ROWS + 1)
    if suffix == ".json":
        from io import BytesIO
        return pd.read_json(BytesIO(file_bytes))
    raise ValueError("Upload a CSV, JSON, or JSONL dataset.")


def clean_dataset(frame: pd.DataFrame, text_column: str = TEXT_COLUMN,
                  label_column: str = LABEL_COLUMN) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = [name for name in (text_column, label_column) if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Dataset exceeds the {MAX_ROWS:,} row CPU safety limit.")
    raw = frame[[text_column, label_column]].copy()
    raw.columns = [TEXT_COLUMN, LABEL_COLUMN]
    start = len(raw)
    raw[TEXT_COLUMN] = raw[TEXT_COLUMN].fillna("").astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())
    raw[LABEL_COLUMN] = raw[LABEL_COLUMN].fillna("").astype(str).str.strip()
    empty = (raw[TEXT_COLUMN].eq("") | raw[LABEL_COLUMN].eq("")).sum()
    raw = raw.loc[raw[TEXT_COLUMN].ne("") & raw[LABEL_COLUMN].ne("")]
    duplicate = raw.duplicated(subset=[TEXT_COLUMN, LABEL_COLUMN]).sum()
    raw = raw.drop_duplicates(subset=[TEXT_COLUMN, LABEL_COLUMN]).reset_index(drop=True)
    conflicts = raw.groupby(TEXT_COLUMN)[LABEL_COLUMN].nunique()
    conflict_texts = set(conflicts[conflicts > 1].index)
    conflict_count = len(conflict_texts)
    raw = raw.loc[~raw[TEXT_COLUMN].isin(conflict_texts)].reset_index(drop=True)
    counts = raw[LABEL_COLUMN].value_counts()
    too_small = counts[counts < 3].to_dict()
    if too_small:
        raise ValueError("Each class needs at least 3 cleaned examples. Small classes: " + ", ".join(f"{k} ({v})" for k, v in too_small.items()))
    if len(raw) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} cleaned rows; found {len(raw)}.")
    if raw[LABEL_COLUMN].nunique() < 2:
        raise ValueError("At least two distinct classes are required.")
    if raw[LABEL_COLUMN].nunique() > 20:
        raise ValueError("For this lightweight lab, datasets are limited to 20 classes.")
    audit = {"input_rows": start, "empty_rows_removed": int(empty), "duplicates_removed": int(duplicate),
             "conflicting_texts_removed": conflict_count, "clean_rows": len(raw)}
    return raw, audit


def reproducible_split(frame: pd.DataFrame, seed: int = 42,
                       train_fraction: float = .70, validation_fraction: float = .15) -> SplitData:
    """Deterministic, per-class split with a minimum of one sample in each partition."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than one.")
    rng = np.random.default_rng(seed)
    train: list[int] = []; validation: list[int] = []; test: list[int] = []
    for _, group in frame.groupby(LABEL_COLUMN, sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = max(1, int(n * train_fraction))
        n_val = max(1, int(n * validation_fraction))
        if n_train + n_val >= n:
            raise ValueError(f"Class '{group[LABEL_COLUMN].iloc[0]}' needs at least 3 examples to split.")
        train.extend(indices[:n_train].tolist())
        validation.extend(indices[n_train:n_train + n_val].tolist())
        test.extend(indices[n_train + n_val:].tolist())
    for values in (train, validation, test):
        rng.shuffle(values)
    def texts(ids: list[int]) -> list[str]: return frame.loc[ids, TEXT_COLUMN].tolist()
    def labels(ids: list[int]) -> list[str]: return frame.loc[ids, LABEL_COLUMN].tolist()
    return SplitData(texts(train), texts(validation), texts(test), labels(train), labels(validation), labels(test), train, validation, test)


def training_memory_estimate_mb(rows: int, max_features: int = 30000) -> int:
    # Conservative rough estimate for sparse counts, intermediate buffers and Python overhead.
    return int(max(64, min(1024, 64 + rows * min(max_features, 30000) * 8 / (1024 * 1024) * .15)))


def train_baseline(split: SplitData, max_features: int = 20000, seed: int = 42) -> tuple[Any, dict[str, Any]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError("scikit-learn is not installed. Install requirements.txt to train a real model; demo mode works without it.") from exc
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=250, random_state=seed, class_weight="balanced", solver="liblinear")),
    ])
    started = time.perf_counter()
    model.fit(split.train_texts, split.train_labels)
    fit_seconds = time.perf_counter() - started
    return model, {"fit_seconds": round(fit_seconds, 4), "max_features": max_features, "algorithm": "TF-IDF + Logistic Regression"}


def metric_report(y_true: list[str], y_pred: list[str], classes: list[str]) -> dict[str, Any]:
    try:
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                     confusion_matrix, classification_report)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average="macro", zero_division=0)
        matrix = confusion_matrix(y_true, y_pred, labels=classes).tolist()
        report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
        per_class = {label: {"precision": report[label]["precision"], "recall": report[label]["recall"], "f1": report[label]["f1-score"], "support": int(report[label]["support"])} for label in classes}
        accuracy = accuracy_score(y_true, y_pred)
    except ImportError:
        # Keeps demo mode usable where scikit-learn hasn't been installed yet.
        matrix = [[sum(a == cls and b == pred for a, b in zip(y_true, y_pred)) for pred in classes] for cls in classes]
        per_class = {}
        ps = []; rs = []; fs = []
        for cls in classes:
            tp = matrix[classes.index(cls)][classes.index(cls)]
            support = sum(1 for x in y_true if x == cls)
            predicted = sum(1 for x in y_pred if x == cls)
            precision = tp / predicted if predicted else 0
            recall = tp / support if support else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
            per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
            ps.append(precision); rs.append(recall); fs.append(f1)
        p, r, f = float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))
        accuracy = sum(a == b for a, b in zip(y_true, y_pred)) / max(len(y_true), 1)
    return {"accuracy": float(accuracy), "macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f),
            "classes": classes, "confusion_matrix": matrix, "per_class": per_class}


def evaluate_model(model: Any, split: SplitData, partition: str = "test") -> dict[str, Any]:
    texts = split.test_texts if partition == "test" else split.validation_texts
    labels = split.test_labels if partition == "test" else split.validation_labels
    started = time.perf_counter()
    predictions = model.predict(texts).tolist()
    elapsed = time.perf_counter() - started
    report = metric_report(labels, predictions, sorted(set(split.train_labels)))
    report["inference_ms_per_sample"] = elapsed / max(len(texts), 1) * 1000
    report["predictions"] = predictions
    report["truth"] = labels
    report["texts"] = texts
    return report


def benchmark_demo(frame: pd.DataFrame, seed: int = 42) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    """Offline deterministic example. Uses fixed predictions and does not claim training took place."""
    split = reproducible_split(frame, seed)
    baseline, tuned = [], []
    for truth, i in zip(split.test_labels, split.test_indices):
        # Stable deterministic misses; no hidden network/model dependency.
        miss = hashlib.sha256(f"{seed}:{i}".encode()).digest()[0] < 54
        baseline.append(("negative" if truth == "positive" else "positive") if miss else truth)
        stronger = hashlib.sha256(f"tuned:{seed}:{i}".encode()).digest()[0] < 28
        tuned.append(("negative" if truth == "positive" else "positive") if stronger else truth)
    a = metric_report(split.test_labels, baseline, sorted(set(split.train_labels)))
    b = metric_report(split.test_labels, tuned, sorted(set(split.train_labels)))
    # Stable illustrative latencies, deliberately labeled simulated.
    a["inference_ms_per_sample"] = 1.8
    b["inference_ms_per_sample"] = 4.9
    errors = []
    for i, (actual, predicted) in enumerate(zip(split.test_labels, baseline)):
        if actual != predicted:
            errors.append({"text": split.test_texts[i], "actual": actual, "predicted": predicted})
    return a, b, errors


def save_experiment(output_dir: str | Path, model: Any, report: dict[str, Any],
                    dataset: pd.DataFrame, split: SplitData, seed: int, metadata: dict[str, Any] | None = None) -> Path:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is missing. Install requirements.txt first.") from exc
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "baseline_model.joblib")
    safe_report = {k: v for k, v in report.items() if k not in {"predictions", "truth", "texts"}}
    payload = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
               "dataset_rows": len(dataset), "dataset_sha256": hashlib.sha256(dataset.to_csv(index=False).encode()).hexdigest(),
               "split_rows": {"train": len(split.train_texts), "validation": len(split.validation_texts), "test": len(split.test_texts)},
               "metrics": safe_report, "metadata": metadata or {}}
    (directory / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return directory


def make_model_card(dataset: pd.DataFrame, audit: dict[str, int], report: dict[str, Any], seed: int,
                    model_name: str = "TF-IDF + Logistic Regression", simulated: bool = False) -> str:
    classes = ", ".join(sorted(dataset[LABEL_COLUMN].unique()))
    status = "**DEMO ONLY:** no model was trained; the scores and latency below are deterministic illustrations.\n\n" if simulated else ""
    training_scope = "No model training was performed for this demo." if simulated else "Supervised training of a lightweight task-specific classifier; this is **not** training a foundation model from scratch."
    latency_scope = "simulated demo value" if simulated else "local measurement; varies by machine"
    return f"""# Model card: {model_name}

{status}## Model details

- Architecture: sparse TF-IDF word and bigram features with Logistic Regression.
- Training approach: {training_scope}
- Reproducibility seed: `{seed}`.

## Dataset
- Rows after cleaning: {len(dataset)} ({audit.get('input_rows', len(dataset))} input rows).
- Classes: {classes}.
- Cleaning: removed {audit.get('empty_rows_removed', 0)} empty rows, {audit.get('duplicates_removed', 0)} duplicate rows and {audit.get('conflicting_texts_removed', 0)} conflicting text groups.
- Source: user-provided or bundled sample; consult the original data license and consent before redistribution.

## Intended use
Educational text classification experiments and triage prototypes. Validate on representative, permissioned data before any operational use.

## Evaluation (held-out test split)
- Accuracy: {report.get('accuracy', 0):.3f}
- Macro precision: {report.get('macro_precision', 0):.3f}
- Macro recall: {report.get('macro_recall', 0):.3f}
- Macro F1: {report.get('macro_f1', 0):.3f}
- Inference latency: {report.get('inference_ms_per_sample', 0):.3f} ms/sample ({latency_scope}).

## Limitations and risks
- The small sample is synthetic and cannot establish real-world quality.
- Results depend on label quality, class balance, language, topic and data collection process.
- Text models can encode demographic and domain biases; errors need human review.
- Do not use for high-stakes decisions, sensitive traits, or as a substitute for expert judgment.
- Test-set results can be optimistic if similar or duplicated examples leak across splits.

## Reproducibility
Install the pinned requirements, use the recorded dataset hash and seed, and follow the Windows steps in README.md. The split is stratified and deterministic.
"""
