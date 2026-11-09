from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lab import (
    benchmark_demo,
    clean_dataset,
    evaluate_model,
    load_sample_dataset,
    make_model_card,
    parse_dataset,
    reproducible_split,
    save_experiment,
    training_memory_estimate_mb,
    train_baseline,
)


st.set_page_config(
    page_title="Project 13 | Model Training Lab",
    page_icon="🧪",
    layout="wide",
)


def reset_sample() -> None:
    st.session_state.dataset = load_sample_dataset()
    st.session_state.audit = {
        "input_rows": len(st.session_state.dataset),
        "empty_rows_removed": 0,
        "duplicates_removed": 0,
        "conflicting_texts_removed": 0,
        "clean_rows": len(st.session_state.dataset),
    }
    st.session_state.runs = None


if "dataset" not in st.session_state:
    reset_sample()

if "runs" not in st.session_state:
    st.session_state.runs = None


def dataframe_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def render_header(page: str) -> None:
    progress_values = {
        "Experiment": 0.25,
        "Benchmark": 0.60,
        "Model card & artifacts": 0.85,
        "Guide": 1.0,
    }

    st.markdown(
        """
        <div style="
            padding: 1.5rem;
            border-radius: 20px;
            background: linear-gradient(
                135deg,
                #4F46E5,
                #7C3AED,
                #06B6D4
            );
            color: white;
            margin-bottom: 1.2rem;
        ">
            <h1 style="margin: 0;">🧪 Model training lab</h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.05rem;">
                Prepare data, train a lightweight classifier, and understand its behaviour.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        progress_values[page],
        text=f"Current stage: {page}",
    )


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("# 🧪 Project 13")
        st.caption("End-to-end AI model training lab")

        page = st.segmented_control(
            "Workspace",
            [
                "Experiment",
                "Benchmark",
                "Model card & artifacts",
                "Guide",
            ],
            default="Experiment",
            key="workspace",
        )

        st.divider()

        st.markdown("### Run status")

        if st.session_state.runs:
            if st.session_state.runs["mode"] == "demo":
                st.warning(
                    "Demo metrics",
                    icon=":material/science:",
                )
            else:
                st.success(
                    "Real model trained",
                    icon=":material/check_circle:",
                )
        else:
            st.info(
                "No experiment yet",
                icon=":material/hourglass_empty:",
            )

        st.caption("CPU-friendly · offline demo · no paid APIs")

    return page or "Experiment"


def render_dataset_section() -> None:
    df = st.session_state.dataset

    with st.container(border=True):
        st.subheader("1 · Choose and prepare data")

        upload = st.file_uploader(
            "Upload labeled text data",
            type=["csv", "json", "jsonl"],
            help=(
                "Required fields are text and label. "
                "Maximum 10 MB and 20,000 rows."
            ),
        )

        if st.button(
            "Load bundled sample",
            type="secondary",
            key="load_sample",
        ):
            reset_sample()
            st.rerun()

        if upload is not None:
            try:
                uploaded = parse_dataset(
                    upload.getvalue(),
                    upload.name,
                )

                columns = list(uploaded.columns)

                text_index = (
                    columns.index("text")
                    if "text" in columns
                    else 0
                )

                label_index = (
                    columns.index("label")
                    if "label" in columns
                    else min(1, len(columns) - 1)
                )

                text_column = st.selectbox(
                    "Text column",
                    columns,
                    index=text_index,
                )

                label_column = st.selectbox(
                    "Label column",
                    columns,
                    index=label_index,
                )

                if st.button(
                    "Validate and use upload",
                    type="primary",
                    key="use_upload",
                ):
                    cleaned, audit = clean_dataset(
                        uploaded,
                        text_column,
                        label_column,
                    )

                    st.session_state.dataset = cleaned
                    st.session_state.audit = audit
                    st.session_state.runs = None

                    st.success(
                        f"Dataset ready: {len(cleaned):,} rows "
                        f"across {cleaned.label.nunique()} classes."
                    )

            except Exception as exc:
                st.error(f"Could not read this dataset: {exc}")

        st.markdown("### Dataset summary")

        metric_row = st.container(horizontal=True)

        with metric_row:
            st.metric(
                "Clean rows",
                f"{len(df):,}",
                border=True,
            )

            st.metric(
                "Classes",
                df["label"].nunique(),
                border=True,
            )

            removed = (
                st.session_state.audit.get("input_rows", len(df))
                - len(df)
            )

            st.metric(
                "Rows removed",
                removed,
                border=True,
            )

            st.metric(
                "Text examples",
                len(df["text"].unique()),
                border=True,
            )

        counts = (
            df["label"]
            .value_counts()
            .rename_axis("class")
            .reset_index(name="examples")
        )

        left, right = st.columns(2)

        with left:
            st.dataframe(
                counts,
                hide_index=True,
                width="stretch",
                alt="Dataset class counts",
            )

        with right:
            st.bar_chart(
                counts,
                x="class",
                y="examples",
                color="#7C3AED",
                alt="Number of examples in each class",
            )

        selected_labels = st.multiselect(
            "Filter classes",
            sorted(df["label"].unique()),
            default=sorted(df["label"].unique()),
        )

        search_text = st.text_input(
            "Search text examples",
            placeholder="Try words such as reliable or crashing",
        )

        filtered_df = df[df["label"].isin(selected_labels)]

        if search_text.strip():
            filtered_df = filtered_df[
                filtered_df["text"].str.contains(
                    search_text,
                    case=False,
                    na=False,
                )
            ]

        with st.expander(
            f"Preview filtered data · {len(filtered_df)} rows"
        ):
            st.dataframe(
                filtered_df.head(50),
                hide_index=True,
                width="stretch",
                alt="Filtered dataset preview",
            )


def render_experiment() -> None:
    render_dataset_section()

    df = st.session_state.dataset

    with st.container(border=True):
        st.subheader("2 · Configure experiment")

        col1, col2, col3 = st.columns(3)

        seed = col1.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
        )

        train_percentage = col2.select_slider(
            "Training share",
            options=[60, 65, 70, 75, 80],
            value=70,
        )

        feature_limit = col3.select_slider(
            "Vocabulary cap",
            options=[5000, 10000, 20000, 30000],
            value=20000,
        )

        split = reproducible_split(
            df,
            int(seed),
            train_percentage / 100,
            0.15,
        )

        st.info(
            f"Split created: {len(split.train_texts):,} train · "
            f"{len(split.validation_texts):,} validation · "
            f"{len(split.test_texts):,} test",
            icon=":material/account_tree:",
        )

        memory_estimate = training_memory_estimate_mb(
            len(split.train_texts),
            int(feature_limit),
        )

        st.warning(
            f"Estimated CPU working memory: approximately "
            f"{memory_estimate} MB",
            icon=":material/memory:",
        )

        st.markdown(
            """
            **Baseline model**

            TF-IDF word and bigram features → Logistic Regression

            This model is designed for CPU execution and small datasets.
            """
        )

        try:
            import sklearn  # noqa: F401

            sklearn_available = True
        except ImportError:
            sklearn_available = False

        if not sklearn_available:
            st.warning(
                "scikit-learn is not installed. "
                "The deterministic demo is still available.",
                icon=":material/download:",
            )

        train_col, demo_col = st.columns(2)

        with train_col:
            train_button = st.button(
                "🚀 Train real baseline",
                type="primary",
                disabled=not sklearn_available,
                width="stretch",
            )

        with demo_col:
            demo_button = st.button(
                "🔬 Run offline demo",
                width="stretch",
            )

        if train_button:
            progress = st.progress(
                0,
                text="Preparing experiment...",
            )

            try:
                progress.progress(
                    25,
                    text="Preparing train and test data...",
                )

                model, train_metadata = train_baseline(
                    split,
                    int(feature_limit),
                    int(seed),
                )

                progress.progress(
                    70,
                    text="Evaluating held-out test data...",
                )

                validation_report = evaluate_model(
                    model,
                    split,
                    "validation",
                )

                test_report = evaluate_model(
                    model,
                    split,
                    "test",
                )

                errors = [
                    {
                        "text": text,
                        "actual": actual,
                        "predicted": predicted,
                    }
                    for text, actual, predicted in zip(
                        test_report["texts"],
                        test_report["truth"],
                        test_report["predictions"],
                    )
                    if actual != predicted
                ]

                st.session_state.runs = {
                    "mode": "trained",
                    "baseline_model": model,
                    "split": split,
                    "baseline": test_report,
                    "validation": validation_report,
                    "tuned": None,
                    "errors": errors,
                    "train_meta": train_metadata,
                    "seed": int(seed),
                    "feature_limit": int(feature_limit),
                }

                progress.progress(
                    100,
                    text="Experiment complete",
                )

                st.success(
                    "The real baseline was trained and evaluated.",
                    icon=":material/check_circle:",
                )

            except Exception as exc:
                st.error(f"Training failed: {exc}")

        if demo_button:
            baseline, tuned, errors = benchmark_demo(
                df,
                int(seed),
            )

            st.session_state.runs = {
                "mode": "demo",
                "baseline": baseline,
                "validation": None,
                "tuned": tuned,
                "errors": errors,
                "seed": int(seed),
                "feature_limit": int(feature_limit),
                "train_meta": {
                    "algorithm": "Deterministic offline illustration",
                    "fit_seconds": 0,
                },
            }

            st.warning(
                "Demo mode is active. Metrics are deterministic illustrations; "
                "no model was trained.",
                icon=":material/science:",
            )

    with st.container(border=True):
        st.subheader("Optional pretrained-model fine-tuning")

        st.write(
            "Hugging Face fine-tuning is optional and may exceed the memory "
            "available on this laptop. Use the cloud guide for heavier models."
        )

        st.caption(
            "Status: optional cloud path documented"
        )

        st.download_button(
            "Download Colab fine-tuning guide",
            (
                ROOT / "docs" / "colab.md"
            ).read_text(encoding="utf-8"),
            file_name="colab_fine_tuning_guide.md",
            mime="text/markdown",
        )


def render_benchmark() -> None:
    runs = st.session_state.runs

    if not runs:
        st.info(
            "Run an experiment first from the Experiment page."
        )
        return

    if runs["mode"] == "demo":
        st.warning(
            "DEMO · illustrative metrics only",
            icon=":material/science:",
        )
    else:
        st.success(
            "TRAINED · local CPU model",
            icon=":material/check_circle:",
        )

    baseline = runs["baseline"]
    tuned = runs.get("tuned")

    models = [
        ("TF-IDF baseline", baseline),
    ]

    if tuned:
        models.append(
            ("Illustrative fine-tuned model", tuned)
        )

    st.subheader("Held-out model comparison")

    metric_options = [
        "accuracy",
        "macro precision",
        "macro recall",
        "macro F1",
    ]

    selected_metrics = st.multiselect(
        "Metrics to display",
        metric_options,
        default=["accuracy", "macro F1"],
    )

    metric_cards = st.container(horizontal=True)

    with metric_cards:
        st.metric(
            "Accuracy",
            f"{baseline['accuracy']:.3f}",
            border=True,
        )

        st.metric(
            "Macro precision",
            f"{baseline['macro_precision']:.3f}",
            border=True,
        )

        st.metric(
            "Macro recall",
            f"{baseline['macro_recall']:.3f}",
            border=True,
        )

        st.metric(
            "Macro F1",
            f"{baseline['macro_f1']:.3f}",
            border=True,
        )

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "accuracy": report["accuracy"],
                "macro precision": report["macro_precision"],
                "macro recall": report["macro_recall"],
                "macro F1": report["macro_f1"],
                "latency (ms/sample)": report[
                    "inference_ms_per_sample"
                ],
            }
            for name, report in models
        ]
    ).set_index("model")

    if selected_metrics:
        st.bar_chart(
            comparison[selected_metrics],
            alt="Selected classification metrics by model",
        )

    st.subheader("Inference latency")

    st.bar_chart(
        comparison[["latency (ms/sample)"]],
        alt="Inference latency by model",
    )

    st.caption(
        "Lower latency is faster. Demo latency is simulated; "
        "trained latency is measured locally."
    )

    st.subheader("Confusion matrix")

    classes = baseline["classes"]

    matrix = pd.DataFrame(
        baseline["confusion_matrix"],
        index=[f"Actual · {label}" for label in classes],
        columns=[f"Predicted · {label}" for label in classes],
    )

    st.dataframe(
        matrix,
        width="stretch",
        alt="Held-out confusion matrix",
    )

    st.subheader("Per-class scores")

    per_class = (
        pd.DataFrame.from_dict(
            baseline["per_class"],
            orient="index",
        )
        .rename_axis("class")
        .reset_index()
    )

    st.dataframe(
        per_class,
        hide_index=True,
        width="stretch",
        alt="Precision recall F1 and support by class",
    )

    st.subheader("Error analysis")

    errors = runs.get("errors", [])

    if errors:
        st.dataframe(
            pd.DataFrame(errors),
            hide_index=True,
            width="stretch",
            alt="Misclassified test examples",
        )
    else:
        st.success(
            "No errors were found in this small held-out sample."
        )


def render_artifacts() -> None:
    runs = st.session_state.runs

    if not runs:
        st.info(
            "Run an experiment before generating artifacts."
        )
        return

    df = st.session_state.dataset
    report = runs["baseline"]

    split = runs.get("split")

    if split is None:
        split = reproducible_split(
            df,
            runs["seed"],
        )

    is_demo = runs["mode"] == "demo"

    card = make_model_card(
        df,
        st.session_state.audit,
        report,
        runs["seed"],
        simulated=is_demo,
    )

    st.subheader("Model card")

    st.markdown(card)

    st.download_button(
        "Download model card",
        card,
        file_name="MODEL_CARD.md",
        mime="text/markdown",
        type="primary",
    )

    clean_metrics = {
        key: value
        for key, value in report.items()
        if key not in {"predictions", "truth", "texts"}
    }

    st.download_button(
        "Download experiment metrics",
        json.dumps(clean_metrics, indent=2),
        file_name="metrics.json",
        mime="application/json",
    )

    st.download_button(
        "Download cleaned dataset",
        dataframe_csv(df),
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

    if runs["mode"] == "trained":
        if st.button(
            "Save model and experiment to artifacts/",
            type="secondary",
        ):
            try:
                artifact_path = save_experiment(
                    ROOT / "artifacts",
                    runs["baseline_model"],
                    report,
                    df,
                    split,
                    runs["seed"],
                    runs["train_meta"],
                )

                (
                    artifact_path / "MODEL_CARD.md"
                ).write_text(
                    card,
                    encoding="utf-8",
                )

                st.success(
                    f"Artifacts saved under: {artifact_path}"
                )

            except Exception as exc:
                st.error(f"Could not save artifacts: {exc}")

    else:
        st.caption(
            "Demo mode does not create a pretend model file."
        )

    st.subheader("Try a prediction")

    if runs["mode"] == "trained":
        text = st.text_area(
            "Enter text to classify",
            placeholder="Example: The service was fast and reliable.",
            max_chars=4000,
        )

        if st.button("Predict", key="predict"):
            if text.strip():
                prediction = runs["baseline_model"].predict([text])[0]

                st.info(
                    f"Predicted class: **{prediction}**",
                    icon=":material/psychology:",
                )
            else:
                st.warning("Enter some text first.")

    else:
        st.caption(
            "Run the real baseline to enable predictions."
        )


def render_guide() -> None:
    st.subheader("How this lab works")

    st.markdown(
        """
        1. Upload labeled text data or use the bundled sample.
        2. Clean and validate the dataset.
        3. Create reproducible train, validation, and test splits.
        4. Train a CPU-friendly TF-IDF and Logistic Regression model.
        5. Inspect accuracy, precision, recall, macro-F1, latency,
           confusion matrix, and errors.
        6. Download the metrics, cleaned data, model card, and artifacts.

        **Fine-tuning versus training from scratch**

        Fine-tuning adapts an existing pretrained model to a specific task.
        Training from scratch starts with random weights and requires much
        more data and compute.

        This project does not claim to train a foundation model from scratch.
        """
    )

    st.download_button(
        "Download Colab fine-tuning guide",
        (
            ROOT / "docs" / "colab.md"
        ).read_text(encoding="utf-8"),
        file_name="colab_fine_tuning_guide.md",
        mime="text/markdown",
    )


page = render_sidebar()
render_header(page)

if page == "Experiment":
    render_experiment()
elif page == "Benchmark":
    render_benchmark()
elif page == "Model card & artifacts":
    render_artifacts()
else:
    render_guide()

st.divider()
st.caption(
    "Project 13 · educational benchmark lab · "
    "results depend on data quality, split, and hardware."
)