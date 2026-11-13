# Model Training Lab

A CPU-friendly workflow for labeled text classification. It prepares data, creates deterministic stratified train, validation, and test splits, trains a TF-IDF and Logistic Regression baseline, evaluates results, supports error analysis, and exports experiment records and a model card.

**Live app:** https://model-training-lab.streamlit.app/

**This project trains a small task-specific classifier.** Optional pretrained-model fine-tuning is documented in [docs/colab.md](docs/colab.md). It does not train a foundation model from scratch. No paid APIs, GPU, or API keys are needed.

## Feature status

| Status | Feature |
| --- | --- |
| Implemented | Offline deterministic demo, bundled sample data, CSV/JSON/JSONL upload, cleaning audit, deterministic stratified split, CPU TF-IDF + Logistic Regression, held-out metrics, confusion matrix, per-class report, latency, error analysis, downloads, model card, model serialization and experiment metadata |
| Optional | Hugging Face transformer fine-tuning in a Colab/cloud notebook (internet and downloaded weights required) |
| Future | Local transformer runner, experiment registry, richer benchmark datasets, drift monitoring, deployment endpoint |

Demo metrics and latency are deterministic UI illustrations, not trained-model results. After installing dependencies, the real training button becomes available. The illustrative sample is for demonstration and does not establish real-world model quality.

## Windows 11 setup (PowerShell)

From the project folder:

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal (normally `http://localhost:8501`). The deterministic demo works offline after Streamlit and pandas are installed. Real training needs the packages in `requirements.txt`.

## Run tests

```powershell
python -m pytest -q
```

## Data format and safeguards

Upload CSV, JSON array, or JSON Lines with `text` and `label` fields; alternate names can be selected in the app. Files are limited to 10 MB and datasets to 20,000 rows and 20 classes. Empty examples, exact text-label duplicates, and texts with conflicting labels are removed and counted. Each class needs at least three examples for every split. Avoid sensitive or private data unless authorized.

Splits are stratified and reproducible for the same cleaned row order and seed. Training uses the training partition; validation is for selection, and final metrics use the separate test partition. A fixed seed does not ensure bit-for-bit reproducibility across library versions or machines.

The app estimates memory, caps vocabulary/data size, and uses CPU. Estimates are rough, not a system monitor. On an 8 GB Ryzen 3 7320U laptop, the linear baseline is the recommended local path. Transformer downloads/training are optional and intended for cloud Colab on this hardware profile.

## Artifacts

Download cleaned data, metrics JSON and a model card. A real run can save `baseline_model.joblib`, `experiment.json`, and `MODEL_CARD.md` under `artifacts/`. The experiment record includes dataset hash, seed, split counts, metrics and training settings. Load joblib files only from trusted sources.

## Resume description (honest wording)

> Built a reproducible CPU-first text-classification experiment lab with Streamlit, TF-IDF, and Logistic Regression; implemented dataset validation, stratified train/validation/test splits, macro-F1 and per-class evaluation, latency benchmarking, error analysis, and model-card/artifact export. Documented optional pretrained-model fine-tuning in Colab; did not train a foundation model from scratch.

Only report measured results from a real run and data you can describe appropriately. Label demo outputs as simulated.

## Project layout

```text
app.py                  Streamlit application
src/lab.py              Workflow functions
dataset.csv             Bundled sentiment sample
tests/test_lab.py       Core workflow tests
docs/colab.md           Optional cloud fine-tuning guide
artifacts/              Locally saved model and run metadata
requirements.txt        Core dependencies
```
