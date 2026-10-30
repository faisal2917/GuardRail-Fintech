
# GuardRail Fintech — Credit Card Fraud Intelligence

A PyTorch fraud-detection model you train on your own machine, served through a Streamlit dashboard with
real-time scoring, SHAP explanations, batch CSV screening and model analytics.

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Collect and prepare the data

`data_pipeline.py` has two classes: `KaggleDataCollector` downloads
[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to `data/raw/`, and
`DataPreparer` cleans it (drops missing, invalid and 1,081 duplicate rows) and writes a stratified 70/15/15 split to
`data/processed/`.

```bash
python data_pipeline.py download   # → data/raw/creditcard.csv
python data_pipeline.py prepare    # → data/processed/train.csv, val.csv, test.csv, data_report.json
# or both at once:
python data_pipeline.py all
```

If Kaggle asks you to sign in, create a token at kaggle.com → Settings → API → *Create New Token* and save it as
`~/.kaggle/kaggle.json`.

## 3. Train the model (locally, PyTorch)

```bash
python train.py
```

The trainer picks the fastest device automatically (CUDA → Apple-silicon MPS → CPU). Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--synthetic` | off | Train on generated data with the same schema (no download; demo only) |
| `--device` | `auto` | Force `cpu`, `cuda` or `mps` |
| `--epochs` | 30 | Max epochs (early stopping on validation PR-AUC, patience 6) |
| `--hidden` | `128 64 32` | Hidden layer widths |
| `--pos-weight` | √(neg/pos) | Positive-class weight for the imbalanced loss |

What it does: prepared splits → standardisation fitted on train only → MLP with BatchNorm/SiLU/Dropout
trained with weighted `BCEWithLogitsLoss` and AdamW → best epoch by validation PR-AUC → decision threshold
chosen to maximise F1 on validation → final metrics on the untouched test set → global SHAP importance.

Everything the app needs is written to `artifacts/`.

## 4. Run the dashboard

```bash
streamlit run app.py
```

Tabs: **Real-Time Risk Analyzer** (presets, gauge, cost matrix) · **Explainable AI** (SHAP contributions and
narrative) · **Batch CSV Scanner** (upload, filter, highlight, download) · **Model Insights** (PR/ROC, confusion
matrix, training curves, feature importance). The sidebar sets the block threshold, review cost and chargeback fee.

## 5. Deploy to Streamlit Community Cloud

1. Train locally and commit `artifacts/` (small). `data/*.csv` is git-ignored.
2. Push to GitHub.
3. At <https://share.streamlit.io> → **Create app** → choose the repo, branch and `app.py`; set Python 3.12 under
   *Advanced settings*. `requirements.txt` pulls the CPU-only PyTorch wheel on Linux.
