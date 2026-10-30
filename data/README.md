Created by `python data_pipeline.py all` — do not edit by hand.

- `raw/creditcard.csv` — Kaggle dataset (mlg-ulb/creditcardfraud), downloaded via kagglehub
- `processed/train.csv`, `val.csv`, `test.csv` — cleaned (duplicates removed), stratified 70/15/15 split
- `processed/data_report.json` — row counts, dropped rows, fraud counts per split

All CSVs are git-ignored. Only `artifacts/` needs to be committed for deployment.
