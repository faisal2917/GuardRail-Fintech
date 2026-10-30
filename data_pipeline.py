"""
Data collection and preparation for GuardRail.

Dataset: Kaggle "Credit Card Fraud Detection" (ULB) — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
284,807 European card transactions over two days; 492 frauds (0.17%).

Usage:
    python data_pipeline.py download   # Kaggle → data/raw/creditcard.csv
    python data_pipeline.py prepare    # clean + stratified split → data/processed/{train,val,test}.csv
    python data_pipeline.py all        # both steps

If Kaggle asks for authentication, create an API token at https://www.kaggle.com/settings (API → Create New Token)
and save it to ~/.kaggle/kaggle.json, or export KAGGLE_USERNAME and KAGGLE_KEY.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_model import FEATURES, TARGET

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
COLUMNS = FEATURES + [TARGET]


class KaggleDataCollector:
    """Downloads the credit-card fraud dataset from Kaggle into `data/raw/`."""

    DATASET = "mlg-ulb/creditcardfraud"
    FILENAME = "creditcard.csv"

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = raw_dir
        self.target = raw_dir / self.FILENAME

    def download(self, force: bool = False) -> Path:
        if self.target.exists() and not force:
            print(f"Already downloaded: {self.target} (use --force to re-download)")
            return self.verify()

        try:
            import kagglehub
        except ImportError:
            raise SystemExit("kagglehub is not installed. Run: pip install -r requirements.txt")

        print(f"Downloading {self.DATASET} from Kaggle…")
        try:
            cache_dir = Path(kagglehub.dataset_download(self.DATASET, force_download=force))
        except Exception as exc:
            raise SystemExit(
                f"Kaggle download failed: {exc}\n\n"
                "Fix one of these and re-run:\n"
                "  • Save your Kaggle API token to ~/.kaggle/kaggle.json (kaggle.com → Settings → API → Create New Token)\n"
                "  • or export KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>\n"
                f"  • or download {self.FILENAME} manually from https://www.kaggle.com/datasets/{self.DATASET} into {self.raw_dir}/"
            )

        source = next(cache_dir.rglob(self.FILENAME), None)
        if source is None:
            raise SystemExit(f"{self.FILENAME} not found in the downloaded files at {cache_dir}")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.target)
        print(f"Saved to {self.target} ({self.target.stat().st_size / 1e6:.1f} MB)")
        return self.verify()

    def verify(self) -> Path:
        header = pd.read_csv(self.target, nrows=0).columns.tolist()
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            raise SystemExit(f"{self.target} is missing expected columns: {missing}")
        return self.target


@dataclass
class DataPreparer:
    """Cleans the raw dataset and writes stratified train / validation / test splits."""

    raw_path: Path = RAW_DIR / KaggleDataCollector.FILENAME
    out_dir: Path = PROCESSED_DIR
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42

    def load(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise SystemExit(f"Raw data not found at {self.raw_path}. Run: python data_pipeline.py download")
        df = pd.read_csv(self.raw_path)
        missing = [c for c in COLUMNS if c not in df.columns]
        if missing:
            raise SystemExit(f"Raw data is missing columns: {missing}")
        return df[COLUMNS]

    def clean(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        report = {"rows_raw": len(df), "frauds_raw": int(df[TARGET].sum())}

        df = df.apply(pd.to_numeric, errors="coerce")
        n = len(df)
        df = df.dropna()
        report["dropped_missing"] = n - len(df)

        n = len(df)
        df = df[(df["Amount"] >= 0) & (df["Time"] >= 0) & df[TARGET].isin([0, 1])]
        report["dropped_invalid"] = n - len(df)

        # Exact duplicate rows would otherwise land in both train and test and inflate the scores.
        n = len(df)
        df = df.drop_duplicates()
        report["dropped_duplicates"] = n - len(df)

        df = df.astype({TARGET: "int8"}).reset_index(drop=True)
        report.update(rows_clean=len(df), frauds_clean=int(df[TARGET].sum()), fraud_rate=float(df[TARGET].mean()))
        return df, report

    def split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        holdout = self.val_size + self.test_size
        train, rest = train_test_split(df, test_size=holdout, stratify=df[TARGET], random_state=self.seed)
        val, test = train_test_split(rest, test_size=self.test_size / holdout, stratify=rest[TARGET], random_state=self.seed)
        return {"train": train, "val": val, "test": test}

    def run(self) -> dict:
        df, report = self.clean(self.load())
        splits = self.split(df)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        for name, part in splits.items():
            part.to_csv(self.out_dir / f"{name}.csv", index=False)

        report.update(
            prepared_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            source=str(self.raw_path.name),
            seed=self.seed,
            splits={name: {"rows": len(p), "frauds": int(p[TARGET].sum())} for name, p in splits.items()},
            amount={k: round(float(v), 2) for k, v in df["Amount"].describe().items()},
        )
        (self.out_dir / "data_report.json").write_text(json.dumps(report, indent=2))

        print(
            f"Raw {report['rows_raw']:,} rows → clean {report['rows_clean']:,} "
            f"(missing −{report['dropped_missing']}, invalid −{report['dropped_invalid']}, duplicates −{report['dropped_duplicates']:,})"
        )
        print(f"Fraud rate {report['fraud_rate']:.4%} ({report['frauds_clean']} frauds)")
        for name, s in report["splits"].items():
            print(f"  {name:<5} {s['rows']:>8,} rows · {s['frauds']:>4} frauds → {self.out_dir / (name + '.csv')}")
        return report


def main() -> None:
    p = argparse.ArgumentParser(description="Collect and prepare the Kaggle credit-card fraud dataset.")
    p.add_argument("step", choices=["download", "prepare", "all"])
    p.add_argument("--force", action="store_true", help="Re-download even if the raw file exists.")
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.step in ("download", "all"):
        KaggleDataCollector().download(force=args.force)
    if args.step in ("prepare", "all"):
        DataPreparer(val_size=args.val_size, test_size=args.test_size, seed=args.seed).run()


if __name__ == "__main__":
    main()
