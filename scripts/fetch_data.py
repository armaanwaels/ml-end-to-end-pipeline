"""Download the credit card fraud dataset from OpenML (ID 1597) into data/raw.

This is the ULB/Worldline dataset that Kaggle also hosts, without the `Time`
column. It needs no account, so the pipeline runs from a fresh clone.
"""
from pathlib import Path

from sklearn.datasets import fetch_openml

OUT = Path("data/raw/creditcard.csv")


def main() -> None:
    df = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
    df["Class"] = df["Class"].astype(str).str.strip("'").astype(int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(df):,} cols={df.shape[1]} frauds={int(df['Class'].sum())}")


if __name__ == "__main__":
    main()
