# src/data.py
import argparse, os, sys
from pathlib import Path

import pandas as pd
import yaml

def load_params(param_arg: str):
    """
    Supports either 'params.yaml' or 'params.yaml:preprocess'.
    Falls back to defaults if file/section missing.
    """
    section = "preprocess"
    path = param_arg
    if ":" in param_arg:
        path, section = param_arg.split(":", 1)

    defaults = {
        "sample_frac": 1.0,      # 0<frac<=1.0; keep all rows by default
        "shuffle": True,
        "random_state": 42,
    }

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        cfg = data.get(section, {}) if isinstance(data, dict) else {}
        return {**defaults, **cfg}
    except FileNotFoundError:
        return defaults

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params", default="params.yaml:preprocess")
    args = ap.parse_args()

    params = load_params(args.params)

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    print(f"[data.py] reading: {inp.resolve()}", flush=True)
    df = pd.read_csv(inp)

    # simple, safe preprocessing
    if params.get("shuffle", True):
        df = df.sample(frac=1.0, random_state=params.get("random_state", 42)).reset_index(drop=True)

    frac = float(params.get("sample_frac", 1.0))
    if 0.0 < frac < 1.0:
        df = df.sample(frac=frac, random_state=params.get("random_state", 42))

    print(f"[data.py] rows={len(df):,}, cols={len(df.columns)}", flush=True)

    df.to_csv(outp, index=False)
    print(f"[data.py] wrote: {outp.resolve()}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
