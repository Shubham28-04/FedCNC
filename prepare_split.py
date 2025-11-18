# prepare_split.py
import argparse
import os
import json
import pandas as pd
import numpy as np

def split_csv(input_csv, out_dir, num_clients):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    if "Product ID" not in df.columns:
        raise KeyError("Dataset must contain 'Product ID' column")

    # Build mapping Product ID -> int (strings as keys)
    product_ids = df["Product ID"].astype(str).unique().tolist()
    mapping = {str(pid): idx for idx, pid in enumerate(product_ids)}

    with open(os.path.join(out_dir, "mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"[prepare_split] Saved mapping.json with {len(mapping)} tools to {out_dir}")

    # Shuffle for fairness
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    splits = np.array_split(df, num_clients)

    for i, part in enumerate(splits):
        out_path = os.path.join(out_dir, f"client_{i}.csv")
        part.to_csv(out_path, index=False)
        print(f"[prepare_split] Wrote {out_path} ({len(part)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to cnc_dataset.csv")
    parser.add_argument("--out-dir", default="data/clients", help="Output folder for client CSVs")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of clients to split into")
    args = parser.parse_args()
    split_csv(args.input, args.out_dir, args.num_clients)
