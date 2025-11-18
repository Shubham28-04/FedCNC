# server.py
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import flwr as fl
from model import MultiTaskModel

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

def detect_client_files(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "client_*.csv")))
    return files, len(files)

def infer_dims(data_dir: str, mapping_path: str):
    if not os.path.exists(mapping_path):
        raise FileNotFoundError("mapping.json missing in " + mapping_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    num_tools = len(mapping)
    client_files, count = detect_client_files(data_dir)
    if count == 0:
        raise FileNotFoundError(f"No client_*.csv found under {data_dir}")
    df = pd.read_csv(client_files[0], nrows=1)
    input_dim = len([c for c in FEATURE_COLS if c in df.columns])
    return input_dim, num_tools, count

def build_model_from_weights(input_dim: int, num_tools: int, weights_list):
    model = MultiTaskModel(input_dim, num_tools)
    keys = list(model.state_dict().keys())
    state_dict = {}
    import torch
    for k, arr in zip(keys, weights_list):
        state_dict[k] = torch.tensor(arr)
    model.load_state_dict(state_dict)
    return model

def evaluate_on_test(model, test_csv, mapping_path, warning_threshold, warning_horizon):
    if test_csv is None or not os.path.exists(test_csv):
        return
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    df = pd.read_csv(test_csv)
    # check columns
    required = FEATURE_COLS + ["Product ID", "Machine failure", "TWF"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("[Server] test CSV missing columns: ", missing)
        return
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    X_t = torch.tensor(X)
    model.eval()
    with torch.no_grad():
        outs = model(X_t)
        machine_probs = F.softmax(outs["machine"], dim=1)[:, 1].cpu().numpy()
        tool_preds = outs["tool"].argmax(dim=1).cpu().numpy()
        toolfail_probs = F.softmax(outs["toolfail"], dim=1)[:, 1].cpu().numpy()

    # tool wear heuristic
    wear = pd.to_numeric(df["Tool wear [min]"], errors="coerce").fillna(0.0)
    wear_90 = float(wear.quantile(0.9))

    warnings = []
    for i in range(len(df)):
        reasons = []
        if machine_probs[i] >= warning_threshold:
            reasons.append(f"machine_prob={machine_probs[i]:.2f}")
        if wear.iloc[i] >= wear_90:
            reasons.append(f"tool_wear={wear.iloc[i]:.1f}min")
        if reasons:
            est_rounds = int(max(1, round((1.0 - machine_probs[i]) * warning_horizon)))
            warnings.append({
                "row": int(i),
                "product_id": str(df["Product ID"].iloc[i]),
                "tool_pred": int(tool_preds[i]),
                "machine_prob": float(machine_probs[i]),
                "toolfail_prob": float(toolfail_probs[i]),
                "tool_wear": float(wear.iloc[i]),
                "est_warning_in_rounds": est_rounds,
                "reasons": reasons,
            })

    # print summary
    print("----- SERVER TEST EVALUATION -----")
    print(f"Test samples: {len(df)}, Warnings flagged: {len(warnings)} (threshold={warning_threshold})")
    for w in warnings[:10]:
        print(f"Row {w['row']} | Product {w['product_id']} | PredTool={w['tool_pred']} | MachineProb={w['machine_prob']:.2f} | ToolFailProb={w['toolfail_prob']:.2f} | ToolWear={w['tool_wear']:.1f}min")
        print(f"  Estimated warning in ~{w['est_warning_in_rounds']} rounds. Reasons: {', '.join(w['reasons'])}")
        print("  Recommended actions: reduce spindle speed; pause machine; inspect/replace tool; schedule maintenance.\n")
    print("----- END TEST EVALUATION -----")

class SaveWarnStrategy(fl.server.strategy.FedAvg):
    def __init__(self, input_dim, num_tools, test_csv=None, mapping=None, warning_threshold=0.7, warning_horizon=10, save_dir="saved_models", **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_tools = num_tools
        self.test_csv = test_csv
        self.mapping = mapping
        self.warning_threshold = warning_threshold
        self.warning_horizon = warning_horizon
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def aggregate_fit(self, rnd, results, failures):
        agg = super().aggregate_fit(rnd, results, failures)
        if agg is None:
            return agg
        aggregated_weights, metrics = agg

        # Save aggregated weights (numpy)
        np.save(os.path.join(self.save_dir, f"global_weights_round_{rnd}.npy"), aggregated_weights)
        print(f"[Server] Saved aggregated weights round_{rnd}.npy")

        # Try to build model and save torch state
        try:
            model = build_model_from_weights(self.input_dim, self.num_tools, aggregated_weights)
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"global_model_round_{rnd}.pth"))
            print(f"[Server] Saved torch model global_model_round_{rnd}.pth")
        except Exception as e:
            print("[Server] Could not build/save torch model:", e)
            model = None

        # Evaluate on test CSV (if provided)
        if model is not None and self.test_csv:
            try:
                evaluate_on_test(model, self.test_csv, self.mapping, self.warning_threshold, self.warning_horizon)
            except Exception as e:
                print("[Server] Error during test evaluation:", e)

        # Human readable summary
        print(f"\n=== ROUND {rnd} SUMMARY ===")
        print("1) Tool prediction: multi-class tool id (see saved model for predictions).")
        print("2) Tool failure prediction (TWF): binary classification.")
        print("3) Machine failure prediction: binary classification.")
        print(f"4) Early-warning heuristic: flags where machine_prob >= {self.warning_threshold} or tool_wear in 90th percentile.")
        print("   Suggested actions when flagged: reduce spindle speed; pause machine; inspect tool; replace tool; schedule maintenance.\n")
        return aggregated_weights, metrics

def main(args):
    client_files, detected = detect_client_files(args.data_dir)
    print(f"[Server] Detected {detected} client files in {args.data_dir}")

    # decide clients to require
    if args.num_clients is None:
        num_clients = detected
    else:
        num_clients = min(args.num_clients, detected)
        if args.num_clients > detected:
            print(f"[Server] Warning: requested --num-clients {args.num_clients} but only found {detected} CSVs. Using {detected}.")

    if num_clients == 0:
        raise RuntimeError("No client CSVs found. Run prepare_split.py first.")

    input_dim, num_tools, _ = infer_dims(args.data_dir, args.mapping)
    print(f"[Server] input_dim={input_dim}, num_tools={num_tools}, using num_clients={num_clients}")

    strategy = SaveWarnStrategy(
        input_dim=input_dim,
        num_tools=num_tools,
        test_csv=args.test_csv,
        mapping=args.mapping,
        warning_threshold=args.warning_threshold,
        warning_horizon=args.warning_horizon,
        save_dir=args.save_dir,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=lambda rnd: {"local_epochs": args.local_epochs},
    )

    server_config = fl.server.ServerConfig(num_rounds=args.num_rounds)
    print(f"[Server] Starting Flower server at {args.host}:{args.port}, rounds={args.num_rounds}.")
    fl.server.start_server(server_address=f"{args.host}:{args.port}", config=server_config, strategy=strategy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--num-rounds", type=int, default=500)
    parser.add_argument("--num-clients", type=int, default=None, help="Override detected client count (optional)")
    parser.add_argument("--data-dir", type=str, default="data/clients")
    parser.add_argument("--mapping", type=str, default="data/clients/mapping.json")
    parser.add_argument("--test-csv", type=str, default=None, help="Optional test CSV path")
    parser.add_argument("--warning-threshold", type=float, default=0.7)
    parser.add_argument("--warning-horizon", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="saved_models")
    parser.add_argument("--local-epochs", type=int, default=1)
    args = parser.parse_args()
    main(args)
