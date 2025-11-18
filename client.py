# client.py
import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import flwr as fl
from model import MultiTaskModel
from sklearn.model_selection import train_test_split

# Features used from your CNC CSV
FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

LABEL_MACHINE = "Machine failure"
LABEL_TOOLFAIL = "TWF"
PRODUCT_ID_COL = "Product ID"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_client_csv(csv_path: str, mapping_path: str, batch_size: int = 32):
    df = pd.read_csv(csv_path)
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)  # keys are strings

    # Validate columns
    for c in FEATURE_COLS:
        if c not in df.columns:
            raise KeyError(f"Feature column '{c}' missing in {csv_path}")
    for col in (LABEL_MACHINE, LABEL_TOOLFAIL, PRODUCT_ID_COL):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' missing in {csv_path}")

    # Features → numeric
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)
    y_tool = df[PRODUCT_ID_COL].astype(str).map(lambda v: mapping.get(v, -1)).astype(np.int64).values
    y_machine = pd.to_numeric(df[LABEL_MACHINE], errors="coerce").fillna(0).astype(np.int64).values
    y_toolfail = pd.to_numeric(df[LABEL_TOOLFAIL], errors="coerce").fillna(0).astype(np.int64).values

    # Train/val split (80/20)
    X_train, X_val, t_tool, v_tool, t_machine, v_machine, t_toolfail, v_toolfail = train_test_split(
        X, y_tool, y_machine, y_toolfail, test_size=0.2, random_state=42
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(t_tool), torch.from_numpy(t_machine), torch.from_numpy(t_toolfail)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(v_tool), torch.from_numpy(v_machine), torch.from_numpy(v_toolfail)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    input_dim = X.shape[1]
    num_tools = len(mapping)
    return train_loader, val_loader, input_dim, num_tools

# helpers
def get_weights(model: nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model: nn.Module, weights):
    state_dict = model.state_dict()
    new_state = {}
    for (k, _), arr in zip(state_dict.items(), weights):
        new_state[k] = torch.tensor(arr)
    model.load_state_dict(new_state)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for X, y_tool, y_machine, y_toolfail in loader:
        X = X.to(DEVICE).float()
        y_tool = y_tool.to(DEVICE).long()
        y_machine = y_machine.to(DEVICE).long()
        y_toolfail = y_toolfail.to(DEVICE).long()

        optimizer.zero_grad()
        outs = model(X)
        loss = criterion(outs["tool"], y_tool) + criterion(outs["machine"], y_machine) + criterion(outs["toolfail"], y_toolfail)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))

def evaluate_local(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct_tool = correct_machine = correct_to = 0
    n = 0
    with torch.no_grad():
        for X, y_tool, y_machine, y_toolfail in loader:
            X = X.to(DEVICE).float()
            y_tool = y_tool.to(DEVICE).long()
            y_machine = y_machine.to(DEVICE).long()
            y_toolfail = y_toolfail.to(DEVICE).long()
            outs = model(X)
            loss = criterion(outs["tool"], y_tool) + criterion(outs["machine"], y_machine) + criterion(outs["toolfail"], y_toolfail)
            loss_sum += loss.item()
            preds_tool = outs["tool"].argmax(dim=1)
            preds_machine = outs["machine"].argmax(dim=1)
            preds_to = outs["toolfail"].argmax(dim=1)
            correct_tool += (preds_tool == y_tool).sum().item()
            correct_machine += (preds_machine == y_machine).sum().item()
            correct_to += (preds_to == y_toolfail).sum().item()
            n += y_tool.size(0)
    return loss_sum / max(1, len(loader)), (correct_tool / n if n else 0.0), (correct_machine / n if n else 0.0), (correct_to / n if n else 0.0)

class CNCClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, local_epochs=1):
        self.model = model.to(DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.local_epochs = local_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config=None):
        return get_weights(self.model)

    def set_parameters(self, parameters):
        set_weights(self.model, parameters)

    def fit(self, parameters, config):
        # update weights from server then train locally
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", self.local_epochs)) if config else self.local_epochs
        for e in range(epochs):
            loss = train_one_epoch(self.model, self.train_loader, self.optimizer, self.criterion)
        # return updated parameters and training set size
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": float(loss)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc_tool, acc_machine, acc_toolfail = evaluate_local(self.model, self.val_loader, self.criterion)
        # Print human-readable evaluation summary on client terminal
        print(f"[Client Eval] loss={loss:.4f} | acc_tool={acc_tool:.3f} | acc_toolfail={acc_toolfail:.3f} | acc_machine={acc_machine:.3f}")
        # If round info supplied and <10 show warning text
        round_idx = config.get("round", None) if config else None
        if round_idx is not None and int(round_idx) < 10:
            print("⚠️ Early-warning check: server round < 10 — consider immediate inspection if metrics are poor.")
        return float(loss), len(self.val_loader.dataset), {"acc_tool": float(acc_tool), "acc_machine": float(acc_machine), "acc_toolfail": float(acc_toolfail)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--data-dir", type=str, default="data/clients")
    parser.add_argument("--mapping", type=str, default="data/clients/mapping.json")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    csv_path = os.path.join(args.data_dir, f"client_{args.client_id}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run prepare_split.py first.")
    if not os.path.exists(args.mapping):
        raise FileNotFoundError(f"{args.mapping} not found. Run prepare_split.py first.")

    train_loader, val_loader, input_dim, num_tools = load_client_csv(csv_path, args.mapping, batch_size=args.batch_size)
    model = MultiTaskModel(input_dim, num_tools)
    client = CNCClient(model, train_loader, val_loader, local_epochs=args.local_epochs)

    # Use new-style start_client with .to_client() to avoid "return NumPyClient" deprecation
    fl.client.start_client(server_address=f"{args.host}:{args.port}", client=client.to_client())
