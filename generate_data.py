# generate_data.py
import numpy as np
import os

def make_signal(seq_len=1024, freq=20.0, amp=1.0, noise=0.1, spike=False):
    t = np.linspace(0, 1, seq_len)
    x = amp * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
    x += noise * np.random.randn(seq_len)
    if spike:
        pos = np.random.randint(seq_len//4, 3*seq_len//4)
        width = np.random.randint(1, 30)
        x[pos:pos+width] += amp * (1.5 + np.random.rand())
    return x.astype(np.float32)

def make_client_data(num_samples=200, seq_len=1024, fault_prob=0.3):
    X, y = [], []
    for _ in range(num_samples):
        if np.random.rand() < fault_prob:
            X.append(make_signal(seq_len=seq_len, freq=np.random.uniform(5,80),
                                 amp=np.random.uniform(0.8,1.5), noise=0.2, spike=True))
            y.append(1)
        else:
            X.append(make_signal(seq_len=seq_len, freq=np.random.uniform(5,80),
                                 amp=np.random.uniform(0.6,1.2), noise=0.08, spike=False))
            y.append(0)
    X = np.stack(X)    # shape (N, seq_len)
    y = np.array(y, dtype=np.int64)
    # shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    # train/val split 80/20
    split = int(0.8 * len(y))
    return X[:split], y[:split], X[split:], y[split:]

def generate(num_clients=4, out_dir="data", samples_per_client=200, seq_len=1024):
    os.makedirs(out_dir, exist_ok=True)
    for cid in range(num_clients):
        fault_prob = 0.15 + 0.2 * (cid % 3)  # vary fault ratio
        X_train, y_train, X_val, y_val = make_client_data(samples_per_client, seq_len, fault_prob)
        np.savez_compressed(os.path.join(out_dir, f"client_{cid}.npz"),
                            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
        print(f"Saved client_{cid}.npz (train {len(y_train)} val {len(y_val)})")

if __name__ == "__main__":
    generate(num_clients=4, out_dir="data", samples_per_client=200, seq_len=1024)
