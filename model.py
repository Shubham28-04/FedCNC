# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskModel(nn.Module):
    def __init__(self, input_dim: int, num_tools: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.tool_head = nn.Linear(64, num_tools)   # tool classification (many classes)
        self.machine_head = nn.Linear(64, 2)        # machine failure (binary)
        self.toolfail_head = nn.Linear(64, 2)       # tool failure (binary)

    def forward(self, x):
        h = self.shared(x)
        return {
            "tool": self.tool_head(h),
            "machine": self.machine_head(h),
            "toolfail": self.toolfail_head(h),
        }
