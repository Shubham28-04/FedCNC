# utils.py
import torch
import numpy as np

def get_weights(model):
    """Return model weights as a list of numpy arrays matching state_dict order."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]

def set_weights(model, weights):
    """Set model weights from list of numpy arrays (matching state_dict order)."""
    state_dict = model.state_dict()
    new_state = {}
    for (k, v), w in zip(state_dict.items(), weights):
        new_state[k] = torch.from_numpy(w).to(v.device)
    model.load_state_dict(new_state)
    return model
