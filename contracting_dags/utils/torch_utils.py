import torch

def to_numpy(x, decimals=None):
    y = x.detach().numpy()
    if decimals is not None:
        y = y.round(decimals=decimals)
    return y

def to_float(x, decimals=None):
    y = float(to_numpy(x))
    if decimals is not None:
        y = round(y, decimals)
    return y

def torch_abs(x):
    return x + 2 * torch.nn.ReLU()(-x)
