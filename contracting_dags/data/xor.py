import torch
import numpy as np


class XORData(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        x = torch.Tensor(np.random.randint(2, size=2))
        y = torch.Tensor([1.0]) if x.sum() == 1 else torch.Tensor([0.0])
        return x, y

    @staticmethod
    def one_of_each():
        x = torch.Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        y = torch.Tensor(np.array([0, 1, 1, 0]))
        return x, y
