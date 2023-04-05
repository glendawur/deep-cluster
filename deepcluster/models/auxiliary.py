import torch
from torch.utils.data import TensorDataset, DataLoader


LABEL_TO_COLOR = {x: c for x, c in enumerate(['#a11a25', '#feadbb', '#3145ee', '#9aaaff', '#e8fd00',
                                              '#51ffbc', '#fe5c0d', '#fe00b9', '#77662c', '#00810a'])}

def std_scaling(x, centering = True):
    x = x - x.mean(axis = 0)
    x = (x - x.mean(axis = 0))/(x.max(axis =0) - x.min(axis =0))
    return x

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.data = x
        self.targets = y

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y