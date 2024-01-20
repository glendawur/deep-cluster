import torch

# x1 = torch.randint(0, 10, (10, 3, 2)).to(torch.float32)
# x2 = torch.randint(0, 10, (10, 3, 2)).to(torch.float32)

# print(torch.norm(x1 - x2, p = 2, dim = 1))
batch_size = None
auxiliary_dataset = torch.randint(0, 10, (20, 3))

# if isinstance(auxiliary_dataset, torch.Tensor):
#     auxiliary_dataset = torch.utils.data.TensorDataset(auxiliary_dataset)
# if isinstance(auxiliary_dataset, torch.utils.data.Dataset):
#     auxiliary_dataset = torch.utils.data.DataLoader(auxiliary_dataset,
#                                     batch_size=batch_size if batch_size is not None else 256,
#                                     shuffle=True)
    
# print(auxiliary_dataset.__class__)

from time import sleep
import tqdm
import numpy as np

# pbar = tqdm.trange(0, 50, 1, leave=True)
# for i in pbar:
#     sleep(0.1)
#     pbar.set_description(f'UPD {i+1}: Loss 1 = {round(np.random.random_sample(1)[0], 5)}; Loss 2 = {round(np.random.random_sample(1)[0], 5)}')

# device = 'cpu'
# print(isinstance(device, str))

# device = torch.device('cpu')
if True:
    print('Achtung!')