from typing import List, Union

import tqdm
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

class RobustSpectralMap(torch.nn.Module):
    """
        Class for Robust Spectral Map model
    """

    def __init__(self,
                 encoder: torch.nn.Module,
                 ) -> None:
        super(RobustSpectralMap).__init__()
        self.encoder = encoder
        
    def encode(self, x):
        """
            Forward X to encoder
        """
        return self.encoder(x)

    def forward(self, x):
        """
            Forward X
        """
        return self.encoder(x)
    
    def fit(self, 
            train_dataset: Union[Dataset, DataLoader, Tensor],
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_config: dict = {'lr': 0.001, 
                                      'etas': (0.9, 0.999), 
                                      'eps': 1e-8, 
                                      'weight_decay': 0},
            graph_kernel = None,
            batch_size: int = 512,
            epochs: int = 1,
            device: Union[str, torch.device] = 'cpu'):
        
        
# class RobustSpectralMap:
#     """
#     """
#     def __init__(self, 
#                  net,
#                  optim_config: dict = {'lr': 0.001, 
#                                        'weight_decay': 1e-05},
#                  batch_size: int = 512,
#                  epochs: int = 100,
#                  k_range: (int, int) = (2,25),
#                  kernel_config: dict = {'proporional': True,
#                                         'mutual': True},
#                  l: float = 1.,
#                  scheduler_timesteps: list = None,
#                  device: str = None) -> None:
#         """
        
#         """
#         if device is None:
#             self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
#         self.net = net.to(device)
#         self.optim = torch.optim.Adam(self.net.parameters(), **optim_config)
#         self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, gamma = 0.1)

#         self.kernel_config = kernel_config
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.k_range = k_range
#         self.l = l
#         self.scheduler_steps = scheduler_timesteps
        


#     def fit(self, X: torch.Tensor, y: torch.Tensor = None):
#         """
#         """
#         data_loader = DataLoader(TensorDataset(X), shuffle=True, batch_size = self.batch_size)
#         self.net = self.net.to(device=self.device)

#         t = trange(self.epochs, leave=True)
#         for epoch in t:
#             overall_loss = 0.0
#             rc_loss = 0.0
#             pen_loss = 0.0
#             for x_batch in data_loader:
#                 self.optim.zero_grad()
                
#                 x_batch = x_batch.to(self.device)
                
#                 # compute kernel and compute Laplacian related matrices
#                 n_neighbors = np.random.randint(self.k_range[0], self.k_range[1]+1, 1) 
#                 S = nearest_neighbors_kernel(x_batch, **self.kernel_config)
#                 if S.device != x_batch.device:
#                     S = S.to(device = x_batch.device)
#                 D = torch.sqrt(S.sum(dim=1)+1e-7).view(-1,1)

#                 # network parameters should be taken into account
#                 z = self.net(x_batch)

#                 # compute losses
#                 ratio_cut = torch.sum(torch.pow(S*torch.cdist(z/D, z/D), 2))/2
#                 penalty = torch.norm(torch.mm(z.T, z) - torch.eye(z.shape[1], device = self.device), p='fro')

#                 loss = ratio_cut + self.l*penalty
#                 loss.backward()
#                 self.optim.step()

#                 overall_loss += loss.item()
#                 rc_loss += ratio_cut.item()
#                 pen_loss += penalty.item()

#             if (self.scheduler_steps is not None) & (epoch in self.scheduler_steps):
#                 self.scheduler.step()

#     def save(self, path: str):
#         """
#         """
#         torch.save(self.net, path)             

#     def predict(self, X: torch.Tensor, device = torch.device('cpu')) -> torch.Tensor:
#         """
#         """
#         self.net = self.net.to(device=device)
#         return self.net(X)

