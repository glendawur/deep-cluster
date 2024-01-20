from typing import List, Union

import tqdm
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from graph_kernels.kernels import BaseKernel, StochasticKNeighborsKernel

from sklearn.cluster import KMeans

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
    
    @torch.inference_mode
    def transform(self, x):
        return self(x)


    def fit(self, 
            train_dataset: Union[Dataset, DataLoader, Tensor],
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_config: dict = {'lr': 0.001, 
                                      'etas': (0.9, 0.999), 
                                      'eps': 1e-8, 
                                      'weight_decay': 1e-05},
            constraint_weight: float = 1.,
            graph_kernel: BaseKernel = StochasticKNeighborsKernel(kernel_parameters=dict(min_value=2, max_value=30),
                                                                   distance_function=torch.cdist,
                                                                   distance_parameters=dict(p=2),
                                                                   random_distribution=np.random.randint,
                                                                   mutual=True,
                                                                   symmetric=True,
                                                                   proportional=True),
            batch_size: int = 256,
            epochs: int = 100,
            scheduler: torch.optim.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR,
            scheduler_params: dict = dict(gamma=0.1),
            scheduler_steps: List[int] = [20, 40, 60, 80],
            device: Union[str, torch.device] = 'cpu',
            tolerance: float = 1e-7):
        """
        """
        if isinstance(device, str):
            if device == 'cuda' & torch.cuda.is_available():
                device = torch.device(device)
            elif not torch.cuda.is_available():
                device = torch.device('cpu')

        # train dataset checks
        if isinstance(train_dataset, Tensor):
            train_dataset = TensorDataset(train_dataset)
        if isinstance(train_dataset, Dataset):
            train_dataset = DataLoader(train_dataset, 
                                       batch_size=batch_size if batch_size is not None else 256,
                                       shuffle=True)    
        self.training_history = {'training loss': []}

        self.encoder = self.encoder.to(device)

        opt = optimizer([{'params': self.encoder.parameters()}],
                         **optimizer_config)
        opt_scheduler = scheduler(opt, **scheduler_params)

        pbar = tqdm.trange(stop=epochs, leave=True)
        for epoch in pbar:
            output_string = ''
            train_loss = 0
            ratio_cut_loss = 0
            penalty_loss = 0
            n_size = 0

            for batch_id, (x, y, idx) in enumerate(train_dataset):
                opt.zero_grad()
                x,y = x.to(device), y.to(device)

                S = graph_kernel.compute(x, x)
                D = torch.sqrt(S.sum(dim = 1)+tolerance).view(-1,1)

                z = self.encode(x)

                # compute losses
                ratio_cut = torch.sum(torch.pow(S*torch.cdist(z/D, z/D), 2))/2
                penalty = torch.norm(torch.mm(z.T, z) - torch.eye(z.shape[1], device = self.device), p='fro')
                loss = ratio_cut + constraint_weight*penalty

                loss.backward()
                opt.step()
            
                train_loss+=loss.item()
                ratio_cut_loss+=ratio_cut.item()
                penalty_loss+=penalty.item()
                n_size+=x.shape[0]
            
            train_loss/=n_size
            ratio_cut_loss/=n_size
            penalty_loss/=n_size
            output_string = output_string + f'Train Loss: {round(train_loss, 5)} ||\
                  Ratio Cut: {round(ratio_cut_loss, 5)} ||\
                    Penalty: {round(penalty_loss, 5)}'
            
            pbar.set_description(f'Epoch {epoch}: '+ output_string)

            if (scheduler_steps is not None) & (epoch+1 in scheduler_steps):
                opt_scheduler.step()

        self.last_device = device
        self.last_config = {'optimizer': opt, 
                            'opt_scheduler': opt_scheduler,
                            'epochs': epochs,
                            'scheduler_steps': scheduler_steps}
        print('Model Training is finished')