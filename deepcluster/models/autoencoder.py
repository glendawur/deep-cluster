from typing import List, Union

import tqdm
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset


class Autoencoder(torch.nn.Module):
    """
        Base class for autoencoder model
    """
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super(Autoencoder).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.last_config = None
        self.training_history = None
        self.last_device = None

    def encode(self, x):
        """
            Forward X to encoder
        """
        return self.encoder(x)

    def decode(self, x):
        """
            Forward X to decoder
        """
        return self.decoder(x)

    def forward(self, x):
        """
            Forward X through the autoencoder
        """
        x_encoded = self.encode(x)
        x_reconstructed = self.decode(x_encoded)
        return x_encoded, x_reconstructed

    def predict(self, x: Union[Dataset, DataLoader, torch.Tensor],):
        """
        """
        pass 


    def fit(self,
            train_dataset: Union[Dataset, DataLoader, Tensor],
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            optimizer_config: dict = {'lr': 0.001, 
                                      'etas': (0.9, 0.999), 
                                      'eps': 1e-8, 
                                      'weight_decay': 0},
            validation_dataset: Union[Dataset, DataLoader, Tensor] = None,
            test_dataset: Union[Dataset, DataLoader, Tensor] = None,
            batch_size: int = 512,
            epochs: int = 1,
            device: Union[str, torch.device] = 'cpu'):
        """
            Method for the scikit-learn like training process with standard Euclidean distance \
            Reconstruction Loss.

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
        
        # validation and test dataset checks
        for auxiliary_dataset in [validation_dataset, test_dataset]:
            if auxiliary_dataset is not None:
                if isinstance(auxiliary_dataset, Tensor):
                    auxiliary_dataset = TensorDataset(auxiliary_dataset)
                if isinstance(auxiliary_dataset, Dataset):
                    auxiliary_dataset = DataLoader(auxiliary_dataset,
                                                   batch_size=batch_size if batch_size is not None else 256,
                                                   shuffle=True)
        if validation_dataset is not None:
            self.training_history['validation loss'] = []
        if test_dataset is not None:
            self.training_history['test loss'] = []


        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        opt = optimizer([{'params': self.encoder.parameters()},
                         {'params': self.decoder.parameters()}],
                         **optimizer_config)
        
        pbar = tqdm.trange(stop=epochs, leave=True)
        for epoch in pbar:
            output_string = ''
            train_loss = 0
            validation_loss = 0
            # train step
            n_size = 0
            for batch_num, (x_batch) in enumerate(train_dataset):
                opt.zero_grad()

                x_batch = x_batch.to(device)
                _, x_reconstructed = self(x_batch)
                loss = torch.sum(torch.pow(torch.norm(x_batch - x_reconstructed, p = 2, dim=1), 2))
                
                # Compute gradient and perfrom optimization step
                loss.backward()
                opt.step()

                # Saving to logs
                train_loss+=loss.item()
                n_size+=x_batch.shape[0]

            train_loss/=n_size
            output_string = output_string + f'Train Loss: {round(train_loss, 5)} || '
            
            # validation step
            n_size=0
            if validation_dataset is not None:
                with torch.no_grad():
                    for batch_num, (x_batch) in enumerate(validation_dataset):
                        x_batch = x_batch.to(device)

                        _, x_reconstructed = self(x_batch)

                        loss = torch.sum(torch.pow(torch.norm(x_batch - x_reconstructed, p = 2, dim=1), 2))
                        validation_loss+=loss.item()
                        n_size+=x_batch.shape[0]
                    validation_loss/=n_size
                    output_string = output_string + f'Validation Loss: {round(validation_loss, 5)} || '

            pbar.set_description(f'Epoch {epoch}: '+output_string)

            if (self.scheduler_steps is not None) & (epoch in self.scheduler_steps):
#                 self.scheduler.step()

        n_size=0
        test_loss = 0
        if test_dataset is not None:
            with torch.no_grad():
                for batch_num, (x_batch) in enumerate(test_dataset):
                    x_batch = x_batch.to(device)

                    _, x_reconstructed = self(x_batch)

                    loss = torch.sum(torch.pow(torch.norm(x_batch - x_reconstructed, p = 2, dim=1), 2))
                    validation_loss+=loss.item()
                    n_size+=x_batch.shape[0]
                test_loss/=n_size
                output_string = output_string + f'Test Loss: {round(test_loss, 5)}'

        print(f'After {epochs} epochs: ' + output_string)

        self.last_device = device
        self.last_config = {'optimizer': opt, 'epochs': epochs}