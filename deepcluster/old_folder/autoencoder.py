from typing import Union

import numpy as np
import torch

import torch.nn as nn


def reconstruction_loss(x, rec, p: Union[int, str] = 2, pow: int = 2):
    return torch.pow(torch.norm(x - rec, p=p), pow).sum()


class Encoder(nn.Module):
    """
    MLP Encoder
    """

    def __init__(self, input_dim: int, embed_dim: int, intermediate: list, activations: list):
        super(Encoder, self).__init__()

        self.layers = nn.Sequential()

        self.layers.append(nn.Sequential(nn.Linear(input_dim, intermediate[0]),
                                         activations[0]))

        for i in range(1, min(len(intermediate), len(activations))):
            self.layers.append(nn.Sequential(nn.Linear(intermediate[i - 1], intermediate[i]),
                                             activations[i]))

        self.layers.append(nn.Sequential(nn.Linear(intermediate[min(len(intermediate),
                                                                    len(activations)) - 1],
                                                   embed_dim)))

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    """
    MLP Decoder
    """

    def __init__(self, embed_dim: int, input_dim: int, intermediate: list, activations: list):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential()

        self.layers.append(nn.Sequential(nn.Linear(embed_dim, intermediate[0]),
                                         activations[0]))

        for i in range(1, min(len(intermediate), len(activations))):
            self.layers.append(nn.Sequential(nn.Linear(intermediate[i - 1], intermediate[i]),
                                             activations[i]))

        self.layers.append(nn.Sequential(nn.Linear(intermediate[min(len(intermediate),
                                                                    len(activations)) - 1],
                                                   input_dim)))

    def forward(self, x):
        return self.layers(x)


class Autoencoder(nn.Module):
    """
    Default symmetrical autoencoder
    """

    def __init__(self, input_dim: int, embedding_dim: int, intermediate: list, activations: list):
        super(Autoencoder, self).__init__()

        max_len = min(len(intermediate), len(activations))
        self.is_trained = False
        self.encoder = Encoder(input_dim, embedding_dim, intermediate[:max_len], activations[:max_len])
        self.decoder = Decoder(embedding_dim, input_dim, intermediate[:max_len][::-1], activations[:max_len][::-1])

    def forward(self, x):
        embedding = self.encoder(x)
        return self.decoder(embedding), embedding

    def train_autoencoder(self, dataloader, criterion=reconstruction_loss, optimizer=torch.optim.Adam,
                          epochs: int = 50,
                          device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-3, 'betas': (0.9, 0.999)},
                          loss_params: dict = {'p': 2, 'pow': 2}, lr_adjustment: dict = {'rate': 0.1, 'freq': 15}):
        self.to(device)
        optimizer = optimizer(self.parameters(), **optimizer_params)

        for epoch in range(epochs):
            overall_loss = 0

            if (epoch > 0) & (epoch % lr_adjustment['freq'] == 0):
                for j, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] *= lr_adjustment['rate']
                    print(f'Learning Rate of param group {j} updated to {param_group["lr"]}')

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                rec, _ = self(x)

                loss = criterion(x, rec, **loss_params)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()

            n_datapoints = dataloader.dataset.tensors[0].shape[0]
            print("\tEpoch", epoch + 1, "\AvgLoss: ", np.round(overall_loss / n_datapoints, 5))

    def greedy_pretrain(self, dataloader, criterion=reconstruction_loss, optimizer=torch.optim.Adam,
                        epochs: int = 50,
                        device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-3, 'betas': (0.9, 0.999)},
                        loss_params: dict = {'p': 2, 'pow': 2}, dropout_rate: float = 0.2,
                        lr_adjustment: dict = {'rate': 0.1, 'freq': 15}):
        previous_layers = nn.Sequential()

        for i, (enc, dec) in enumerate(
                zip(list(self.encoder.layers.children()), list(self.decoder.layers.children())[::-1])):
            previous_layers.eval()
            dropout1 = nn.Dropout(p=dropout_rate)
            dropout2 = nn.Dropout(p=dropout_rate)
            enc.to(device)
            dec.to(device)

            optim = optimizer([{'params': enc.parameters()}, {'params': dec.parameters()}], **optimizer_params)

            for epoch in range(epochs):
                loss_value = 0
                length = 0

                if (epoch > 0) & (epoch % lr_adjustment['freq'] == 0):
                    for j, param_group in enumerate(optim.param_groups):
                        param_group['lr'] *= lr_adjustment['rate']
                        #print(f'Learning Rate of param group {j} updated to {param_group["lr"]}')
                    print(f'Learning Rate is updated')

                for batch_idx, (x, y) in enumerate(dataloader):
                    x, y = x.to(device), y.to(device)

                    optim.zero_grad()
                    previous = previous_layers(x)
                    embed = enc(dropout2(previous))
                    rec = dec(dropout1(embed))

                    loss = criterion(rec, previous_layers(x), **loss_params)
                    loss_value += loss.item()
                    length += x.shape[0]

                    loss.backward()
                    optim.step()

                n_datapoints = dataloader.dataset.tensors[0].shape[0]
                #print
            previous_layers.append(enc)
            print(f'Layer {i} trained')
