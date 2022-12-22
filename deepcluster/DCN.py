from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import KMeans

from autoencoder import Autoencoder, reconstruction_loss
from auxiliary import vis


def dcn_loss(x, reconstruction, embedding, labels, centers, gamma: float = 0.5, p1: int = 2, p2: int = 2, pow1: int = 2,
             pow2: int = 2):
    rec_loss = torch.pow(torch.norm(x - reconstruction, p=p1, dim=1), pow1).mean()
    cls_loss = torch.pow(torch.norm(embedding - torch.matmul(labels, centers), p=p2, dim=1), pow2).mean()
    total_loss = rec_loss + (gamma / 2) * cls_loss
    return total_loss, rec_loss, cls_loss


class DCN(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, n_clusters: int, intermediate: list, activation_f: list,
                 is_trained: bool = False):
        super(DCN).__init__()

        self.autoencoder = Autoencoder(input_dim, embed_dim, intermediate, activation_f)
        self.centers = torch.nn.Parameter(nn.init.uniform_(torch.zeros([n_clusters, embed_dim]), a=-1.0, b=1.0),
                                          requires_grad=True)

    def forward(self, x, p: Union[int, str] = 2):
        embedding = self.autoencoder.encoder(x)
        reconstruction = self.autoencoder.decoder(embedding)

        distance = -torch.cdist(embedding, self.centers, p=p)
        labels = F.one_hot(torch.max(distance, dim=1)[1])
        return reconstruction, embedding, labels.data.detach()

    def ae_pretrain(self, dataloader, criterion=reconstruction_loss, optimizer=torch.optim.Adam,
                    epochs: int = 100, dropout_rate: float = 0,
                    device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-1, 'betas': (0.9, 0.999)},
                    loss_params: dict = {'p': 2, 'pow': 2}, lr_adjustment: dict = {'rate': 0.1, 'freq': np.inf}):
        self.autoencoder.greedy_pretrain(dataloader, criterion=reconstruction_loss, optimizer=optimizer,
                                         epochs=epochs, device=device,
                                         optimizer_params=optimizer_params, loss_params=loss_params,
                                         dropout_rate=dropout_rate, lr_adjustment=lr_adjustment)

    def fit(self, dataloader, criterion=dcn_loss, optimizer=torch.optim.Adam,
            epochs: int = 100, n_init: int = 20,
            device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-1, 'betas': (0.9, 0.999)},
            loss_params: dict = {'p': 2, 'pow': 2}, vis_freq: int = None):
        self.to(device)
        self.train()
        optimizer = optimizer(self.parameters(), **optimizer_params)

        km = KMeans(n_clusters=self.centers.shape[0], n_init=n_init). \
            fit(self.autoencoder.encoder(dataloader.dataset.tensors[0]).detach().cpu().numpy())
        self.centers.data = torch.Tensor(km.cluster_centers_)

        if vis_freq is not None:
            vis_freq = np.arange(0, epochs + 1, vis_freq)
        else:
            vis_freq = list()

        for epoch in range(epochs):
            overall_loss = 0
            overall_r_loss = 0
            overall_c_loss = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                embedding, rec, labels = self(x)

                loss, r_loss, c_loss = criterion(x, embedding, rec, labels, self.centers, **loss_params)

                overall_loss += loss.item()
                overall_r_loss += r_loss.item()
                overall_c_loss += c_loss.item()

                loss.backward()
                optimizer.step()

            if epoch in vis_freq:
                vis(self, dataloader, epoch)

            print("\tEpoch", epoch + 1, "\t AvgLoss: ", np.round(overall_loss / len(dataloader), 5), "\t Rec Loss: ",
                  np.round(overall_r_loss / len(dataloader), 5), "\t Clss Loss: ",
                  np.round(overall_c_loss / len(dataloader), 5))