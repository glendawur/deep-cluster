from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset

import numpy as np

from sklearn.cluster import KMeans
from torch.utils.data.dataset import T_co

from .autoencoder import Autoencoder, reconstruction_loss
from .auxiliary import vis

from torch.utils.data import Dataset


class DCNDataset(Dataset):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index], index


def dcn_loss(x, reconstruction, embedding, labels, centers, gamma: float = 0.5, p1: int = 2, p2: int = 2, pow1: int = 2,
             pow2: int = 2):
    rec_loss = torch.pow(torch.norm(x - reconstruction, p=p1, dim=1), pow1).mean()
    cls_loss = torch.pow(torch.norm(embedding - torch.matmul(labels.float(), centers), p=p2, dim=1), pow2).mean()
    total_loss = rec_loss + (gamma / 2) * cls_loss
    return total_loss, rec_loss, cls_loss


class DCN(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, n_clusters: int, intermediate: list, activation_f: list,
                 is_trained: bool = False):
        super(DCN, self).__init__()

        self.autoencoder = Autoencoder(input_dim, embed_dim, intermediate, activation_f)
        self.centers = torch.nn.Parameter(nn.init.uniform_(torch.zeros([n_clusters, embed_dim]), a=-1.0, b=1.0),
                                          requires_grad=False)

        self.count = torch.ones(n_clusters, dtype=torch.int)

    def forward(self, x):
        embedding = self.autoencoder.encoder(x)
        reconstruction = self.autoencoder.decoder(embedding)

        return reconstruction, embedding

    def predict(self, x, one_hot: bool = False, p = 2):
        distance = torch.cdist(x, self.centers, p=p)
        label = distance.argmin(dim=1)
        if one_hot:
            return F.one_hot(label, num_classes=self.centers.data.shape[0])
        else:
            return label

    def ae_train(self, dataloader, criterion=reconstruction_loss, optimizer=torch.optim.Adam,
                    epochs: int = 100, dropout_rate: float = 0,
                    device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-1, 'betas': (0.9, 0.999)},
                    loss_params: dict = {'p': 2, 'pow': 2}, lr_adjustment: dict = {'rate': 0.1, 'freq': np.inf}):
        self.autoencoder.greedy_pretrain(dataloader, criterion=reconstruction_loss, optimizer=optimizer,
                                         epochs=epochs, device=device,
                                         optimizer_params=optimizer_params, loss_params=loss_params,
                                         dropout_rate=dropout_rate, lr_adjustment=lr_adjustment)

    def batch_km(self, x: torch.Tensor, p=2):
        # update assignment
        distance = torch.cdist(x, self.centers, p=p)
        new_assignment = distance.argmin(dim=1)

        # update centers
        for obs, label in zip(x, new_assignment):
            self.count[label.item()] += 1
            eta = 1 / self.count[label.item()]
            self.centers.data[label.item()] = self.centers.data[label.item()] * (1 - eta) + eta * obs

        return new_assignment

    def fit_finetune(self, dataloader, criterion=dcn_loss, optimizer=torch.optim.Adam,
            epochs: int = 100, n_init: int = 20,
            device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-1, 'betas': (0.9, 0.999)},
            loss_params: dict = {'gamma': 0.5, 'p1': 2, 'pow1': 2, 'p2': 2, 'pow2': 2}, vis_freq: int = None):

        self.count = 100 * torch.ones(self.count.shape[0], dtype=torch.int)
        X = dataloader.dataset.tensors[0]
        aux_dataloader = DataLoader(DCNDataset(X), batch_size=256)

        self.to(device)
        self.train()
        optimizer = optimizer(self.parameters(), **optimizer_params)

        km = KMeans(n_clusters=self.centers.shape[0], n_init=n_init). \
            fit(self.autoencoder.encoder(X).detach().cpu().numpy())
        self.centers.data = torch.Tensor(km.cluster_centers_)
        Y = km.predict(self.autoencoder.encoder(X).detach().cpu().numpy())

        if vis_freq is not None:
            vis_freq = np.arange(0, epochs + 1, vis_freq)
        else:
            vis_freq = list()

        for epoch in range(epochs):
            overall_loss = 0
            overall_r_loss = 0
            overall_c_loss = 0

            for batch_idx, (x, idx) in enumerate(aux_dataloader):
                x = x.to(device)

                optimizer.zero_grad()

                # stochastic gradient step
                rec, embedding = self(x)
                labels = torch.Tensor(Y[idx])
                loss, r_loss, c_loss = criterion(x, rec, embedding, F.one_hot(labels.to(torch.int64),
                                                                              num_classes=self.centers.shape[0]),
                                                 self.centers, **loss_params)

                overall_loss += loss.item()
                overall_r_loss += r_loss.item()
                overall_c_loss += c_loss.item()

                loss.backward()
                optimizer.step()

                # alternate optimization
                new_labels = self.batch_km(embedding)
                Y[idx] = new_labels.numpy()

            if epoch in vis_freq:
                vis(self, dataloader, epoch)

            print("\tEpoch", epoch + 1, "\t AvgLoss: ", np.round(overall_loss / len(dataloader), 5), "\t Rec Loss: ",
                  np.round(overall_r_loss / len(dataloader), 5), "\t Cls Loss: ",
                  np.round(overall_c_loss / len(dataloader), 5))
