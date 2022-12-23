import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.cluster import KMeans

from .autoencoder import Autoencoder, reconstruction_loss
from .auxiliary import vis


def init_weights_xavier(layer):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight)


def dkm_loss(x, reconstruction, embedding, labels, centers, p1: int = 2, p2: int = 2, pow1: int = 2, pow2: int = 2):
    rec_loss = torch.pow(torch.norm(x - reconstruction, p=p1, dim=1), pow1).sum()
    cls_loss = torch.pow(torch.norm(embedding - torch.matmul(labels, centers), p=p2, dim=1), pow2).sum()
    total_loss = rec_loss + cls_loss
    return total_loss, rec_loss, cls_loss


class DKM(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, n_clusters: int, intermediate: list, activation_f: list,
                 is_trained: bool = False):
        super(DKM, self).__init__()

        self.autoencoder = Autoencoder(input_dim, embed_dim, intermediate, activation_f)
        self.autoencoder.apply(init_weights_xavier)

        self.centers = torch.nn.Parameter(nn.init.uniform_(torch.zeros([n_clusters, embed_dim]), a=-1.0, b=1.0),
                                          requires_grad=True)

    def forward(self, x, alpha=1000):
        reconstruction, embedding = self.autoencoder(x)

        distances = torch.cdist(embedding, self.centers)
        labels = F.softmax(-alpha * distances, dim=1)

        return reconstruction, embedding, labels

    def ae_train(self, dataloader, criterion=reconstruction_loss, optimizer=torch.optim.Adam, epochs: int = 100,
                 device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-3, 'betas': (0.9, 0.999)},
                 loss_params: dict = {'p': 2, 'pow': 2}, lr_adjustment: dict = {'rate': 0.1, 'freq': np.inf}):
        self.autoencoder.train_autoencoder(dataloader, criterion, optimizer=optimizer, epochs=epochs,
                                           device=device, optimizer_params=optimizer_params,
                                           loss_params=loss_params, lr_adjustment=lr_adjustment)
        self.autoencoder.is_trained = True

    def fit_finetune(self, dataloader, criterion=dkm_loss, optimizer=torch.optim.Adam, epochs: int = 50,
                     device=torch.device('cpu'), n_init: int = 20,
                     optimizer_params: dict = {'lr': 1e-3, 'betas': (0.9, 0.999)},
                     loss_params: dict = {'p1': 2, 'pow1': 2, 'p2': 2, 'pow2': 2}, vis_freq: int = None):

        km = KMeans(n_clusters=self.centers.shape[0], n_init=n_init). \
            fit(self.autoencoder.encoder(dataloader.dataset.tensors[0]).detach().cpu().numpy())
        self.centers.data = torch.Tensor(km.cluster_centers_)

        if self.autoencoder.is_trained:
            alpha = 1000
        else:
            alpha = 0.1

        self.to(device)
        self.train()
        optimizer = optimizer(self.parameters(), **optimizer_params)



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

                embedding, rec, labels = self(x, alpha)

                loss, r_loss, c_loss = criterion(x, embedding, rec, labels, self.centers, **loss_params)

                overall_loss += loss.item()
                overall_r_loss += r_loss.item()
                overall_c_loss += c_loss.item()

                loss.backward()
                optimizer.step()

            n_datapoints = dataloader.dataset.tensors[0].shape[0]

            if self.autoencoder.is_trained:
                alpha = 1000
            else:
                alpha = (2 ** (1 / (np.log(epoch + 1) + int((epoch + 1) == 1)) ** 2)) * alpha

            if epoch in vis_freq:
                vis(self, dataloader, epoch)

            print("\tEpoch", epoch + 1, "\t AvgLoss: ", np.round(overall_loss / n_datapoints, 5), "\t Rec Loss: ",
                  np.round(overall_r_loss / n_datapoints, 5), "\t Cls Loss: ",
                  np.round(overall_c_loss / n_datapoints, 5),
                  "\t Alpha: ", np.round(alpha, 5))
