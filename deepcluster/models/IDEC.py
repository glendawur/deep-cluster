import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from sklearn.cluster import KMeans

from .autoencoder import Autoencoder, reconstruction_loss
from .auxiliary import vis
from .DKM import init_weights_xavier

def init_weights_normal(layer):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(layer.weight)


def idec_loss(x, reconstruction, p, q, gamma: float = 0.1, p1: int = 2, pow1=2):
    loss_kl = torch.sum(p * torch.log(p / (q + 1e-6)), dim=1).sum()
    loss_r = torch.pow(torch.norm(x - reconstruction, p=p1, dim=1), pow1).sum()
    return loss_r + gamma * loss_kl, loss_r, loss_kl


class IDEC(nn.Module):
    """
    MLP Decoder
    """

    def __init__(self, input_dim: int, embed_dim: int, n_clusters: int, intermediate: list, activation_f: list,
                 alpha=1., is_trained: bool = False):
        super(IDEC, self).__init__()

        self.autoencoder = Autoencoder(input_dim, embed_dim, intermediate, activation_f)

        self.autoencoder.apply(init_weights_xavier)

        self.autoencoder.is_trained = is_trained

        self.centers = torch.nn.Parameter(nn.init.uniform_(torch.zeros([n_clusters, embed_dim]), a=-1.0, b=1.0),
                                          requires_grad=True)

    def forward(self, x, alpha: float = 2):
        reconstruction, embedding = self.autoencoder(x)
        # compute q -> NxK
        q = 1.0 / (1.0 + torch.sum((embedding.unsqueeze(1) - self.centers) ** 2, dim=2) / alpha)
        q = q ** (alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return reconstruction, embedding, q

    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def ae_train(self, dataloader, criterion=reconstruction_loss, optimizer=torch.optim.Adam,
                 epochs: int = 100, pretrain_epochs: int = 100, dropout_rate: float = 0.0,
                 device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-3, 'betas': (0.9, 0.999)},
                 loss_params: dict = {'p': 2, 'pow': 2}):
        self.autoencoder.greedy_pretrain(dataloader, criterion=reconstruction_loss, optimizer=optimizer,
                                         epochs=pretrain_epochs, device=device,
                                         optimizer_params=optimizer_params, loss_params=loss_params,
                                         dropout_rate=dropout_rate)

        self.autoencoder.train_autoencoder(dataloader, criterion, optimizer=optimizer, epochs=epochs,
                                           device=device, optimizer_params=optimizer_params,
                                           loss_params=loss_params)
        self.autoencoder.is_trained = True

    def fit_finetune(self, dataloader, criterion=idec_loss, optimizer=torch.optim.Adam, epochs: int = 50,
                     update_interval: int = 1, tol: float = 1e-3, n_init: int = 20,
                     device=torch.device('cpu'), optimizer_params: dict = {'lr': 1e-3, 'betas': (0.9, 0.999)},
                     loss_params: dict = {'gamma': 0.1, 'p1': 2, 'pow1': 2},
                     vis_freq: int = None):
        self.to(device=device)
        self.train()
        optimizer = optimizer(self.parameters(), **optimizer_params)

        if vis_freq is not None:
            vis_freq = np.arange(0, epochs + 1, vis_freq)
        else:
            vis_freq = list()

        km = KMeans(n_clusters=self.centers.shape[0], n_init=n_init). \
            fit(self.autoencoder.encoder(dataloader.dataset.tensors[0]).detach().cpu().numpy())
        self.centers.data = torch.Tensor(km.cluster_centers_)
        y_old = km.predict(self.autoencoder.encoder(dataloader.dataset.tensors[0]).detach().cpu().numpy())

        self.train()

        for epoch in range(epochs):

            if epoch % update_interval == 0:
                _, _, q = self(dataloader.dataset.tensors[0])
                p = self.target_distribution(q).data

                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_old).astype(np.float32) / y_pred.shape[0]

                my_dataset = TensorDataset(dataloader.dataset.tensors[0],
                                           dataloader.dataset.tensors[1],
                                           p)
                dataloader = DataLoader(my_dataset, batch_size=dataloader.batch_size, shuffle=True)

                y_old = y_pred

                if epoch > 0 and delta_label < tol:
                    print(f'delta label is {delta_label}. Convergence reported at epoch {epoch}.')
                    break

            overall_loss = 0
            overall_r_loss = 0
            overall_c_loss = 0

            for batch_idx, (x, _, p) in enumerate(dataloader):
                x, p = x.to(device), p.to(device)

                optimizer.zero_grad()

                rec, _, q = self(x)
                loss, loss_r, loss_kl = criterion(x, rec, p, q, **loss_params)

                overall_loss += loss.item()
                overall_r_loss += loss_r.item()
                overall_c_loss += loss_kl.item()

                loss.backward()
                optimizer.step()

            if epoch in vis_freq:
                vis(self, dataloader, epoch)

            print("\tEpoch", epoch + 1, "\t AvgLoss: ",
                  np.round(overall_loss / dataloader.dataset.tensors[0].shape[0], 5),
                  "\t Rec Loss: ", np.round(overall_r_loss / dataloader.dataset.tensors[0].shape[0], 5),
                  "\t KL Loss: ", np.round(overall_c_loss / dataloader.dataset.tensors[0].shape[0], 5))
