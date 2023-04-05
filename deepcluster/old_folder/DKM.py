import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

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

    def forward(self, x):
        reconstruction, embedding = self.autoencoder(x)

        return reconstruction, embedding
    
    def predict(self, z,  alpha=1000):
        distances = torch.cdist(z, self.centers)
        labels = F.softmax(-alpha * distances, dim=1)
        return labels

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

                rec, embedding = self(x)
                labels = self.predict(embedding, alpha)

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

def train_dkm_net(model, dataloader_train, dataloader_test =  None,  epochs: int = 200,
              loss_w: dict = {'rec': 0.75, 'cls': 0.25}, alpha: float = 0.01, name: str = 'DKM', dataset: str = ''):

    writer = SummaryWriter(comment = f'_{name}_{dataset}')

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cpu')

    _, z = model(dataloader_test.dataset.data)
    km = KMeans(n_clusters=model.centers.shape[0]).fit(z.detach().numpy())
    model.centers.data = torch.Tensor(km.cluster_centers_)

    for epoch in range(epochs):
        overall_loss = 0
        rec_loss = 0
        cls_loss = 0
        
        for batch_idx, (x, y) in enumerate(dataloader_train):            
            x, y = x.to(device), y.to(device)

            opt.zero_grad()

            rec, z = model(x)    
                
            rec_batch = torch.pow(torch.norm(x.view(x.shape[0], -1) - rec.view(rec.shape[0], -1), p=2, dim=1), 2).sum()
            labels = model.predict(z, alpha)
            cls_batch = torch.pow(torch.norm(z - torch.matmul(labels, model.centers), p=2, dim=1), 2).sum()

            loss=loss_w['rec']*rec_batch + loss_w['cls']*cls_batch

            loss.backward()
            
            opt.step()
            
            overall_loss += loss.item()
            rec_loss += rec_batch.item()
            cls_loss += cls_batch.item()
            
        n_datapoints = dataloader_train.dataset.data.shape[0]

        overall_loss /= n_datapoints
        rec_loss /= n_datapoints
        cls_loss /= n_datapoints

        if dataloader_test is not None:
            sample_x, sample_y = next(iter(dataloader_test))
            _, z = model(sample_x)
            y_pred = model.predict(z.detach(), alpha = 100).argmax(dim = 1).detach()
            writer.add_scalar('Metrics/Validation_ARI', adjusted_rand_score(y_pred.numpy(), sample_y.numpy()), epoch)
            
        _, z = model(dataloader_train.dataset.data)
        labels = model.predict(z.detach(), alpha = 100).argmax(dim=1)
        writer.add_scalar('Metrics/Train_ARI', adjusted_rand_score(labels.numpy().shape[0],
                                                              dataloader_train.dataset.targets.numpy()), epoch)
        writer.add_scalar('Metrics/UniqueLabels', labels.unique().numpy(), epoch)
        
        writer.add_scalar('Alpha', alpha,  epoch)
        writer.add_scalar("Loss/Total", loss, epoch)
        writer.add_scalar("Loss/Clustering", cls_loss, epoch)
        writer.add_scalar('Loss/Reconstruction', rec_loss, epoch)
        
        writer.flush()
        
        alpha = (2 ** (1 / (np.log(epoch + 3) + int((epoch + 3) == 1)) ** 2)) * alpha

    _, z = model(dataloader_train.dataset.data)
    y_pred = model.predict(z.detach(), alpha = 100).argmax(dim = 1).detach()
    print("\t Final ARI on train: ", adjusted_rand_score(y_pred.numpy(), dataloader_train.dataset.targets.numpy()))
    
    writer.add_graph(model, next(iter(dataloader_train))[0])
    #writer.add_embedding(z, metadata = dataloader_train.dataset.targets, tag='Final Latent Space')
    
    if dataloader_test is not None:
        _, z = model(dataloader_test.dataset.data)
        y_pred = model.predict(z.detach(), alpha = 100).argmax(dim = 1).detach()
        print("\t Final ARI on test: ", adjusted_rand_score(y_pred.numpy(), dataloader_test.dataset.targets.numpy()))

    writer.flush()

    return model
