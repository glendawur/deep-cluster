import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from itertools import combinations

from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay

from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

class LeNetEmbedding(nn.Module):
    def __init__(self, embed_size: int = 84, final_activation: nn.Module = None):
        super(LeNetEmbedding, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 5), nn.MaxPool2d(2), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.MaxPool2d(2), nn.ReLU())
        self.fc1   = nn.Sequential(nn.Linear(16*4*4, 120), nn.ReLU())
        self.fc2   = nn.Sequential(nn.Linear(120, embed_size))
        if final_activation is not None:
            self.fc2.append(final_activation)

    def forward(self, x):
        conv_out = self.conv2(self.conv1(x))
        linear_in = conv_out.view(conv_out.size(0), -1)
        out = self.fc2(self.fc1(linear_in))
        return out


class DeepSpectralCluster(nn.Module):
    def __init__(self, model: nn.Module, n_clusters: int = 2, alpha: float = 1.):
        super(DeepSpectralCluster, self).__init__()
        self.net = model
        self.centers = torch.nn.Parameter(torch.zeros((n_clusters, list(model.parameters())[-1].shape[0])), requires_grad=True)
        self.alpha = alpha

    def forward(self, x):
        return self.net(x)

    def predict(self, x, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        distances = torch.cdist(x, self.centers)
        labels = F.softmax(-alpha*distances, dim = 1)
        
        return labels


def train_net(model, dataloader_train, dataloader_test =  None, graph_k: int = 16, epochs: int = 200, alpha = 0.1,
              loss_w: dict = {'attr': 0.5, 'rep': 0.5, 'penalty': 0.5, 'cls': 0.5}, name:str = 'DeepSpectral', dataset:str=''):

    writer = SummaryWriter(comment = f'_{name}_{dataset}')

    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cpu')

    z = model(dataloader_test.dataset.data)
    km = KMeans(n_clusters=model.centers.shape[0]).fit(z.detach().numpy())
    model.centers.data = torch.Tensor(km.cluster_centers_)

    model.alpha = alpha

    for epoch in range(epochs):
        overall_loss = 0
        rep_loss = 0
        attr_loss = 0
        cls_loss = 0
        ortho_constraint = 0

        for batch_idx, (x, y) in enumerate(dataloader_train):            
            x, y = x.to(device), y.to(device)

            opt.zero_grad()

            if (graph_k is None) or ():
                graph_k = int(np.sqrt(x.shape[0]))
            
            # build mKNN graph for the attraction loss
            dist = torch.cdist(x.view(x.shape[0], -1), x.view(x.shape[0], -1))
            sigma = torch.sort(dist, dim=1)[0][:,graph_k+1].view(-1, 1)
            knn = (dist <= sigma).to(int)
            W = ((knn + knn.T) == 2).to(int)
            
            ### Next is not working with Higher Dimensions
            # build Delaunay graph for the repulsive loss
            # d = Delaunay(x.view(x.shape[0], -1).detach().numpy())
            # delaunay = torch.zeros((x.shape[0], x.shape[0]))
            # for s in d.simplices:
            #     for (i,j) in combinations(s, 2):
            #         delaunay[i,j]=1.
            # delaunay = ((delaunay + delaunay.T) > 0).to(int)
            # D = ((delaunay + W) == 1).to(int)  

            nknn = (dist > torch.sort(dist, axis = 1)[0][:, graph_k+1].view(-1, 1)).to(int)
            dknn = (dist <= torch.sort(dist, axis = 1)[0][:, int(1.5*graph_k)+1].view(-1, 1)).to(int)
            D = (nknn*dknn).to(int)

            z = model(x)    
                
            attractive_batch = torch.sum(W*torch.pow(torch.cdist(z, z), 2))/2

            repulsive_batch = torch.sqrt(torch.sum((D*torch.pow(torch.cdist(z,z), 2))/2))
            
            penalty_batch = torch.norm((z.T@z) - torch.eye(z.shape[1]), p='fro')
            
            #labels = torch.nn.functional.softmax(-model.alpha*torch.cdist(z, model.centers), dim = 1)
            labels = model.predict(z, model.alpha)
            cls_batch = torch.pow(torch.norm(z - torch.matmul(labels, model.centers), p=2, dim=1), 2).sum()

            loss=-loss_w['rep']*repulsive_batch + loss_w['attr']*attractive_batch \
                + loss_w['penalty']*penalty_batch + loss_w['cls']*cls_batch

            loss.backward()
            
            opt.step()
            
            overall_loss += loss.item()
            rep_loss += repulsive_batch.item()
            attr_loss += attractive_batch.item()
            cls_loss += cls_batch.item()
            ortho_constraint += penalty_batch.item()

        n_datapoints = dataloader_train.dataset.data.shape[0]

        overall_loss /= n_datapoints
        rep_loss /= n_datapoints
        attr_loss /= n_datapoints
        cls_loss /= n_datapoints
        ortho_constraint /= n_datapoints
        

        if dataloader_test is not None:
            sample_x, sample_y = next(iter(dataloader_test))
            z = model(sample_x).detach()
            labels = model.predict(z, alpha = 100).argmax(dim = 1)
            writer.add_scalar('Metrics/Validation_ARI', adjusted_rand_score(labels.numpy(), sample_y.numpy()), epoch)
           
        z = model(dataloader_train.dataset.data).detach()
        labels = model.predict(z, alpha = 100).argmax(dim=1)
        writer.add_scalar('Metrics/Train_ARI', adjusted_rand_score(labels.numpy(),
                                                           dataloader_train.dataset.targets.numpy()), epoch)
        writer.add_scalar('Metrics/UniqueLabels', labels.unique().numpy().shape[0], epoch)

        writer.add_scalar('Alpha', model.alpha, epoch)
        writer.add_scalar("Loss/Total", loss, epoch)
        writer.add_scalar("Loss/Clustering", cls_loss, epoch)
        writer.add_scalar('Loss/Repulsive', rep_loss, epoch)
        writer.add_scalar("Loss/Attractive", attr_loss, epoch)
        writer.add_scalar("Loss/Constraint", ortho_constraint, epoch)
        
        

        writer.flush()
        
        model.alpha = (2 ** (1 / (np.log(epoch + 3) + int((epoch + 3) == 1)) ** 2)) * model.alpha

    z = model(dataloader_train.dataset.data).detach()
    labels = model.predict(z, alpha = 2000).argmax(dim = 1).detach()
    print("\t Final ARI on train: ", adjusted_rand_score(labels, dataloader_train.dataset.targets))
    
    writer.add_graph(model, next(iter(dataloader_train))[0])
    writer.add_embedding(z, metadata = dataloader_train.dataset.targets, tag='Final Latent Space')
    
    if dataloader_test is not None:
        z = model(dataloader_test.dataset.data).detach()
        labels = model.predict(z, alpha = 2000).argmax(dim = 1).detach()
        print("\t Final ARI on test: ", adjusted_rand_score(labels, dataloader_test.dataset.targets))
        


    writer.flush()

    return model