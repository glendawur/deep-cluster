import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardCluster(nn.Module):
    def __init__(self, model: nn.Module, n_clusters: int = 2, alpha: float = 1.):
        super(FeedForwardCluster, self).__init__()
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


class AutoencoderCluster(nn.Module):
    def __init__(self, model: nn.Module, n_clusters: int = 2, alpha: float = 1.):
        super(AutoencoderCluster, self).__init__()
        self.net = model
        self.centers = torch.nn.Parameter(torch.zeros((n_clusters, list(model.encoder.parameters())[-1].shape[0])), requires_grad=True)
        self.alpha = alpha

    def forward(self, x):
        embedding = self.net.encoder(x)
        reconstruction = self.net.decoder(embedding)
        return reconstruction, embedding

    def predict(self, z, alpha: float = None):
        if alpha is None:
            alpha = self.alpha
        distances = torch.cdist(z, self.centers)
        labels = F.softmax(-alpha*distances, dim = 1)
        return labels