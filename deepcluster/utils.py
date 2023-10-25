import torch

def nearest_neighbors_kernel(A: torch.Tensor, 
                             n_neighbors: int, 
                             proportional: bool = True,
                             mutual: bool = True, 
                             p: int = 2):
    """
         This function computes nearest neighbors graph kernel for given data matrix or
         distance matrix.

        Parameters
        ----------
            A : torch.Tensor
                Distance (Affinity) matrix if A is two-dimensional and the dimensions have
                the same size. Otherwise, A treated as data matrix. 
            n_neighbors : int
                number of nearest neighbors.
            proportional : bool, optional
                if True, computes proportional nearest neighbors kernel; else returns regular 
                binary nearest neighbor kernel.
            mutual : bool, optional
                if True, computes mutual nearest neighbors kernel.
            p : int, optional
                if A is not distance matrix, p defines the p parameters for Minkowski distance.

        Returns
        -------
            S : torch.Tensor
                Square adjacency matrix for nearest neighbors kernel.
    """
    size = A.shape[0]

    if (len(A.shape) == 2) & (size == A.shape[1]):
        dist = A
    else:
        dist = torch.cdist(A.view(size, -1), p=p)    

    k = min(size-2, n_neighbors)

    sorted_idx = torch.argsort(torch.argsort(dist, dim=1))

    if proportional:
        S = (1-1*(0**torch.maximum((sorted_idx*(sorted_idx < k+1)),
                                     torch.zeros_like(sorted_idx))))/2**(torch.maximum((sorted_idx*(sorted_idx < k+1)) - 1,
                                                                                       torch.zeros_like(sorted_idx)))
    else:
        S = (torch.argsort(torch.argsort(dist, dim=1), dim=1) < k+1).to(dtype = int)
        S = S - torch.diag(torch.diag(S))

    if mutual:
        S = torch.minimum(S, S.T)
    else:
        S = (S + S.T)/2

    return S
