"""
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Callable
import torch
import numpy as np

class BaseKernel(ABC):
    """
        Base abstract class for graph kernel
    """
    __slots__ = ('distance_function', 'distance_parameters', 'kernel_parameters')

    def __init__(self,
                 kernel_parameters = None,
                 distance_function: Callable = None,
                 distance_parameters: Dict = None) -> None:
        self.distance_function = distance_function if distance_function is not None else torch.cdist
        self.distance_parameters = distance_parameters if distance_parameters is not None else dict(p= 2)
        self.kernel_parameters = kernel_parameters

    @abstractmethod
    def compute(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        """
            Compute kernel for the input data
        """
        pass

class StochasticKernel(BaseKernel):
    """
        Base class for stochastic graph kernel
    """
    __slots__ = ('random_distribution', 'parameters_history')

    def __init__(self,
                 kernel_parameters = None,
                 distance_function: Callable = None,
                 distance_parameters: Dict = None,
                 random_distribution: Callable = np.random.uniform) -> None:
        super().__init__(kernel_parameters = kernel_parameters,
                         distance_function = distance_function,
                         distance_parameters = distance_parameters)
        self.random_distribution = random_distribution
        self.parameters_history = []

    @abstractmethod
    def draw_parameters(self):
        pass

    def compute(self,  x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pass

class DeterministicKernel(BaseKernel):
    """
        Base class for graph kernel (deterministic)
    """
    def __init__(self,
                 kernel_parameters = None,
                 distance_function: Callable = None,
                 distance_parameters: Dict = None) -> None:
        super().__init__(kernel_parameters = kernel_parameters,
                         distance_function = distance_function,
                         distance_parameters = distance_parameters)

    def compute(self,  x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        pass

class StochasticKNeighborsKernel(StochasticKernel):
    """
        Stochastic k-nearest neighbors kernel
    """
    def __init__(self,
                 kernel_parameters = None,
                 distance_function: Callable = None,
                 distance_parameters: Dict = None,
                 random_distribution: Callable = np.random.randint,
                 mutual: bool = True,
                 symmetric: bool = True,
                 proportional: bool = True) -> None:
        super().__init__(kernel_parameters = kernel_parameters,
                         distance_function = distance_function,
                         distance_parameters = distance_parameters,
                         random_distribution = random_distribution)
        self.mutual = mutual
        self.symmetric = symmetric
        self.proportional = proportional

    def draw_parameters(self):
        min_k = self.kernel_parameters.get('min_value') if self.kernel_parameters.get('min_value') is not None else 2
        max_k = self.kernel_parameters.get('max_value') if self.kernel_parameters.get('min_value') is not None else 25
        k = self.random_distribution(min_k, max_k, size = 1)[0]
        self.parameters_history.append(k)
        return k

    def compute(self,  x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if y is None:
            y = x
        elif y.shape[1] != x.shape[1]:
            print(f'x {x.shape} and y {y.shape} have different dimensions!')
            return None
        elif (y.shape[0] != x.shape[0]) & self.symmetric:
            print(f'x {x.shape} and y {y.shape} have different shape so kernel matrix cannot be symmetric!')
            return None
        limit = x.shape[0] - 2 
        k = min(limit, self.draw_parameters())

        dist = self.distance_function(x, y, **self.distance_parameters)
        
        if self.proportional:
            sorted_idx = torch.argsort(torch.argsort(dist, dim=1))
            knn = (1-1*(0**torch.maximum((sorted_idx*(sorted_idx < k+1)),
                                              torch.zeros_like(sorted_idx))))/2**(torch.maximum((sorted_idx*(sorted_idx < k+1)) - 1,
                                                                                                torch.zeros_like(sorted_idx)))
        else:
            knn = (dist<=torch.sort(dist, dim=1)[0][:,k].view(-1,1)).to(int)
        
        if self.symmetric:
            if self.mutual:
                kernel_matrix = torch.minimum(knn, knn.T)
            else: 
                kernel_matrix = (knn + knn.T)/2
        else:
            kernel_matrix = knn
        return kernel_matrix

class KNeighbhorsKernel(DeterministicKernel):
    """
        Conventional k-nearest neighbors kernel
    """
    def __init__(self,
                 kernel_parameters = None,
                 distance_function: Callable = None,
                 distance_parameters: Dict = None,
                 mutual: bool = True,
                 symmetric: bool = True,
                 proportional: bool = True) -> None:
        super().__init__(kernel_parameters = kernel_parameters,
                         distance_function = distance_function,
                         distance_parameters = distance_parameters)
        self.mutual = mutual
        self.symmetric = symmetric
        self.proportional = proportional

    def compute(self,  x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        if y is None:
            y = x
        elif y.shape[1] != x.shape[1]:
            print(f'x {x.shape} and y {y.shape} have different dimensions!')
            return None
        elif (y.shape[0] != x.shape[0]) & self.symmetric:
            print(f'x {x.shape} and y {y.shape} have different shape so kernel matrix cannot be symmetric!')
            return None
        limit = x.shape[0] - 2 
        k = min(limit, self.kernel_parameters.get('value') if self.kernel_parameters.get('value') is not None else 25)

        dist = self.distance_function(x, y, **self.distance_parameters)
        
        if self.proportional:
            sorted_idx = torch.argsort(torch.argsort(dist, dim=1))
            knn = (1-1*(0**torch.maximum((sorted_idx*(sorted_idx < k+1)),
                                              torch.zeros_like(sorted_idx))))/2**(torch.maximum((sorted_idx*(sorted_idx < k+1)) - 1,
                                                                                                torch.zeros_like(sorted_idx)))
        else:
            knn = (dist<=torch.sort(dist, dim=1)[0][:,k].view(-1,1)).to(int)
        
        if self.symmetric:
            if self.mutual:
                kernel_matrix = torch.minimum(knn, knn.T)
            else: 
                kernel_matrix = (knn + knn.T)/2
        else:
            kernel_matrix = knn
        return kernel_matrix
