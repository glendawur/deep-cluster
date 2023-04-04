# import torch


# def adjusted_rand_index(y1, y2):
#     assert y1.shape == y2.shape, 'Different Lenght'
#     if (y1.unique().shape[0] >= y1.shape[0]):
#         return 0, print('Incorrect input y1')
#     elif (y2.unique().shape[0] >= y2.shape): 
#         return 0, print('Incorrect input y2')
    
#     Y1 = (y1 == y1.unique().view(-1,1)).T.to(int)
#     Y2 = (y2 == y2.unique().view(-1,1)).T.to(int)
#     pairing_matrix = torch.einsum('ij,ik -> ijk', Y1, Y2).sum(dim=0)

#     CombN = pairing_matrix*(pairing_matrix - 1)/2