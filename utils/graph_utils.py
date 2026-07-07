import torch 
import numpy as np 


class GraphUtils:
    @staticmethod 
    def get_knn_indices(coords,mask=None,k=8):
        B,N,C=coords.shape 
        dist = torch.cdist(coords,coords) # B,N,N
        diag_mask = torch.eye(N,dtype=torch.bool,device = coords.device)
        dist = dist.masked_fill(diag_mask.unsqueeze(0),float('inf'))
        if mask is not None:
            mask = mask.unsqueeze(1).expand(B,N,N)
            dist = dist.masked_fill(mask,float('inf'))
        real_k = min(k,N-1)
        if real_k <=0:
            return torch.empty((B, N, 0),dtype=torch.long, device=coords.device)
        dist = dist.float()
        _,neighbor_indices = torch.topk(dist,k=real_k,dim=-1,largest=False)
        return neighbor_indices 
        

