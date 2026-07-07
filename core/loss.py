import torch.nn as nn
import torch 
from torch_geometric.nn import knn_interpolate,knn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import numpy as np 

class RecLoss_var(nn.Module):
    def __init__(self,gene_weights = None):
        super().__init__()
        self.mse = nn.MSELoss()
        if gene_weights is not None:
            self.register_buffer('gene_weights',gene_weights.float())
        else:
            self.gene_weights = None 

    def forward(self, y_pred, targets,valid_mask = None):
        if valid_mask is not None:
            y_pred = y_pred[valid_mask]
            targets = targets[valid_mask]

        if self.gene_weights is None:
            rec_loss = F.mse_loss(y_pred,targets)
        else:
            se = (y_pred - targets) ** 2  # (N,G)
            w = self.gene_weights.to(se.device).view(1,-1)
            rec_loss = (se * w).mean() 

        return {
            "total_loss": rec_loss,
            "rec_loss": rec_loss.item(),
        }
    

class SSA_Loss(nn.Module):
    def __init__(self, k=6, eps=1e-6, zero_grad_threshold=0.1,
                 gene_weights = None):
        super().__init__()
        self.k = k
        self.eps = eps
        self.zero_grad_threshold = zero_grad_threshold
        if gene_weights is not None:
            self.register_buffer('gene_weights',gene_weights.float())
        else:
            self.gene_weights = None 
    
    def forward(self, pred, target, coords, batch):
        edge_index = knn(coords, coords, k=self.k+1, batch_x=batch, batch_y=batch)
        mask = edge_index[0] != edge_index[1]
        src, dst = edge_index[0][mask], edge_index[1][mask]
        if src.numel() == 0:
            return torch.zeros((), device=pred.device, requires_grad=True)
        coord_diff = coords[dst] - coords[src]
        coord_dist = torch.norm(coord_diff, dim=-1, keepdim=True) + self.eps
        v_true = (target[dst] - target[src]) / coord_dist
        v_pred = (pred[dst]   - pred[src])   / coord_dist
        edge_magnitude = torch.norm(v_true.detach(), p=2, dim=-1) 
        valid_mask = edge_magnitude > self.zero_grad_threshold
        
        if valid_mask.sum() == 0:
            return torch.zeros((), device=pred.device, requires_grad=True)
        
        if torch.rand(1).item() < 0.05:
            print(f"SSA Loss Valid edges: {valid_mask.sum().item()}/{valid_mask.numel()} "
                f"({valid_mask.float().mean().item():.1%})")
        
        v_true = v_true[valid_mask]
        v_pred = v_pred[valid_mask]
        edge_mag = edge_magnitude[valid_mask]
        if self.gene_weights is not None:
            sw = self.gene_weights.to(v_true.device).sqrt().view(1,-1) 
            v_true = v_true *sw 
            v_pred = v_pred *sw 
        
        v_true_norm = F.normalize(v_true, p=2, dim=-1, eps=self.eps)
        v_pred_norm = F.normalize(v_pred, p=2, dim=-1, eps=self.eps)
        cos_sim = (v_true_norm * v_pred_norm).sum(dim=-1) 
        per_edge_loss = 1.0 - cos_sim                      
        edge_weights = edge_mag / (edge_mag.sum() + self.eps)
        loss = (per_edge_loss * edge_weights).sum()
        
        return loss
