import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DistanceWeightedHypergraphLayer
from torch_geometric.nn import knn_interpolate,global_mean_pool 
import torch
import torch.nn as nn
import math
from torch_geometric.nn import knn
import numpy as np


class KnowledgeDrivenSpotGeneDecoder(nn.Module):
    def __init__(self, hidden_dim, gene_embed_dim,n_heads=4):
        super().__init__()
        self.spot_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.gene_proj = nn.Sequential(
            nn.Linear(gene_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )       
        self.gene_prior_proj = nn.Linear(1, hidden_dim)
        self.relation_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3), 
            nn.Linear(hidden_dim // 2, 1)
        )
        self.nonlinear_alpha = nn.Parameter(torch.tensor([0.01]))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, spot_feat, gene_embed):
        N = spot_feat.size(0)
        G = gene_embed.size(0)
        S = self.spot_proj(spot_feat)       # (N, H)
        Gene_H = self.gene_proj(gene_embed) # (G, H)
        gene_sim = F.normalize(gene_embed, dim=-1) @ F.normalize(gene_embed, dim=-1).T
        gene_popularity = gene_sim.mean(dim=0).unsqueeze(-1)
        pop_bias = self.gene_prior_proj(gene_popularity)
        Gene_H = Gene_H + pop_bias
        S_norm = F.normalize(S, p=2, dim=-1)  # [N,H]
        Gene_H_norm = F.normalize(Gene_H, p=2, dim=-1) # [G,H] 
        linear_logits = torch.matmul(S_norm, Gene_H_norm.T) 
        pair_feat = S_norm.unsqueeze(1) * Gene_H_norm.unsqueeze(0)
        nonlinear_logits = self.relation_decoder(pair_feat).squeeze(-1)
        raw_logits = linear_logits + self.nonlinear_alpha * nonlinear_logits
        logit_scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        gene_logits = raw_logits * logit_scale
        
        nonlinear_logits = linear_logits
        
        return gene_logits,linear_logits,nonlinear_logits
    


  
class HypergraphScaleBlock(nn.Module):
    def __init__(self,
                 feature_dim=1024,
                 output_dim=50,
                 hidden_dim=256,
                 num_layers = 8,
                 skips=[4],
                 n_neighbors=8,
                 dropout_rate=0.2,
                 scgpt_embedding_matrix = None, 
                 gene_embed_dim = 512
                 ):
        super().__init__()
        
        self.num_layers = num_layers 
        self.skips = skips 
        self.dropout_rate = dropout_rate 
        self.gene_embed_dim = gene_embed_dim

        assert scgpt_embedding_matrix is not None, "scgpt_embedding_matrix is None"
        self.register_buffer('fixed_gene_embeddings',scgpt_embedding_matrix)

        self.hypergraph_layer = DistanceWeightedHypergraphLayer(
            in_channels=feature_dim,
            out_channels=feature_dim,
            k=n_neighbors,
            hgat_dropout_rate=0.0,
        )


        self.neighbor_weight = nn.Parameter(torch.tensor([1.0]))
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim,hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=self.dropout_rate)
        )

        self.spatial_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.spatial_layers.append(nn.Linear(hidden_dim,hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.spot_gene_decoder = KnowledgeDrivenSpotGeneDecoder(
            hidden_dim=hidden_dim,
            gene_embed_dim=self.gene_embed_dim,
            n_heads=4,
        )
            




    def forward(self,img_features,coords,batch):
        neighbor_context_features = self.hypergraph_layer(img_features,coords,batch=batch)
        context_img_features = img_features + self.neighbor_weight*neighbor_context_features 
        base_feat = self.feature_proj(context_img_features)
        x = base_feat 
        for i in range(self.num_layers):
            identity = x 
            h = self.spatial_layers[i](x)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h,p=0.1,training=self.training)
            x = identity + h

        y_final,y_part1,y_part2 = self.spot_gene_decoder(
            x,
            self.fixed_gene_embeddings,
        )
        gate_weights = None 
            
        return y_final,x,gate_weights,y_part1,y_part2


class SlopeST(nn.Module):
    def __init__(self,
                 feature_dim=1024,               
                 output_dim=50,         
                 hidden_dim=256,        
                 micro_n_neighbors = 8,  
                 scgpt_embedding_matrix = None,
                 gene_embed_dim=512,

                 ):            
        super().__init__()

        assert scgpt_embedding_matrix is not None , "scgpt_embedding_matrix is None"

        self.macro_block = HypergraphScaleBlock(
            feature_dim=feature_dim, 
            output_dim=output_dim,
            hidden_dim=hidden_dim, 
            num_layers=4,                
            skips=[],                   
            n_neighbors=16,              
            dropout_rate = 0.2,
            scgpt_embedding_matrix=scgpt_embedding_matrix,
            gene_embed_dim=gene_embed_dim,
        )
        
        self.meso_feature_fusion = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU()
        )


        self.meso_block = HypergraphScaleBlock(
            feature_dim=feature_dim, 
            output_dim=output_dim,
            hidden_dim=hidden_dim, 
            num_layers=6,               
            skips=[3],                  
            n_neighbors=8,              
            dropout_rate = 0.2,
            scgpt_embedding_matrix=scgpt_embedding_matrix, 
            gene_embed_dim=gene_embed_dim,
        )
        

        self.micro_feature_fusion = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU()
        )

        self.micro_block = HypergraphScaleBlock(
            feature_dim=feature_dim,  
            output_dim=output_dim,
            hidden_dim=hidden_dim, 
            num_layers=8,           
            skips=[4],              
            n_neighbors=micro_n_neighbors,         
            dropout_rate = 0.2,
            scgpt_embedding_matrix=scgpt_embedding_matrix,
            gene_embed_dim=gene_embed_dim,
            
        )


        self.meso_query_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU() 
        )
        
        self.micro_query_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )

        self.macro_to_meso_fusion = CrossScalePriorFusion(
            hidden_dim=hidden_dim, 
            gene_dim=output_dim, 
            n_heads=4, 
            k=8
        )
        
        self.meso_to_micro_fusion = CrossScalePriorFusion(
            hidden_dim=hidden_dim, 
            gene_dim=output_dim, 
            n_heads=4, 
            k=5 
        )

    def forward(self, batched_pyramid):
        macro_dict = batched_pyramid['macro']
        meso_dict  = batched_pyramid['meso']
        micro_dict = batched_pyramid['micro']

        macro_y, macro_hidden, macro_gate,_,_ = self.macro_block(
            img_features = macro_dict['feats'], 
            coords = macro_dict['coords'], 
            batch = macro_dict['batch'],
        )

        meso_coarse_context = self.macro_to_meso_fusion(
            fine_hidden=self.meso_query_projection(meso_dict['feats']), 
            fine_coords = meso_dict['coords'],
            fine_batch=meso_dict['batch'],
            coarse_hidden=macro_hidden, 
            coarse_pred = macro_y,        
            coarse_coords = macro_dict['coords'],
            coarse_batch=macro_dict['batch'],
        )
        
        meso_fused_feats = self.meso_feature_fusion(torch.cat([meso_dict['feats'], meso_coarse_context], dim=-1))
        meso_y, meso_hidden, meso_gate,_,_ = self.meso_block(
            img_features = meso_fused_feats,
            coords = meso_dict['coords'],
            batch = meso_dict['batch'],
        )
        
        micro_coarse_context = self.meso_to_micro_fusion(
            fine_hidden=self.micro_query_projection(micro_dict['feats']),
            fine_coords=micro_dict['coords'],
            fine_batch=micro_dict['batch'],
            coarse_hidden=meso_hidden,
            coarse_pred=meso_y,           
            coarse_coords=meso_dict['coords'],
            coarse_batch=meso_dict['batch'],
        )


        micro_fused_feats = self.micro_feature_fusion(torch.cat([micro_dict['feats'], micro_coarse_context], dim=-1))
        
        micro_y, micro_hidden, micro_gate,y_part1,y_part2 = self.micro_block(
            img_features = micro_fused_feats,
            coords = micro_dict['coords'],
            batch = micro_dict['batch'],
        )
        
        return macro_y, meso_y, micro_y, micro_gate,y_part1,y_part2



class CrossScalePriorFusion(nn.Module):
    def __init__(self, hidden_dim=256, gene_dim=50, n_heads=4, k=8,               
                 ):
        super().__init__()
        self.k = k
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        assert hidden_dim % n_heads == 0, "hidden_dim % n_heads != 0"

        
        self.coarse_encoder = nn.Sequential(
            nn.Linear(hidden_dim + gene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU()
        )
        

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.dist_bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, n_heads)
        )

        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(self,
                fine_hidden,      # (N_fine, hidden_dim)
                fine_coords,      # (N_fine, 2)
                fine_batch,       # (N_fine,)
                coarse_hidden,    # (N_coarse, hidden_dim)
                coarse_pred,      # (N_coarse, gene_dim) 
                coarse_coords,    # (N_coarse, 2)
                coarse_batch,     # (N_coarse,)
                ):    
        
    
        coarse_combined = self.coarse_encoder(
            torch.cat([coarse_hidden, coarse_pred], dim=-1)
        )
        
        fused_features_list = []
        
        for slide_id in fine_batch.unique():
            f_mask = (fine_batch == slide_id)
            c_mask = (coarse_batch == slide_id)
            
            fh = fine_hidden[f_mask]      # (n_fine, D)
            fc = fine_coords[f_mask]      # (n_fine, 2)
            ch = coarse_combined[c_mask]  # (n_coarse, D)
            cc = coarse_coords[c_mask]    # (n_coarse, 2)
            
            n_fine, n_coarse = fh.shape[0], ch.shape[0]
            current_k = min(self.k, n_coarse)
            
            assign = knn(x=cc, y=fc, k=current_k)
            neighbor_feats = ch[assign[1]].view(n_fine, current_k, -1)   # (n_fine, k, D)
            neighbor_coords = cc[assign[1]].view(n_fine, current_k, 2)   # (n_fine, k, 2)
              
            Q = self.q_proj(fh).view(n_fine, 1, self.n_heads, self.head_dim) 
            K = self.k_proj(neighbor_feats).view(n_fine, current_k, self.n_heads, self.head_dim) 

            Q = self.q_norm(Q)
            K = self.k_norm(K)
            Q = Q.transpose(1, 2) # (n_fine, heads, 1, head_dim)
            K = K.transpose(1,2) # (n_fine,heads,k,head_dim)
            
            V = self.v_proj(neighbor_feats).view(n_fine, current_k, self.n_heads, self.head_dim).transpose(1, 2)     # (n_fine, heads, k, head_dim)
            
            raw_logits = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.head_dim) # (n_fine, heads, 1, k)
            dist = ((fc.unsqueeze(1) - neighbor_coords) ** 2).sum(-1, keepdim=True).sqrt() 
            dist_bias = self.dist_bias_mlp(dist).permute(0, 2, 1).unsqueeze(2) 
            raw_logits = raw_logits + dist_bias
                
            attn_weights = torch.softmax(raw_logits, dim=-1) # (n_fine, heads, 1, k)
            
            attn_out = torch.matmul(attn_weights, V) # (n_fine, heads, 1, head_dim)
            attn_out = attn_out.transpose(1, 2).contiguous().view(n_fine, -1) # (n_fine, D)
            attn_out = self.out_proj(attn_out)
            
            gate = self.gate(torch.cat([fh, attn_out], dim=-1))
            fused = self.norm(fh + gate * attn_out)
             
            fused_features_list.append(fused)
            
        return torch.cat(fused_features_list, dim=0)
