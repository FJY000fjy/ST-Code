import torch 
import numpy as np 
import math
import os 
import json 
from scipy.spatial import KDTree
import h5py 
import scanpy as sc
from torch.utils.data import Dataset
from utils import read_assets_from_h5
from typing import List
import torch.nn.functional as F
from .transforms import normalize_adata
from utils import build_pyramid 


def load_adata(expr_path, genes = None, barcodes = None, normalize_method=normalize_adata):
    adata = sc.read_h5ad(expr_path)
    if barcodes is not None:
        adata = adata[barcodes]
    if normalize_method is not None:
        adata = normalize_method(adata)
    if 'in_tissue' in adata.obs:
        in_tissue = adata.obs['in_tissue'].values.astype(bool)
    else:
        in_tissue = np.ones(adata.shape[0], dtype=bool)
    if genes is not None:
        adata_df = adata[:, genes].to_df()
    else:
        adata_df = adata.to_df()
    return adata_df,in_tissue

class SPData_var:
    features: torch.Tensor | None = None 
    labels: torch.Tensor | None = None 
    coords: torch.Tensor | None = None 
    slice_name:torch.Tensor| None = None 

    def __init__(self, features, 
                 labels, 
                 coords,
                 in_tissue=None,
                 scgpt_feats=None,
                 do_normalize = True,
                 slice_name=None
                 ):
        self.features = features
        self.labels = labels
        self.coords = coords
        self.in_tissue = in_tissue

        self.slice_name = slice_name 
        self.feat_mean = self.features.mean(dim=0,keepdim=True) 
        self.feat_std = self.features.std(dim=0,unbiased=False,keepdim=True) + 1e-6  


        if do_normalize:
            c_min = coords.min(dim=0, keepdim=True)[0] 
            c_max = coords.max(dim=0, keepdim=True)[0] 
            c_center = (c_max + c_min) / 2.0
            c_range = c_max - c_min
            max_range = c_range.max()
            self.coords = (coords - c_center) / (max_range / 2.0 + 1e-6)


        self.pyramid =build_pyramid(self.coords,self.features,self.labels, self.in_tissue)



    def __len__(self):
        return len(self.features)

    def chunk(self, index):
        return SPData_var(
            features=self.features[index],
            labels=self.labels[index],
            coords=self.coords[index],
            in_tissue = self.in_tissue[index],  
            do_normalize=False, 
        )
      
class HESTDatasetPath:
    name: str | None = None
    h5_path: str | None = None
    h5ad_path: str | None = None
    gene_list_path: str | None = None

    def __init__(self, name, h5_path, h5ad_path, gene_list_path, **kwargs):
        self.name = name
        self.h5_path = h5_path
        self.h5ad_path = h5ad_path
        self.gene_list_path = gene_list_path

        for k, v in kwargs.items():
            setattr(self, k, v)
    
class HESTDataset_var(Dataset):
    def __init__(self, dataset: HESTDatasetPath, normalize_method):
        super().__init__()

        self.name = dataset.name
        data_dict, _ = read_assets_from_h5(dataset.h5_path)
        barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
        coords = data_dict["coords"] 
        embeddings = data_dict["embeddings"] 

        with open(os.path.join(dataset.gene_list_path), 'r') as f:
            genes = json.load(f)['genes']
        
        self.gene_list = genes
        labels_df,in_tissue_np = load_adata(dataset.h5ad_path, genes=genes, barcodes=barcodes, normalize_method=normalize_method)
        labels = labels_df.values
        self.sp_dataset = SPData_var(
                features=torch.from_numpy(embeddings).float(),
                labels=torch.from_numpy(labels).float(),
                coords=torch.from_numpy(coords).float(),
                in_tissue = torch.from_numpy(in_tissue_np).bool(),
            )
        
    def __len__(self):
        return 1

    def __getitem__(self,idx):
        return self.sp_dataset


class MultiHESTDataset_var(Dataset):
    def __init__(self, dataset_list: List[HESTDatasetPath], normalize_method,  sample_times=5):
        super().__init__()

        self.dataset_list = dataset_list
        self.sp_datasets = []
        self.n_chunks, self.sample_times = [], sample_times

        for i, dataset in enumerate(self.dataset_list):
            data_dict, _ = read_assets_from_h5(dataset.h5_path)
            barcodes = data_dict["barcodes"].flatten().astype(str).tolist()
            coords = data_dict["coords"]
            embeddings = data_dict["embeddings"]

            with open(os.path.join(dataset.gene_list_path), 'r') as f:
                genes = json.load(f)['genes']

            labels_df,in_tissue_np = load_adata(dataset.h5ad_path, genes=genes, barcodes=barcodes, normalize_method=normalize_method)
            labels = labels_df.values
            self.n_chunks.append(sample_times)
            self.sp_datasets.append(
                SPData_var(
                    features=torch.from_numpy(embeddings).float(),
                    labels=torch.from_numpy(labels).float(),
                    coords=torch.from_numpy(coords).float(),
                    in_tissue = torch.from_numpy(in_tissue_np).bool(),
                )
            )
            self.sp_datasets[-1].slice_name = dataset.name
        
    def __len__(self):
        return sum(self.n_chunks)

    def __getitem__(self, idx):
        for i, n_chunk in enumerate(self.n_chunks):
            if idx < n_chunk:
                return self.sp_datasets[i]
            idx -= n_chunk


def pyg_pyramid_collate_fn(batch_list):
    batched_pyramid = {}
    slide_ids = [sp_data.slice_name for sp_data in batch_list]
    for scale in ['macro','meso','micro']:
        all_coords = []
        all_feats = []
        all_targets = [] 
        batch_vectors = []
        all_in_tissue = []

        for i,sp_data in enumerate(batch_list):
            scale_data = sp_data.pyramid[scale]
            num_nodes = scale_data['coords'].shape[0]
            all_coords.append(scale_data['coords'])
            all_feats.append(scale_data['feats'])
            all_in_tissue.append(scale_data['in_tissue'])
            if scale_data['targets'] is not None:
                all_targets.append(scale_data['targets'])
            batch_vector = torch.full((num_nodes,),i,dtype=torch.long)
            batch_vectors.append(batch_vector)


        batched_pyramid[scale]={
            'coords':torch.cat(all_coords,dim=0),
            'feats':torch.cat(all_feats,dim=0),
            'targets':torch.cat(all_targets,dim=0),
            'in_tissue': torch.cat(all_in_tissue, dim=0), 
            'batch':torch.cat(batch_vectors,dim=0),
        }


    batched_pyramid['slide_ids'] = slide_ids 
    batched_pyramid['feat_mean'] = torch.cat(
        [sp_data.feat_mean for sp_data in batch_list],
        dim=0
    )
    batched_pyramid['feat_std'] = torch.cat(
        [sp_data.feat_std for sp_data in batch_list],
        dim=0
    )
    
        
    return batched_pyramid 






