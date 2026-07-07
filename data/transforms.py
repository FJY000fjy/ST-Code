import scanpy as sc
import scprep
import scipy
from sklearn.preprocessing import MaxAbsScaler
import numpy as np 


def log1p(adata):
    process_data = adata.copy()
    sc.pp.log1p(process_data)
    return process_data

def stdiff_normalize(adata):
    process_adata = adata.copy()
    sc.pp.normalize_total(process_adata, target_sum=1e4)
    sc.pp.log1p(process_adata)
    process_adata = scale(process_adata)
    if isinstance(process_adata.X, scipy.sparse.csr_matrix):
        process_adata.X.data = process_adata.X.data * 2 - 1
    else:
        process_adata.X = process_adata.X * 2 - 1
    return process_adata

def scVGAE_normalize(adata):
    process_adata = adata.copy()
    process_adata.X = scprep.normalize.library_size_normalize(process_adata.X)
    process_adata.X = scprep.transform.sqrt(process_adata.X)
    return process_adata


def scale(adata):
    scaler = MaxAbsScaler()
    normalized_data = scaler.fit_transform(adata.X.T).T
    adata.X = normalized_data
    return adata



def get_normalize_method(normalize_method, **kwargs):
    if normalize_method is None:
        return None
    elif normalize_method == "log1p":
        return log1p
    elif normalize_method == "stdiff":
        return stdiff_normalize
    elif normalize_method == "scVGAE":
        return scVGAE_normalize
    else:
        raise ValueError(f"Unknown normalize method: {normalize_method}")


def normalize_adata(adata: sc.AnnData, smooth=False) -> sc.AnnData:
    """
    Normalize each spot by total gene counts + Logarithmize each spot
    """
    filtered_adata = adata.copy()
    filtered_adata.X = filtered_adata.X.astype(np.float64)

    if smooth:
        adata_df = adata.to_df()
        for index, df_row in adata.obs.iterrows():
            row = int(df_row['array_row'])
            col = int(df_row['array_col'])
            neighbors_index = adata.obs[((adata.obs['array_row'] >= row - 1) & (adata.obs['array_row'] <= row + 1)) & \
                ((adata.obs['array_col'] >= col - 1) & (adata.obs['array_col'] <= col + 1))].index
            neighbors = adata_df.loc[neighbors_index]
            nb_neighbors = len(neighbors)
            
            avg = neighbors.sum() / nb_neighbors
            filtered_adata[index] = avg
    
    # Logarithm of the expression
    sc.pp.log1p(filtered_adata)
    return filtered_adata
    