import torch
import numpy as np


def build_scgpt_embedding_matrix(
    scgpt_embeddings_dict: dict,
    gene_name_list: list,
    expected_embed_dim: int | None = 512,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    missing_genes = [gene_name for gene_name in gene_name_list if gene_name not in scgpt_embeddings_dict]
    if missing_genes:
        preview = ", ".join(missing_genes[:10])
        suffix = "..." if len(missing_genes) > 10 else ""
        raise KeyError(
            f"scgpt_embeddings_dict is missing {len(missing_genes)} genes from gene_name_list."
            f"{preview}{suffix}"
        )

    embeddings = []
    for gene_name in gene_name_list:
        emb = torch.as_tensor(scgpt_embeddings_dict[gene_name], dtype=dtype, device=device)
        if emb.ndim != 1:
            raise ValueError(f"Gene [{gene_name}]'s embedding must be 1D,current shape={tuple(emb.shape)}")
        if expected_embed_dim is not None and emb.shape[0] != expected_embed_dim:
            raise ValueError(
                f"Gene [{gene_name}]'s embedding dim mest be {expected_embed_dim}."
                f"Currently is {emb.shape[0]}."
            )
        embeddings.append(emb)

    return torch.stack(embeddings, dim=0)



def get_imputed_embedding(
    corr_matrix: np.ndarray, 
    target_gene_idx: int, 
    gene_name_list: list, 
    scgpt_embeddings_dict: dict, 
    k: int = 3, 
    pcc_threshold: float = 0.3
) -> torch.Tensor:
    
    target_gene_name = gene_name_list[target_gene_idx]
    corrs = corr_matrix[target_gene_idx].copy() 
    
    valid_candidates = []

    for j, gene_name in enumerate(gene_name_list):
        if j == target_gene_idx:
            continue
        if gene_name not in scgpt_embeddings_dict:
            continue
        if corrs[j] < pcc_threshold:
            continue
        valid_candidates.append((gene_name, corrs[j]))
    if len(valid_candidates) == 0:
        print(f"[Warning] No valid gene with PCC > {pcc_threshold} was found for gene [{target_gene_name}.")
        print(f"   -> Trigger fallback mechanism: Use the global average feature of all known genes in the current slice.")
        all_valid_embs = [emb for name, emb in scgpt_embeddings_dict.items() if name in gene_name_list]
        return torch.stack(all_valid_embs).mean(dim=0)
    
    valid_candidates.sort(key=lambda x: x[1], reverse=True)
    top_k_candidates = valid_candidates[:k]
    
    top_k_names = [item[0] for item in top_k_candidates]
    top_k_corrs = [item[1] for item in top_k_candidates]
    
    print(f"Gene interpolation [{target_gene_name}] was successful! {len(top_k_names)} baseline genes were used.")
    for name, pcc in zip(top_k_names, top_k_corrs):
        print(f"   - {name}: PCC = {pcc:.4f}")
        
    top_k_embs = torch.stack([scgpt_embeddings_dict[name] for name in top_k_names])
    imputed_emb = top_k_embs.mean(dim=0)
    
    return imputed_emb


