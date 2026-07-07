import torch 
import numpy as np 

def pearson_corr(preds, targets):
    preds_flat = preds.reshape(-1, preds.shape[-1])
    targets_flat = targets.reshape(-1, targets.shape[-1])
    vx = preds_flat - torch.mean(preds_flat, dim=0)
    vy = targets_flat - torch.mean(targets_flat, dim=0)
    cost = torch.sum(vx * vy, dim=0) / (torch.sqrt(torch.sum(vx ** 2, dim=0)) * torch.sqrt(torch.sum(vy ** 2, dim=0)) + 1e-8)
    return torch.mean(cost) 


from scipy.stats import pearsonr, spearmanr
def metric_func(preds_all: np.ndarray, y_test: np.ndarray, genes: list):   
    errors = []
    r2_scores = []
    r2_genes = []
    pearson_corrs = []
    pearson_genes = []
    spearman_corrs = []
    spearman_genes = []
    
    n_nan_genes = 0
    n_nan_spearman_genes = 0
    for i, target in enumerate(range(y_test.shape[1])):
        
        preds = preds_all[:, target] 
        target_vals = y_test[:, target]

        errors.append(float(np.mean((preds - target_vals) ** 2)))
        r2 = float(1 - np.sum((target_vals - preds) ** 2) / np.sum((target_vals - np.mean(target_vals)) ** 2))
        r2_scores.append(r2)
        pearson_corr, _ = pearsonr(target_vals, preds)
        spearman_corr, _ = spearmanr(target_vals, preds)

        if np.isnan(pearson_corr):
            n_nan_genes += 1
            pearson_corr = 0.0 
        if np.isnan(spearman_corr):
            n_nan_spearman_genes += 1
            spearman_corr = 0.0 
        pearson_corr = float(pearson_corr)
        spearman_corr = float(spearman_corr)
        pearson_corrs.append(pearson_corr)
        spearman_corrs.append(spearman_corr) 
        score_dict = {
            'name': genes[i],
            'pearson_corr': pearson_corr,
        }
        pearson_genes.append(score_dict)

        spearman_score_dict = {
            'name': genes[i],
            'spearman_corr': spearman_corr,
        }
        spearman_genes.append(spearman_score_dict)

        r2_score_dict = {
            'name': genes[i],
            'r2_score': r2,
        }

        r2_genes.append(r2_score_dict)

    if n_nan_genes > 0:
        print(f"Warning: {n_nan_genes} genes have NaN Pearson correlation")
    if n_nan_spearman_genes > 0:
        print(f"Warning: {n_nan_spearman_genes} genes have NaN Spearman correlation")

    return {'l2_errors': list(errors), 
            'r2_scores': list(r2_scores),
            'r2_genes':r2_genes,
            'pearson_corrs': pearson_genes,
            'pearson_mean': float(np.mean(pearson_corrs)),
            'pearson_std': float(np.std(pearson_corrs)),
            'spearman_corrs': spearman_genes,
            'spearman_mean': float(np.mean(spearman_corrs)),
            'spearman_std': float(np.std(spearman_corrs)),
            'l2_error_q1': float(np.percentile(errors, 25)),
            'l2_error_q2': float(np.median(errors)),
            'l2_error_q3': float(np.percentile(errors, 75)),
            'r2_score_q1': float(np.percentile(r2_scores, 25)),
            'r2_score_q2': float(np.median(r2_scores)),
            'r2_score_q3': float(np.percentile(r2_scores, 75))
        }


from collections import defaultdict 
def merge_all_split_results(all_split_results):
    output_dict = {}
    per_slide_gene_corrs_map = defaultdict(list)
    per_slide_gene_spearman_map = defaultdict(list)
    per_slide_gene_r2_map = defaultdict(list)

    for split_key, value in all_split_results.items():
        if split_key.endswith('per slide per gene pcc'):
            for item in value:
                gene_name = item['name']
                corr_value = float(item['pearson_corr'])
                per_slide_gene_corrs_map[gene_name].append(corr_value)

        elif split_key.endswith('per slide per gene scc'):
            for item in value:
                gene_name = item['name']
                corr_value = float(item['spearman_corr'])
                per_slide_gene_spearman_map[gene_name].append(corr_value)
                
        elif split_key.endswith('per slide per gene r2_score'):
            for item in value:
                gene_name = item['name']
                r2_value = float(item['r2_score'])
                per_slide_gene_r2_map[gene_name].append(r2_value)
                
        else:
            output_dict[split_key] = value


    def aggregate_gene_metrics(gene_metric_map, raw_key):
        metric_list = []
        for name, values in gene_metric_map.items():
            metric_list.append({
                "name": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                raw_key: values,
            })
        return metric_list

    output_dict["pearson_corrs"] = aggregate_gene_metrics(per_slide_gene_corrs_map, "pearson_corrs_raw")
    output_dict["spearman_corrs"] = aggregate_gene_metrics(per_slide_gene_spearman_map, "spearman_corrs_raw")
    output_dict["r2_genes"] = aggregate_gene_metrics(per_slide_gene_r2_map, "r2_scores_raw")

    return output_dict
