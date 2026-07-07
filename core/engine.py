import argparse
import numpy as np 
import torch 
from core.loss  import RecLoss_var
from core import metric_func
import torch.nn.functional as F 
from tqdm import tqdm 
from data import get_normalize_method,HESTDatasetPath,HESTDataset_var,pyg_pyramid_collate_fn
import os 
import json 
from models import SlopeST


def calculate_tensor_pcc(pred,target):
    pred_mean = pred.mean(dim=0,keepdim=True)
    target_mean = target.mean(dim=0,keepdim=True)
    pred_diff = pred - pred_mean 
    target_diff = target - target_mean 
    cov = (pred_diff * target_diff).sum(dim=0)
    pred_std = torch.sqrt((pred_diff ** 2).sum(dim=0))
    target_std = torch.sqrt((target_diff**2).sum(dim=0))
    pcc_per_gene = cov / (pred_std * target_std+1e-8)
    return pcc_per_gene.mean().item()


def evaluate_var(model,val_loaders,device,gene_name_list,gene_mean,gene_std):
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    macro_loss = 0.0
    meso_loss = 0.0
    pyramid_loss = 0.0
    total_pcc = 0.0
    total_mse = 0.0 
    total_mae = 0.0 
    num_batches = 0 
   
    criterion_rec = RecLoss_var().to(device)
    scc_metrics_log = [] 
    slides_info = {} 
    with torch.no_grad():
        for current_sample_idx,loader in tqdm(enumerate(val_loaders),desc="Evaluating",leave = False):
            for batched_pyramid in loader:
                for scale in ['macro','meso','micro']:
                    for key,val in batched_pyramid[scale].items():
                        if val is not None:
                            batched_pyramid[scale][key] = val.to(device)

                for scale in ['macro','meso','micro']:
                    feats = batched_pyramid[scale]['feats']
                    norm_feats = feats = (feats - batched_pyramid['meso']['feats'].mean(0)) / (batched_pyramid['meso']['feats'].std(0) + 1e-6)
                    batched_pyramid[scale]['feats'] = norm_feats

                targets_biological = batched_pyramid['micro']['targets']


                norm_genes = (targets_biological - gene_mean.to(device))/gene_std.to(device)
                macro_pred, meso_pred, micro_pred,micro_gate,pred_part1,pred_part2 = model(batched_pyramid)
                loss_dict = criterion_rec(y_pred=micro_pred,targets=norm_genes)
                total_loss +=loss_dict["total_loss"].item() 
                if macro_pred is not None:
                    targets_macro = batched_pyramid['macro']['targets']
                    macro_norm_genes = (targets_macro - gene_mean.to(device))/gene_std.to(device)
                    macro_loss += criterion_rec(y_pred=macro_pred,targets = macro_norm_genes)["total_loss"].item() 

                if meso_pred is not None:
                    targets_meso = batched_pyramid['meso']['targets']
                    meso_norm_genes = (targets_meso -  gene_mean.to(device))/gene_std.to(device)
                    meso_loss += criterion_rec(y_pred=meso_pred,targets = meso_norm_genes)["total_loss"].item() 

                pred_exp_biological = micro_pred * gene_std.to(device) + gene_mean.to(device) # micro
                if meso_pred is not None:
                    meso_exp_biological = meso_pred * gene_std.to(device) + gene_mean.to(device) # meso 
                if macro_pred is not None:
                    macro_exp_biological = macro_pred * gene_std.to(device) + gene_mean.to(device) # macro 
            

                pcc = calculate_tensor_pcc(pred_exp_biological,targets_biological)
                total_pcc += pcc 

                mse = F.mse_loss(pred_exp_biological,targets_biological)
                mae = F.l1_loss(pred_exp_biological,targets_biological)

                total_mse +=mse.item() 
                total_mae +=mae.item() 
                num_batches += 1

                preds_np = pred_exp_biological.cpu().numpy() 
                targets_np = targets_biological.cpu().numpy()
                coords_np = batched_pyramid['micro']['coords'].cpu().numpy() 

                slide_metrics = metric_func(preds_np,targets_np,gene_name_list)

                scc_metrics_log.append(slide_metrics['spearman_mean'])
                slides_info[f"{current_sample_idx}"] = {
                    "preds_np": preds_np,
                    "targets_np": targets_np,
                    "coords_np": coords_np,
                    "pcc_per_slide": slide_metrics['pearson_mean'],
                    "scc_per_slide": slide_metrics['spearman_mean'],
                    "mse_per_slide": slide_metrics['l2_error_q2'],
                    "pcc_per_gene": slide_metrics['pearson_corrs'],
                    "scc_per_gene": slide_metrics['spearman_corrs'],
                    "r2_score_per_gene": slide_metrics['r2_genes'],
                }
    avg_loss = total_loss/num_batches 
    avg_pcc = total_pcc / num_batches 
    avg_mse = total_mse / num_batches
    avg_mae = total_mae / num_batches 
    
    macro_avg_loss = macro_loss/num_batches
    meso_avg_loss = meso_loss/num_batches 
    pyramid_avg_loss = pyramid_loss/num_batches

    
    avg_scc = np.mean(scc_metrics_log)

    res_dict = {}

    res_dict['avg_loss'] = avg_loss 
    res_dict['macro_loss']=macro_avg_loss
    res_dict['meso_loss'] = meso_avg_loss 
    res_dict['pyramid_loss'] = pyramid_avg_loss 
    res_dict['avg_pcc'] = avg_pcc
    res_dict['avg_mse'] = avg_mse 
    res_dict['avg_mae'] = avg_mae 
    res_dict['avg_scc'] = avg_scc

    return res_dict,slides_info


def test_var(args,split_id,test_sample_ids,checkpoint_save_dir,
             scgpt_tensor_matrix,
             ):   
    normalize_method = get_normalize_method(args.normalize_method) 

    print(f"\n{split_id}'s Training Finished. Loading Best Model for Testing...")
    print(f"This is split{split_id}")
    test_sample_id_paths = [
            HESTDatasetPath(
                name=sample_id,
                h5_path = os.path.join(args.embed_dataroot,args.dataset[0],args.feature_encoder,f"fp32/{sample_id}.h5"),
                h5ad_path = os.path.join(args.source_dataroot,args.dataset[0],f"adata/{sample_id}.h5ad"),
                gene_list_path = os.path.join(args.source_dataroot,args.dataset[0],args.gene_list),
                
            ) for sample_id in test_sample_ids 
        ]

    val_loaders = [
        torch.utils.data.DataLoader(
            HESTDataset_var(
                sample_id_path,
                normalize_method = normalize_method,
            ),
            batch_size = 1,
            collate_fn = pyg_pyramid_collate_fn,
            shuffle=False  
        ) for sample_id_path in test_sample_id_paths
    ]



    gene_list_path = os.path.join(args.source_dataroot,args.dataset[0],args.gene_list)

    with open(gene_list_path,'r',encoding='utf-8') as f:
        gene_name_dict = json.load(f)
    gene_name_list = gene_name_dict['genes']

    gene_stats = torch.load(os.path.join(checkpoint_save_dir, "gene_stats.pt"))
    gene_mean = gene_stats["gene_mean"]
    gene_std = gene_stats["gene_std"]


    model = SlopeST(
        feature_dim=args.feature_dim, 
        output_dim=args.n_genes,       
        hidden_dim=args.block_hidden_dim,                    
        scgpt_embedding_matrix=scgpt_tensor_matrix,
        gene_embed_dim=512,
    ).to(args.device)

    model.load_state_dict(torch.load(os.path.join(checkpoint_save_dir,"best_model.pth")))


    res_dict, slides_info = evaluate_var(model, val_loaders, 
                                         args.device, 
                                         gene_name_list, 
                                         gene_mean, gene_std,
                                         )

    test_loss = res_dict['avg_loss']
    test_pcc = res_dict['avg_pcc']
    test_scc = res_dict['avg_scc']
    test_mse = res_dict['avg_mse'] 
    test_mae = res_dict['avg_mae'] 

    print("=" * 40)
    print(f"FINAL TEST RESULTS:")
    print(f"Test MSE Loss: {test_loss:.4f}")
    print(f"Per-slide avg PCC:      {test_pcc:.4f}")
    print(f"Per-slide avg SCC:      {test_scc:.4f}")
    print(f"Test MSE (Real):  {test_mse:.4f}") 
    print(f"Test MAE (Real):  {test_mae:.4f}") 
    print("=" * 40)
    
    all_slides_pcc_per_gene = []
    all_slides_scc_per_gene = []
    all_slides_r2_score_per_gene = []
    
    for slide_idx,slide_info in slides_info.items():
        slide_idx = int(slide_idx)
        all_slides_pcc_per_gene.extend(slide_info['pcc_per_gene'])
        all_slides_scc_per_gene.extend(slide_info['scc_per_gene'])
        all_slides_r2_score_per_gene.extend(slide_info['r2_score_per_gene'])



    print("=" * 40)
    return (
        test_pcc,
        test_scc,
        test_mse,
        test_mae,
        all_slides_pcc_per_gene,
        all_slides_scc_per_gene,
        all_slides_r2_score_per_gene,
    )
