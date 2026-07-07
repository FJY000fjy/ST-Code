import argparse 
from configs import get_args_parser
import os 
from utils import set_random_seed,get_current_time
from utils import sample_micro_subgraphs
import json
from data import HESTDatasetPath,MultiHESTDataset_var,HESTDataset_var
from data import get_normalize_method
from data import pyg_pyramid_collate_fn
import torch 
from models import SlopeST
from core import merge_all_split_results
from core import test_var,evaluate_var
from core import RecLoss_var
from core import SSA_Loss
import torch.optim as optim 
from tqdm import tqdm
import pandas as pd 
from operator import itemgetter
import pandas as pd 
import numpy as np 
from utils import get_imputed_embedding,build_scgpt_embedding_matrix

def main(args,split_id,train_sample_ids,val_sample_ids,val_save_dir,checkpoint_save_dir):
    accumulation_steps = 1  
    normalize_method = get_normalize_method(args.normalize_method)
    train_sample_id_paths = [
        HESTDatasetPath(
            name = sample_id,
            h5_path = os.path.join(args.embed_dataroot,args.dataset[0],args.feature_encoder,f"fp32/{sample_id}.h5"),
            h5ad_path = os.path.join(args.source_dataroot,args.dataset[0],f"adata/{sample_id}.h5ad"),
            gene_list_path = os.path.join(args.source_dataroot,args.dataset[0],args.gene_list),
        ) for sample_id in train_sample_ids
    ]

    gene_list_path = os.path.join(args.source_dataroot,args.dataset[0],args.gene_list)

    with open(gene_list_path,'r',encoding='utf-8') as f:
        gene_name_dict = json.load(f)
    gene_name_list = gene_name_dict['genes']

    train_dataset = MultiHESTDataset_var(train_sample_id_paths,
                                 normalize_method = normalize_method,
                                 sample_times = args.slice_sample_times,
                                 )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=pyg_pyramid_collate_fn,
        drop_last=False, 
        shuffle=False,   
        
        )
    
    all_train_labels = torch.cat([sp_data.labels for sp_data in train_dataset.sp_datasets],dim=0)
    gene_mean = all_train_labels.mean(dim=0,keepdim=True)
    gene_std = all_train_labels.std(dim=0,keepdim=True)+1e-6
    labels_np = all_train_labels.numpy()
    train_set_corr_matrix = np.corrcoef(labels_np.T)


    val_sample_id_paths = [
        HESTDatasetPath(
            name=sample_id,
            h5_path = os.path.join(args.embed_dataroot,args.dataset[0],args.feature_encoder,f"fp32/{sample_id}.h5"),
            h5ad_path = os.path.join(args.source_dataroot,args.dataset[0],f"adata/{sample_id}.h5ad"),
            gene_list_path = os.path.join(args.source_dataroot,args.dataset[0],args.gene_list),
            
        ) for sample_id in val_sample_ids 
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
        ) for sample_id_path in val_sample_id_paths
    ]

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    gene_emb_path = os.path.join(args.gene_emb_dataroot,args.dataset[0],args.gene_emb_filename)
    print(f"gene_emb_path:{gene_emb_path}")
    gene_emb_arr = np.load(gene_emb_path,allow_pickle=True)
    data_dict = gene_emb_arr.item()
    scgpt_embedding_dict = dict(zip(data_dict['genes'],torch.from_numpy(data_dict['embeddings'])))
    for missing_g_name in data_dict['missing_genes']:
        print(f"missing_gene is {missing_g_name}")
        miss_g_idx = gene_name_list.index(missing_g_name)
        imputed_emb = get_imputed_embedding(
            corr_matrix = train_set_corr_matrix,
            target_gene_idx=miss_g_idx, 
            gene_name_list=gene_name_list, 
            scgpt_embeddings_dict=scgpt_embedding_dict, 
            k = 3, 
            pcc_threshold= 0.3,
        ) 
        scgpt_embedding_dict[missing_g_name]=imputed_emb
    
    scgpt_tensor_matrix = build_scgpt_embedding_matrix(
        scgpt_embedding_dict,
        gene_name_list,
    )

    torch.save({
        "gene_mean": gene_mean, 
        "gene_std": gene_std,  
        }, os.path.join(checkpoint_save_dir, "gene_stats.pt"))
    print("Global statistics and clusters saved.")

    model = SlopeST(
        feature_dim = args.feature_dim,
        output_dim =args.n_genes,
        hidden_dim = args.block_hidden_dim,
        scgpt_embedding_matrix=scgpt_tensor_matrix, 
        gene_embed_dim=512,
    ).to(args.device)
    criterion_rec = RecLoss_var(
        gene_weights=None,
    ).to(args.device)
    criterion_edgl_g_norm = SSA_Loss(
        k=8, eps=1e-8,
        gene_weights = None ).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)

    # print("Training")
    best_pearson,best_val_dict=-1,None 
    epoch_iter = tqdm(range(1,args.epochs+1),ncols=100)
    train_loss_list, val_loss_list, pcc_mean_list = [], [], []
    mse_mean_list, mae_mean_list, scc_mean_list = [], [], []
    macro_loss_list,meso_loss_list = [],[]
    macro_val_loss_list,meso_val_loss_list = [],[]
    scaler = torch.cuda.amp.GradScaler()

    for epoch in epoch_iter:
        avg_train_loss = 0
        macro_train_loss = 0
        meso_train_loss = 0
        model = model.to(args.device)
        model.train()
        optimizer.zero_grad()
        model.zero_grad() 
        for i,batched_pyramid in enumerate(train_loader):
            for scale in ['macro','meso','micro']:
                for key,val in batched_pyramid[scale].items():
                    if val is not None:
                        batched_pyramid[scale][key]=val.to(args.device)
            batched_pyramid['micro'] = sample_micro_subgraphs(
                batched_pyramid['micro'],
                points_per_anchor=args.points_per_anchor,  
                target_ratio=args.target_ratio,                    
                max_anchors=8,                       
            )

            for scale in ['macro', 'meso', 'micro']:
                is_tissue = batched_pyramid[scale]['in_tissue'].to(args.device)
                exp_sum = batched_pyramid[scale]['targets'].sum(dim=-1).to(args.device)
                batched_pyramid[scale]['valid_mask'] = (~is_tissue) | (exp_sum > 1e-5)                 
            valid_mask_macro = batched_pyramid['macro']['valid_mask']
            valid_mask_meso  = batched_pyramid['meso']['valid_mask']
            valid_mask_micro = batched_pyramid['micro']['valid_mask']
            macro_targets = (batched_pyramid['macro']['targets'] - gene_mean.to(args.device)) / gene_std.to(args.device)
            meso_targets  = (batched_pyramid['meso']['targets']  - gene_mean.to(args.device)) / gene_std.to(args.device)
            micro_targets = (batched_pyramid['micro']['targets'] - gene_mean.to(args.device)) / gene_std.to(args.device)
            for scale in ['macro','meso','micro']:
                feats = batched_pyramid[scale]['feats']
                norm_feats = feats = (feats - batched_pyramid['meso']['feats'].mean(0)) / (batched_pyramid['meso']['feats'].std(0) + 1e-6)
                batched_pyramid[scale]['feats'] = norm_feats 
            with torch.cuda.amp.autocast():
                macro_pred, meso_pred, micro_pred,micro_gate,pred_part1,pred_part2 = model(batched_pyramid)
                loss_macro = None 
                loss_meso = None 
                loss_micro = criterion_rec(y_pred=micro_pred,targets=micro_targets,valid_mask=valid_mask_micro)["total_loss"]
                if macro_pred is not None:
                    loss_macro = criterion_rec(y_pred=macro_pred,targets=macro_targets,valid_mask=valid_mask_macro)["total_loss"]
                if meso_pred is not None:
                    loss_meso = criterion_rec(y_pred=meso_pred,targets=meso_targets,valid_mask=valid_mask_meso)["total_loss"]
      
            loss_edf_g = 0.0
            loss_edf_g_micro = criterion_edgl_g_norm(
                pred=micro_pred.float(),
                target=micro_targets.float(),
                coords=batched_pyramid['micro']['coords'].float(),
                batch=batched_pyramid['micro']['batch'],
            )

            loss_edf_g_meso = criterion_edgl_g_norm(
                pred=meso_pred.float(),
                target = meso_targets.float(),
                coords=batched_pyramid['meso']['coords'].float(),
                batch=batched_pyramid['meso']['batch'],
            )
            loss_edf_g = 0.2 * loss_edf_g_micro + 0.1 * loss_edf_g_meso 
            loss_mse = 0.2 * loss_macro + 0.3 * loss_meso + 0.5 * loss_micro 
            loss = loss_mse + loss_edf_g 



            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                if args.clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_norm)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                model.zero_grad() 

            

            rec_loss_val = torch.nn.functional.mse_loss(micro_pred, micro_targets).item()
            avg_train_loss += rec_loss_val
            if macro_pred is not None:
                macro_train_loss += torch.nn.functional.mse_loss(macro_pred, macro_targets).item()
            if meso_pred is not None:
                meso_train_loss += torch.nn.functional.mse_loss(meso_pred, meso_targets).item()

        avg_train_loss /= len(train_loader)
        macro_train_loss /= len(train_loader)
        meso_train_loss /= len(train_loader)
        macro_loss_list.append(macro_train_loss)
        meso_loss_list.append(meso_train_loss)
        epoch_iter.set_description(f"epoch: {epoch}, avg_loss: {avg_train_loss:.3f}")
        train_loss_list.append(avg_train_loss)

        torch.cuda.empty_cache()
        if epoch % args.eval_step == 0 or epoch == args.epochs:
            
            res_dict,slide_info = evaluate_var(model,val_loaders,args.device,
                                               gene_name_list,
                                               gene_mean,gene_std,
                                     )
            avg_val_loss = res_dict['avg_loss']
            avg_macro_val_loss = res_dict['macro_loss']
            avg_meso_val_loss = res_dict['meso_loss']
            avg_val_pcc = res_dict['avg_pcc']
            avg_val_mse = res_dict['avg_mse'] 
            avg_val_mae = res_dict['avg_mae'] 
            avg_val_scc = res_dict['avg_scc']
            

            val_loss_list.append(avg_val_loss)
            macro_val_loss_list.append(avg_macro_val_loss)
            meso_val_loss_list.append(avg_meso_val_loss)
            current_pcc = res_dict['avg_pcc']
            pcc_mean_list.append(avg_val_pcc)
            scc_mean_list.append(avg_val_scc)

            mse_mean_list.append(avg_val_mse) 
            mae_mean_list.append(avg_val_mae) 
            if current_pcc > best_pearson:
                best_pearson = current_pcc
                checkpoint_save_path = os.path.join(checkpoint_save_dir,"best_model.pth") 
                torch.save(model.state_dict(),checkpoint_save_path)
                val_results_filename = "val_results.json" if args.split_mode == "fixed" else "results_kfold.json"
                val_results_payload = {"Note": "This is the result of the validation set."} if args.split_mode == "fixed" else {}
                with open(os.path.join(val_save_dir, val_results_filename),'w') as f:
                    for key, value in res_dict.items():
                        if hasattr(value, 'tolist'): 
                            value = value.tolist()
                        val_results_payload[key] = value
                    json.dump(val_results_payload,f,ensure_ascii = False,indent = 4)

                np.savez(os.path.join(val_save_dir,f"slide_info"),slide_info,allow_pickle=True)
               
                print(f" -> New Best Model Saved at{checkpoint_save_path}")
                print(f"Val epoch{epoch}")
                print(f"Epoch {epoch+1} Results:")
                print(f"  Val Loss:   {avg_val_loss:.4f}")
                print(f" Per-slide avg PCC:    {avg_val_pcc:.4f}")
                print(f" Per-slide avg SCC:    {avg_val_scc:.4f}")
                print(f"  Val MSE (Real):  {avg_val_mse:.4f}") 
                print(f"  Val MAE (Real):  {avg_val_mae:.4f}")

                print("-"*30)

    results = {}
    results['val_loss_list'] = val_loss_list
    results['macro_val_loss_list'] = macro_val_loss_list
    results['meso_val_loss_list'] = meso_val_loss_list
    results['train_loss_list'] = train_loss_list
    results['macro_loss_list'] = macro_loss_list
    results['meso_loss_list'] = meso_loss_list 
    results['pcc_mean_list'] = pcc_mean_list
    results['scc_mean_list'] = scc_mean_list
    results['mse_mean_list'] = mse_mean_list 
    results['mae_mean_list'] = mae_mean_list 
    return results,scgpt_tensor_matrix


def _read_sample_ids(csv_path):
    split_df = pd.read_csv(csv_path)
    if 'sample_id' not in split_df.columns:
        raise ValueError(f"Split file {csv_path} must contain a 'sample_id' column.")
    return split_df['sample_id'].tolist()


def load_split_specs(args):
    split_dir = os.path.join(args.source_dataroot,args.dataset[0],args.split_dir)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    if args.split_mode == 'kfold':
        train_files = [
            file_name for file_name in os.listdir(split_dir)
            if file_name.startswith('train_') and file_name.endswith('.csv')
        ]
        split_ids = sorted(
            int(file_name[len('train_'):-len('.csv')])
            for file_name in train_files
        )
        split_specs = []
        for split_id in split_ids:
            train_csv = os.path.join(split_dir,f"train_{split_id}.csv")
            test_csv = os.path.join(split_dir,f"test_{split_id}.csv")
            if not os.path.exists(test_csv):
                raise FileNotFoundError(f"Missing test split file for split {split_id}: {test_csv}")

            train_sample_ids = _read_sample_ids(train_csv)
            test_sample_ids = _read_sample_ids(test_csv)
            split_specs.append({
                "split_id": split_id,
                "train_sample_ids": train_sample_ids,
                "val_sample_ids": test_sample_ids,
                "test_sample_ids": test_sample_ids,
            })
        return split_specs

    if args.split_mode == 'fixed':
        train_csv = os.path.join(split_dir,"train_0.csv")
        val_csv = os.path.join(split_dir,"val_0.csv")
        test_csv = os.path.join(split_dir,"test_0.csv")
        for csv_path in [train_csv, val_csv, test_csv]:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Missing fixed split file: {csv_path}")

        train_sample_ids = _read_sample_ids(train_csv)
        return [{
            "split_id": 0,
            "train_sample_ids": train_sample_ids,
            "val_sample_ids": _read_sample_ids(val_csv),
            "test_sample_ids": _read_sample_ids(test_csv),
        }]

    raise ValueError(f"Unsupported split_mode: {args.split_mode}")


def get_total_results_filename(args):
    if args.split_mode == 'kfold':
        return 'total_results_kfold.json'
    return 'total_results_fixed.json'


def run(args):
    split_specs = load_split_specs(args)
    all_split_results = {}

    for split_spec in split_specs:
        i = split_spec["split_id"]
        print(f"Running dataset {args.dataset[0]} split {i}")

        train_sample_ids = split_spec["train_sample_ids"]
        val_sample_ids = split_spec["val_sample_ids"]
        test_sample_ids = split_spec["test_sample_ids"]

        kfold_save_dir = os.path.join(args.save_dir,f"split{i}")
        os.makedirs(kfold_save_dir,exist_ok = True)
        checkpoint_save_dir = os.path.join(kfold_save_dir,'checkpoints')
        os.makedirs(checkpoint_save_dir,exist_ok = True)

        results,scgpt_tensor_matrix = main(args,i,train_sample_ids,val_sample_ids,kfold_save_dir,checkpoint_save_dir)
        (
            test_pcc,
            test_scc,
            test_mse,
            test_mae,
            all_slides_pcc_per_gene,
            all_slides_scc_per_gene,
            all_slides_r2_score_per_gene,
        ) = test_var(args,i,test_sample_ids,checkpoint_save_dir,
                     scgpt_tensor_matrix,
                     )
        all_split_results[f'split {i}'] = results
        all_split_results[f'split {i} test pcc'] = test_pcc
        all_split_results[f'split {i} test scc'] = test_scc
        all_split_results[f'split {i} test mse'] = test_mse 
        all_split_results[f'split {i} test mae'] = test_mae
        all_split_results[f'split {i} per slide per gene pcc'] = all_slides_pcc_per_gene
        all_split_results[f'split {i} per slide per gene scc'] = all_slides_scc_per_gene
        all_split_results[f'split {i} per slide per gene r2_score'] = all_slides_r2_score_per_gene 

    all_split_results = merge_all_split_results(all_split_results)
    print("Final result!!!")
    
    final_mean_pcc = 0
    final_mean_scc = 0.0
    final_mean_mse = 0.0
    final_mean_mae = 0.0
    num_splits = len(split_specs)
    for split_spec in split_specs:
        i = split_spec["split_id"]
        final_mean_pcc += all_split_results[f'split {i} test pcc']
        final_mean_scc += all_split_results[f'split {i} test scc']
        final_mean_mse += all_split_results[f'split {i} test mse']
        final_mean_mae += all_split_results[f'split {i} test mae']
    final_mean_pcc /= num_splits
    final_mean_scc /= num_splits
    final_mean_mse /= num_splits 
    final_mean_mae /= num_splits 
    all_split_results['final_mean_pcc'] = final_mean_pcc
    all_split_results['final_mean_scc'] = final_mean_scc
    all_split_results['final_mean_mse'] = final_mean_mse 
    all_split_results['final_mean_mae'] = final_mean_mae 
    all_split_results['split_mode'] = args.split_mode
    all_split_results['split_dir'] = args.split_dir
    all_split_results['num_splits'] = num_splits

    total_results_filename = get_total_results_filename(args)
    with open(os.path.join(args.save_dir, total_results_filename), 'w') as f:
        p_corrs = all_split_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        all_split_results['pearson_corrs'] = p_corrs

        s_corrs = all_split_results['spearman_corrs']
        s_corrs = sorted(s_corrs, key=itemgetter('mean'), reverse=True)
        all_split_results['spearman_corrs'] = s_corrs

        r2 = all_split_results['r2_genes'] 
        r2 = sorted(r2,key=itemgetter('mean'),reverse=True)
        all_split_results['r2_genes'] = r2 

        for key, value in all_split_results.items():
            if hasattr(value, 'tolist'):  
                all_split_results[key] = value.tolist()
        json.dump(all_split_results, f, sort_keys=True, indent=4, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else str(x))

    return total_results_filename

  
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
parser = argparse.ArgumentParser('STFlow training script', parents=[get_args_parser()])
args = parser.parse_args()

set_random_seed(args.seed)

if args.save_dir==None:
    args.save_dir = f'./results/SlopeST_{args.baseline_method}_results'
if args.exp_code is None:
    args.exp_code = f"{args.backbone}::{get_current_time()}"
else:
    args.exp_code = args.exp_code + f"_{args.feature_encoder}"+f"_{args.backbone}::{get_current_time()}"

save_dir = os.path.join(args.save_dir,args.exp_code)
print(f"save_dir path:{args.save_dir}")
os.makedirs(save_dir,exist_ok=True)

save_dir = os.path.join(save_dir,args.dataset[0])
os.makedirs(save_dir,exist_ok=True)
print(f"save_dir is {save_dir}")
with open(os.path.join(save_dir, 'config.json'), 'w',encoding='utf-8') as f:
    json.dump(vars(args), f, indent=4)

args.save_dir = save_dir
total_results_filename = run(args)

final_result_file_path = os.path.join(save_dir,total_results_filename)

with open(final_result_file_path,'r',encoding='utf-8') as f:
    data = json.load(f)

print(data.keys())
print(data['split 0'].keys())
print(f"final_mean_pcc:{data['final_mean_pcc']}")
print(f"final_mean_scc:{data['final_mean_scc']}")
print(f"final_mean_mae:{data['final_mean_mae']}")
print(f"final_mean_mse:{data['final_mean_mse']}")
split_specs = load_split_specs(args)
for split_spec in split_specs:
    i = split_spec["split_id"]
    print(f"split {i} test pcc:{data[f'split {i} test pcc']}")
    print(f"split {i} test scc:{data[f'split {i} test scc']}")
    print(f"split {i} test mae:{data[f'split {i} test mae']}")
    print(f"split {i} test mse:{data[f'split {i} test mse']}")
