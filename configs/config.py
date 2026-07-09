import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='STFlow Training and Evaluation', add_help=False)
    path_group = parser.add_argument_group('Path & Basic')
    # "LUNG", "HCC", "COAD", "SKCM", "PAAD", "READ", "LYMPH_IDC", "PRAD", "IDC", "CCRCC"
    path_group.add_argument('--dataset',nargs='+',default=['HCC'],help='Dataset list')

    path_group.add_argument('--embed_dataroot',type=str,default='./dataset/embed_datasets')
    path_group.add_argument('--source_dataroot',type=str,default='./dataset/')

    # scGPT embedding
    path_group.add_argument('--gene_emb_dataroot',type=str,default='./scgpt_data/scgpt_emb')
    path_group.add_argument('--gene_emb_filename',type=str,default='gene_embeddings_scgpt.npy')
    path_group.add_argument('--gene_list',type=str,default='var_50genes.json')
    # 'kfold'  or  'fixed'
    path_group.add_argument('--split_mode',type=str,default='kfold',choices=['kfold','fixed'],
                            help='kfold: read train_i/test_i splits; fixed: read train_0/val_0/test_0 split')
    path_group.add_argument('--split_dir',type=str,default='splits',
                            help='Directory under source_dataroot/dataset that stores split CSV files')
    path_group.add_argument('--baseline_method',type=str,default='SlopeST')
    path_group.add_argument('--save_dir',type=str,default=None,help='If not specified, it will be automatically generated based on baseline_method.')
    path_group.add_argument('--exp_code',type=str,default='test')


    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument('--device',type=int,default=0)
    train_group.add_argument('--seed',type=int,default=1)
    train_group.add_argument('--batch_size', type=int, default=2, help='Batch size') 
    train_group.add_argument('--lr', type=float, default=5e-4)
    train_group.add_argument('--epochs', type=int, default=100)
    train_group.add_argument('--clip_norm', type=float, default=1.) 
    train_group.add_argument('--eval_step', type=int, default=1) 




    data_group = parser.add_argument_group('Data Processing')
    data_group.add_argument('--normalize_method',type=str,default='log1p')
    data_group.add_argument('--patch_distribution',type=str,default='uniform')
    data_group.add_argument('--points_per_anchor',type=int,default=256) 
    data_group.add_argument('--target_ratio',type = float, default = 0.2)
    data_group.add_argument('--slice_sample_times',type=int,default = 5)
    data_group.add_argument('--micro_n_neighbors',type=int,default=8)




    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--feature_encoder', type=str, default='uni_v1_official', choices=['uni_v1_official', 'gigapath', 'ciga'])
    model_group.add_argument('--backbone',type=str,default='backbone')
    model_group.add_argument('--n_genes',type=int,default=50)
    model_group.add_argument('--feature_dim', type=int, default=1024, help="uni:1024, ciga:512, gigapath: 1536") 
    model_group.add_argument('--block_hidden_dim',type=int,default=256)
    # HGNN
    model_group.add_argument('--n_neighbors',type=int, default=8)
    model_group.add_argument('--dropout',type=float,default=0.2)

    return parser
