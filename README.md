# ST-Code
Code for our paper.

This repository contains the code and instructions needed to reproduce the experiments.

## Environment

The recommended Python version is 3.10. The provided `requirements.txt` uses
PyTorch and PyG wheels for CUDA 12.1, which should work for common reviewer GPUs
such as RTX 3090/4090. If your machine uses a different CUDA stack, install the
matching PyTorch and PyG wheels from the official PyTorch and PyG instructions.

```bash
conda create -n slopest python=3.10 pip
conda activate slopest

pip install -r requirements.txt
```


## Data And Weights

The experiments use HEST benchmark data and pretrained pathology image encoders
from Hugging Face. By default, all files are stored under `./dataset`.

`MahmoodLab/UNI` and `prov-gigapath/prov-gigapath` may require Hugging Face
access approval.

```bash
from huggingface_hub import login
login(token="ENTER YOUR TOKEN")

import os
from huggingface_hub import snapshot_download, hf_hub_download

source_dataroot = "./dataset/"
weights_root = "./dataset/weights_root"

os.makedirs(source_dataroot, exist_ok=True)
os.makedirs(weights_root, exist_ok=True)

snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=weights_root, allow_patterns=['fm_v1/*'])
snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=source_dataroot, ignore_patterns=['fm_v1/*'])
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "uni/"))
hf_hub_download("prov-gigapath/prov-gigapath", filename="pytorch_model.bin", local_dir=os.path.join(weights_root, "gigapath/"))
```

## Feature Extraction


Generate UNI tile embeddings before training. The output files are written to:

```text
dataset/embed_datasets/<DATASET>/uni_v1_official/fp32/<SAMPLE_ID>.h5
```

For HCC:

```bash
python Step1-embedding_uni.py \
  --datasets HCC \
  --source_dataroot ./dataset \
  --embed_dataroot ./dataset/embed_datasets \
  --weights_root ./dataset/weights_root \
  --encoder uni_v1_official \
  --precision fp32 \
  --batch_size 128 \
  --num_workers 1
```


## Training

Run model on HCC with UNI features:

```bash
python main.py \
  --dataset HCC \
  --source_dataroot ./dataset \
  --embed_dataroot ./dataset/embed_datasets \
  --gene_emb_dataroot ./scgpt_data/scgpt_emb 
```


The final metrics are saved under:

```text
results/<Baseline_method>_results/<EXP_CODE>_uni_v1_official_backbone::<TIME>/HCC/total_results_kfold.json
```

Each split also contains:

```text
split*/checkpoints/best_model.pth
split*/checkpoints/gene_stats.pt
split*/results_kfold.json
split*/slide_info.npz
```