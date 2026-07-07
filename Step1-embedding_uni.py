import argparse
import os
import random

import h5py
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
UNI_V1_OFFICIAL_CONFIG = {
    "name": "uni_v1",
    "img_norm": "imagenet",
    "loader": "timm",
    "loader_kwargs": {
        "model_name": "vit_large_patch16_224",
        "dynamic_img_size": True,
        "num_classes": 0,
        "init_values": 1.0,
    },
    "checkpoint_path": "",
    "load_state_dict_strict": True,
}
LOCAL_CKPTS = {
    "uni_v1_official": "uni/pytorch_model.bin",
}


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_path(path):
    def get_path_relative(file, path) -> str:
        curr_dir = os.path.dirname(os.path.abspath(file))
        return os.path.join(curr_dir, path)

    src = get_path_relative(__file__, "./")
    if path.startswith("./"):
        new_path = os.path.join(src, path)
    else:
        new_path = path
    return new_path


def get_constants(norm="imagenet"):
    if norm == "imagenet":
        return IMAGENET_MEAN, IMAGENET_STD
    if norm == "none":
        return None, None
    raise ValueError(f"Invalid norm: {norm}")


def get_eval_transforms(mean, std, target_img_size=-1, center_crop=False):
    trsforms = []

    if target_img_size > 0:
        trsforms.append(transforms.Resize(target_img_size))
    if center_crop:
        assert target_img_size > 0, "target_img_size must be set if center_crop is True"
        trsforms.append(transforms.CenterCrop(target_img_size))

    trsforms.append(transforms.ToTensor())
    if mean is not None and std is not None:
        trsforms.append(transforms.Normalize(mean, std))
    trsforms = transforms.Compose(trsforms)

    return trsforms


def save_hdf5(output_fpath, asset_dict, attr_dict=None, mode="a", auto_chunk=True, chunk_size=None):
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f:
                data_type = val.dtype
                if data_type == np.object_:
                    data_type = h5py.string_dtype(encoding="utf-8")
                if auto_chunk:
                    chunks = True
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(
                        key,
                        shape=data_shape,
                        chunks=chunks,
                        maxshape=(None,) + data_shape[1:],
                        dtype=data_type,
                    )
                    if attr_dict is not None and key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except Exception:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                assert dset.dtype == val.dtype
                dset[-data_shape[0]:] = val

    return output_fpath


class H5TileDataset(Dataset):
    def __init__(self, h5_path, img_transform=None, chunk_size=1000):
        self.h5_path = h5_path
        self.img_transform = img_transform
        self.chunk_size = chunk_size

        with h5py.File(h5_path, "r") as f:
            if chunk_size == -1:
                self.n_chunks = 1
                self.chunk_size = len(f["barcode"])
            else:
                self.n_chunks = int(np.ceil(len(f["barcode"]) / chunk_size))

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        end_idx = (idx + 1) * self.chunk_size
        with h5py.File(self.h5_path, "r") as f:
            imgs = f["img"][start_idx:end_idx]
            barcodes = f["barcode"][start_idx:end_idx].flatten().tolist()
            coords = f["coords"][start_idx:end_idx]

        if self.img_transform:
            imgs = torch.stack([self.img_transform(Image.fromarray(img)) for img in imgs])

        return {"imgs": imgs, "barcodes": barcodes, "coords": coords}


def build_model(config):
    load_state_dict = False
    eval_transform = None

    if config.get("checkpoint_path", None) is not None:
        if not os.path.exists(config["checkpoint_path"]):
            if os.environ.get("CHECKPOINT_PATH", None) is not None:
                config["checkpoint_path"] = os.environ["CHECKPOINT_PATH"]
            else:
                raise ValueError(
                    f"checkpoint_path does not exist: {config['checkpoint_path']} "
                    "and no CHECKPOINT_PATH environment variable set"
                )
        load_state_dict = True

    if config["loader"] == "timm":
        model = timm.create_model(**config["loader_kwargs"])
    else:
        raise ValueError(f"Unsupported loader type: {config['loader']}")

    if load_state_dict:
        ckpt_path = config["checkpoint_path"]
        strict = config.get("load_state_dict_strict", False)
        print(f"Loading model from checkpoint: {ckpt_path}")
        print(f"load_state_dict_strict: {strict}")
        missing, unexpected = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=strict)
        if missing or unexpected:
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

    return model, eval_transform


def get_encoder(model_name, overwrite_kwargs=None, img_size=224):
    if overwrite_kwargs is None:
        overwrite_kwargs = {}
    if model_name != "uni_v1_official":
        raise ValueError("Step1-embedding_uni.py only supports the uni_v1_official encoder")

    config = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in UNI_V1_OFFICIAL_CONFIG.items()
    }
    for key in overwrite_kwargs:
        if key not in config:
            raise ValueError(f"Invalid overwrite key: {key}")
        config[key] = overwrite_kwargs[key]

    model, eval_transform = build_model(config)
    mean, std = get_constants(config["img_norm"])

    if eval_transform is None:
        eval_transform = get_eval_transforms(mean, std, target_img_size=img_size)
    return model, eval_transform, config


def load_encoder(enc_name, device, weights_root, private_weights_root=None):
    if enc_name not in LOCAL_CKPTS:
        raise ValueError("Step1-embedding_uni.py only supports the uni_v1_official encoder")

    overwrite_kwargs = {
        "checkpoint_path": os.path.join(weights_root, LOCAL_CKPTS[enc_name]),
    }
    encoder, img_transforms, enc_config = get_encoder(model_name=enc_name, overwrite_kwargs=overwrite_kwargs)

    _ = encoder.eval()
    encoder.to(device)
    return encoder, img_transforms, enc_config


class LazyEncoder:
    def __init__(self, name, weights_root, private_weights_root=None, transforms=None, model=None):
        self.name = name
        self.model = model
        self.transforms = transforms
        self.weights_root = weights_root
        self.private_weights_root = private_weights_root

    def get_model(self, device):
        if self.model is not None:
            return self.model, self.transforms
        encoder, img_transforms, _ = load_encoder(
            self.name,
            device,
            self.weights_root,
            self.private_weights_root,
        )
        return encoder, img_transforms


def embed_tiles(dataloader, model, embedding_save_path, device, precision=torch.float32, use_coords=None):
    def post_collate_fn(batch):
        if batch["imgs"].dim() == 5:
            assert batch["imgs"].size(0) == 1
            batch["imgs"] = batch["imgs"].squeeze(0)
        if batch["coords"].dim() == 3:
            assert batch["coords"].size(0) == 1
            batch["coords"] = batch["coords"].squeeze(0)
        return batch

    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Embedding Tiles", ncols=100):
        batch = post_collate_fn(batch)
        imgs = batch["imgs"].to(device)

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
            if use_coords:
                embeddings = model(imgs, batch["coords"].to(device))
            else:
                embeddings = model(imgs)

        mode = "w" if batch_idx == 0 else "a"
        asset_dict = {"embeddings": embeddings.cpu().numpy()}
        asset_dict.update({key: np.array(val) for key, val in batch.items() if key != "imgs"})
        save_hdf5(embedding_save_path, asset_dict=asset_dict, mode=mode)

    return embedding_save_path


def read_split_sample_rows(dataset_root, split_dir):
    split_root = os.path.join(dataset_root, split_dir)
    split_files = [
        os.path.join(split_root, name)
        for name in os.listdir(split_root)
        if name.endswith(".csv")
    ]
    if not split_files:
        raise FileNotFoundError(f"No split CSV files found in {split_root}")

    rows = []
    seen_sample_ids = set()
    for split_file in sorted(split_files):
        split_df = pd.read_csv(split_file)
        if "sample_id" not in split_df.columns:
            raise ValueError(f"Split file {split_file} must contain a sample_id column")
        for _, row in split_df.iterrows():
            sample_id = row["sample_id"]
            if sample_id in seen_sample_ids:
                continue
            patches_path = row["patches_path"] if "patches_path" in split_df.columns else f"patches/{sample_id}.h5"
            rows.append({"sample_id": sample_id, "patches_path": patches_path})
            seen_sample_ids.add(sample_id)
    return rows


def embed_dataset(args, dataset_name, lazy_encoder, device, precision):
    source_dataroot = get_path(args.source_dataroot)
    dataset_root = os.path.join(source_dataroot, dataset_name)
    embedding_dir = os.path.join(get_path(args.embed_dataroot), dataset_name, lazy_encoder.name, args.precision)
    os.makedirs(embedding_dir, exist_ok=True)

    print(f"Embedding tiles using {lazy_encoder.name} encoder")
    encoder, img_transforms = lazy_encoder.get_model(device)

    sample_rows = read_split_sample_rows(dataset_root, args.split_dir)
    for row in tqdm(sample_rows, desc=f"Embedding {dataset_name}", ncols=100):
        sample_id = row["sample_id"]
        tile_h5_path = os.path.join(dataset_root, row["patches_path"])
        if not os.path.isfile(tile_h5_path):
            raise FileNotFoundError(f"Tile file {tile_h5_path} does not exist")

        embed_path = os.path.join(embedding_dir, f"{sample_id}.h5")
        if os.path.isfile(embed_path) and not args.overwrite:
            print(f"Skipping {sample_id} as it already exists")
            continue

        tile_dataset = H5TileDataset(
            tile_h5_path,
            chunk_size=args.batch_size,
            img_transform=img_transforms,
        )
        tile_dataloader = torch.utils.data.DataLoader(
            tile_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
        embed_tiles(
            tile_dataloader,
            encoder,
            embed_path,
            device,
            precision=precision,
            use_coords=(lazy_encoder.name == "gigapathslide"),
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source_dataroot", type=str, default="./dataset/")
    parser.add_argument("--embed_dataroot", type=str, default="./dataset/embed_datasets")
    parser.add_argument("--weights_root", type=str, default="./dataset/weights_root")
    parser.add_argument("--private_weights_root", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--split_dir", type=str, default="splits")
    parser.add_argument("--encoder", type=str, default="uni_v1_official", choices=["uni_v1_official"])
    parser.add_argument("--datasets", nargs="+", default=["LUNG"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precisions = {"fp16": torch.float16, "fp32": torch.float32}
    precision = precisions.get(args.precision, torch.float32)

    lazy_encoder = LazyEncoder(
        args.encoder,
        weights_root=get_path(args.weights_root),
        private_weights_root=args.private_weights_root,
    )

    for dataset_name in args.datasets:
        embed_dataset(args, dataset_name, lazy_encoder, device, precision)
