import h5py 
import scanpy as sc
import torch 
import numpy as np 
import random 
import datetime

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 

    try:
        torch.use_deterministic_algorithms(True)
    except AttributeError:
        print("Warning: torch.use_deterministic_algorithms not supported in this version.")



def read_assets_from_h5(h5_path, keys=None, skip_attrs=False, skip_assets=False):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        if keys is None:
            keys = list(f.keys())
        for key in keys:
            if not skip_assets:
                assets[key] = f[key][:]
            if not skip_attrs:
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
    return assets, attrs



def get_current_time():
    now = datetime.datetime.now()
    year = now.year % 100  # convert to 2-digit year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{year:02d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"