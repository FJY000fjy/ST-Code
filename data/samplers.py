import numpy as np 
import math 
from functools import partial
from scipy.spatial import KDTree


def constant_distribution(ratio=0.2):
    return ratio


def uniform_distribution():
    return np.random.rand()


def beta_distribution(alpha=3, beta=9):
    return np.random.beta(alpha, beta)


def cosine_distribution():
    return (1 - math.cos(np.random.rand() * math.pi * 0.5))


def square_root_distribution():
    return math.sqrt(np.random.rand())


def square_distribution():
    return np.random.rand() ** 2

def get_distribution(name: str):
    if "constant" in name:  # constant_0.2
        ratio = float(name.split("_")[1])
        return partial(constant_distribution, ratio=ratio)
    elif "beta" in name:
        alpha, beta = [float(x) for x in name.split("_")[1:]]
        return partial(beta_distribution, alpha=alpha, beta=beta)
    elif name == "uniform":
        return uniform_distribution
    elif name == "cosine":
        return cosine_distribution
    elif name == "sqrt":
        return square_root_distribution
    elif name == "square":
        return square_distribution
    else:
        raise ValueError(f"Unknown distribution: {name}")



class PatchSampler:
    def __init__(self, distribution: str='batch_128', min_samples=2):
        self.distribution = distribution
        self.distribution_func = get_distribution(distribution)
        self.min_samples = min_samples

    def sample_nearest_patch(self, coords, num_samples):
        num_samples = min(len(coords), num_samples)

        if num_samples == len(coords):
            return np.arange(len(coords))
        tree = KDTree(coords)
        
        center_idx = np.random.randint(0, len(coords))
        center_coord = coords[center_idx]
        
        _, idx_nearest = tree.query(center_coord, k=num_samples)
        return idx_nearest

    def get_distribution_expectation(self):
        return np.mean([self.distribution() for _ in range(10000)])

    def __call__(self, coords):
        total_samples = max(self.min_samples, int(len(coords) * self.distribution_func()))
        return self.sample_nearest_patch(coords, total_samples)

