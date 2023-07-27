import itertools
from tqdm import tqdm
import numpy as np
from os.path import join

import torch
from torchvision.utils import save_image
import torch.utils.data as data
from dataset_loader import DynamicsDataset


def get_dataloaders(root, batch_size, train_len=9500, test_len=500):
    
    total_len = train_len + test_len

    dataset = DynamicsDataset(percentage = 1.0)
    print("==== Total data points:", len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    test_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def save_np_images(images, path, nrow=10):
    images /= 255.
    images = torch.FloatTensor(images).permute(0, 3, 1, 2).contiguous()
    save_image(images, path, nrow=nrow)


def construct_run_command(script, arguments):
    command = f'python {script}'
    for k, v in arguments.items():
        if isinstance(v, bool):
            if v:
                command += f' --{k}'
        else:
            command += f' --{k} {str(v)}'
    return command


# Expects a list of dictionaries
def construct_variants(variants, default_dict=dict(), name_key='name'):
    level_keys = []
    variant_levels = []
    for var_level in variants:
        keys, values = zip(*var_level.items())
        assert all([len(v) == len(values[0]) for v in values])

        variants = list(zip(*values))
        level_keys.append(keys)
        variant_levels.append(variants)
    all_keys = sum(level_keys, tuple())
    all_variants = list(itertools.product(*variant_levels))
    all_variants = [sum(v, tuple()) for v in all_variants]
    assert all([len(v) == len(all_keys) for v in all_variants])

    final_variants = []
    for variant in all_variants:
        d = default_dict.copy()
        d.update({k: v for k, v in zip(all_keys, variant)})
        if name_key:
            d[name_key] = '_'.join([f"[{k}]_{v.replace('/', '_') if type(v) == str else v}" for k, v in zip(all_keys, variant)])
        final_variants.append(d)

    return final_variants


class Stats:
    def __init__(self):
        self.stats = dict()

    def add(self, key, value):
        if key not in self.stats:
            self.stats[key] = []
        self.stats[key].append(value)

    def keys(self):
        return list(self.stats.keys())

    def __getitem__(self, key):
        return self.stats[key]

    def items(self):
        return self.stats.items()