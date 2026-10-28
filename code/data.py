"""CIFAR-10 four-animal subset and deterministic train/validation split."""
from __future__ import annotations

import random
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASSES = ["cat", "dog", "deer", "horse"]
ORIGINAL_IDS = [3, 5, 4, 7]


def _subset(dataset):
    indices = [i for i, y in enumerate(dataset.targets) if y in ORIGINAL_IDS]
    remap = {old: new for new, old in enumerate(ORIGINAL_IDS)}
    dataset.targets = [remap.get(y, -1) for y in dataset.targets]
    return Subset(dataset, indices)


def loaders(root="data", batch_size=128, seed=2026, workers=2):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.15, 0.05), transforms.ToTensor(),
        transforms.Normalize((0.49, 0.48, 0.45), (0.24, 0.24, 0.26)),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.49, 0.48, 0.45), (0.24, 0.24, 0.26))])
    base_a = _subset(datasets.CIFAR10(root, train=True, download=True, transform=train_tf))
    base_b = _subset(datasets.CIFAR10(root, train=True, download=False, transform=eval_tf))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(base_a), generator=g).tolist()
    cut = int(0.85 * len(perm))
    train_set = Subset(base_a, perm[:cut])
    val_set = Subset(base_b, perm[cut:])
    test_set = _subset(datasets.CIFAR10(root, train=False, download=True, transform=eval_tf))
    kwargs = dict(batch_size=batch_size, num_workers=workers, pin_memory=True)
    return (DataLoader(train_set, shuffle=True, generator=g, **kwargs),
            DataLoader(val_set, shuffle=False, **kwargs),
            DataLoader(test_set, shuffle=False, **kwargs))

