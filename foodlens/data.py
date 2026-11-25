"""Food-101 loading, splitting and preprocessing."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SEED = 42
IMG_SIZE = 224       # network input: 224 x 224
RESIZE_SIZE = 256    # shorter side before cropping, keeps the aspect ratio
VAL_FRACTION = 0.10  # taken from the official training split, stratified per class

# ImageNet statistics. The pretrained backbones expect them, and the custom CNN
# uses them too so that every model sees identically normalised inputs.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_food101(seed: int = SEED, val_fraction: float = VAL_FRACTION):
    """Return ``(train, val, test, class_names)``.

    * train / val: stratified 90/10 split of the official training split (75,750 images)
    * test: the official test split (25,250 images). On the Hugging Face hub this
      split is published under the name ``validation``.
    """
    from datasets import load_dataset

    ds = load_dataset("ethz/food101")
    parts = ds["train"].train_test_split(
        test_size=val_fraction, stratify_by_column="label", seed=seed
    )
    class_names = ds["train"].features["label"].names
    return parts["train"], parts["test"], ds["validation"], class_names


def subset(hf_split, n: int, seed: int = SEED):
    """Random subset, only for quick pipeline checks."""
    n = min(n, len(hf_split))
    return hf_split.shuffle(seed=seed).select(range(n))


def build_transforms(train: bool, augment: bool = True) -> transforms.Compose:
    """Train: random crop, flip, small rotation, colour jitter. Eval: fixed centre crop."""
    if train and augment:
        steps = [
            transforms.Resize(RESIZE_SIZE),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    else:
        steps = [transforms.Resize(RESIZE_SIZE), transforms.CenterCrop(IMG_SIZE)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class Food101Torch(Dataset):
    """Wraps a Hugging Face split so a PyTorch DataLoader can read it."""

    def __init__(self, hf_split, transform):
        self.split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        item = self.split[idx]
        image = item["image"].convert("RGB")  # guarantees 3 channels (greyscale/CMYK files)
        return self.transform(image), item["label"]


def make_loaders(train_split, val_split, test_split, batch_size: int, augment: bool = True,
                 num_workers: int = 2, eval_batch_size: int = 128):
    """Train / val / test DataLoaders. Only the training loader shuffles and augments."""
    pin = torch.cuda.is_available()
    keep = num_workers > 0
    train_loader = DataLoader(
        Food101Torch(train_split, build_transforms(train=True, augment=augment)),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin,
        persistent_workers=keep, generator=torch.Generator().manual_seed(SEED),
    )
    eval_tf = build_transforms(train=False)
    val_loader = DataLoader(
        Food101Torch(val_split, eval_tf), batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin, persistent_workers=keep,
    )
    test_loader = DataLoader(
        Food101Torch(test_split, eval_tf), batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader


def pretty_label(name: str) -> str:
    """'apple_pie' -> 'Apple pie'."""
    return name.replace("_", " ").capitalize()
