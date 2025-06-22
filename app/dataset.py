import random
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist

import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset, Image as HFImage


class PKSampler(BatchSampler):
    def __init__(self, dataset, dataset_labels, p, k, sampler=None, drop_last=True):
        self.dataset = dataset
        self.dataset_labels = dataset_labels
        self.p = p
        self.k = k
        self.base_sampler = sampler
        self.drop_last = drop_last

        self.indices = list(self.base_sampler) if self.base_sampler else list(range(len(dataset_labels)))
        self.labels_to_anchors = defaultdict(list)
        self.labels_to_positives = defaultdict(list)
        for idx in self.indices:
            label = self.dataset_labels[idx]
            item = self.dataset[idx]
            if item["image_type"] == "anchor":
                self.labels_to_anchors[label].append(idx)
            elif item["image_type"] == "positive":
                self.labels_to_positives[label].append(idx)
        self.valid_labels = [
            label for label in self.labels_to_anchors
            if len(self.labels_to_anchors[label]) >= 1 and len(self.labels_to_positives[label]) >= (self.k - 1)
        ]

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        self.local_labels = self.valid_labels[self.rank::self.world_size]
        num_batches_local = len(self.local_labels) // self.p

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dist.is_initialized():
            num_batches_tensor = torch.tensor(num_batches_local, device=device)
            dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.MIN)
            self.num_batches = max(1, num_batches_tensor.item())
        else:
            self.num_batches = max(1, num_batches_local)

    def __iter__(self):
        random.shuffle(self.local_labels)
        for i in range(self.num_batches):
            try:
                batch_indices = []
                selected_labels = self.local_labels[i * self.p: (i + 1) * self.p]
                for label in selected_labels:
                    anchor_idx = random.choice(self.labels_to_anchors[label])
                    positive_indices = random.sample(self.labels_to_positives[label], self.k - 1)
                    batch_indices.extend([anchor_idx] + positive_indices)
                if len(batch_indices) == 0:
                    continue
                random.shuffle(batch_indices)
                yield batch_indices
            except Exception as e:
                print(f"[Rank {self.rank}] Failed to sample batch {i}: {e}")
                continue

    def __len__(self):
        return self.num_batches


class ImageTripletDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transforms = A.Compose([
            A.Resize(height=config['image_size'], width=config['image_size']),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

    def prepare_data(self):
        load_dataset(self.config['dataset_name'], cache_dir=self.config['cache_dir'])

    def setup(self, stage=None):
        name = self.config['dataset_name']
        cache_dir = self.config['cache_dir']

        self.train_dataset = load_dataset(name, split="train", cache_dir=cache_dir).cast_column("image", HFImage()).with_format("torch")
        self.val_dataset = load_dataset(name, split="validation", cache_dir=cache_dir).cast_column("image", HFImage())
        self.test_dataset = load_dataset(name, split="test", cache_dir=cache_dir).cast_column("image", HFImage())

    def train_dataloader(self):
        sampler = torch.utils.data.SequentialSampler(self.train_dataset)
        labels_as_str = [item['image_group_id'] for item in self.train_dataset]

        pk_batch_sampler = PKSampler(
            dataset=self.train_dataset,
            dataset_labels=labels_as_str,
            p=self.config['p'],
            k=self.config['k'],
            sampler=sampler
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=pk_batch_sampler,
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=min(len(self.val_dataset), self.config['p'] * self.config['k']),
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=min(len(self.test_dataset), self.config['p'] * self.config['k']),
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            pin_memory=True
        )

    def _collate_fn(self, batch):
        def to_numpy_image(image):
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy()  # CHW → HWC
            elif hasattr(image, 'numpy'):
                image = image.numpy()
            else:
                image = np.array(image)
            return image

        images = [
            self.transforms(image=to_numpy_image(item['image']))['image'] for item in batch
        ]
        labels = [item['image_group_id'] for item in batch]
        return torch.stack(images), torch.tensor(labels)
