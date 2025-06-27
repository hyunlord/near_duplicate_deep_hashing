import math
import numpy as np

import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler
import torch.distributed as dist

import pytorch_lightning as pl

import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset, Image as HFImage


class GroupBlockSampler(BatchSampler):
    def __init__(self, group_indices, batch_groups, sampler=None, drop_last=True):
        if sampler is not None:
            self.group_indices = list(sampler)
        else:
            self.group_indices = group_indices
        self.group_indices = group_indices
        self.batch_groups = batch_groups
        self.drop_last = drop_last
        self.num_groups = len(group_indices)
        if drop_last:
            self.num_batches = self.num_groups // batch_groups
        else:
            self.num_batches = math.ceil(self.num_groups / batch_groups)

        # DDP 설정
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        order = np.random.permutation(self.num_groups)
        for i in range(self.num_batches):
            batch = []
            selected = order[i*self.batch_groups:(i+1)*self.batch_groups]
            for gid in selected:
                batch.extend(self.group_indices[gid])
            yield batch

    def __len__(self):
        return self.num_batches

class ImageTripletDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transforms = A.Compose([
            A.Resize(height=384, width=384),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

    def prepare_data(self):
        load_dataset(self.config['dataset_name'], cache_dir=self.config['cache_dir'])

    def setup(self, stage=None):
        ds = load_dataset(self.config['dataset_name'], cache_dir=self.config['cache_dir'])
        self.train_ds = ds['train'][:10000].cast_column('image', HFImage())
        self.val_ds = ds['validation'][:1000].cast_column('image', HFImage())
        self.test_ds = ds['test'][:1000].cast_column('image', HFImage())

        # train 그룹 인덱스 생성
        self.train_group_indices = [
            list(range(i, i + self.config['images_per_group']))
            for i in range(0, len(self.train_ds), self.config['images_per_group'])
        ]

    def train_dataloader(self):
        sampler = GroupBlockSampler(self.train_group_indices, self.config['batch_groups'], drop_last=True)
        return DataLoader(
            self.train_ds,
            shuffle=False,
            batch_sampler=sampler,
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2
        )

    def val_dataloader(self):
        batch_size = self.config['batch_groups'] * self.config['images_per_group']
        return DataLoader(
            self.val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2
        )

    def test_dataloader(self):
        batch_size = self.config['batch_groups'] * self.config['images_per_group']
        return DataLoader(
            self.test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['num_workers'],
            collate_fn=self._collate_fn,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2
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