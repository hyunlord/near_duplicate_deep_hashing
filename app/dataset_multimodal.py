"""
이미지-텍스트 멀티모달 데이터셋 (단일 GPU 최적화)
Colab 환경에 맞춘 메모리 효율적인 구현
"""
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import pytorch_lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class SimpleBatchSampler(Sampler):
    """
    단일 GPU용 단순화된 배치 샘플러
    그룹 기반 샘플링으로 triplet mining에 적합
    """
    def __init__(self, dataset, batch_groups, images_per_group, shuffle=True):
        """
        Args:
            dataset: Dataset
            batch_groups: 배치당 그룹 수 (예: 5)
            images_per_group: 그룹당 이미지 수 (예: 4)
            shuffle: 에포크마다 셔플 여부
        """
        self.dataset = dataset
        self.batch_groups = batch_groups
        self.images_per_group = images_per_group
        self.batch_size = batch_groups * images_per_group
        self.shuffle = shuffle

        # 그룹별로 인덱스 정리
        self.groups = self._organize_groups()
        self.num_groups = len(self.groups)
        self.num_batches = self.num_groups // batch_groups

    def _organize_groups(self):
        """데이터를 그룹별로 정리"""
        groups = []
        current_group = []

        for idx in range(len(self.dataset)):
            current_group.append(idx)
            if len(current_group) == self.images_per_group:
                groups.append(current_group)
                current_group = []

        return groups

    def __iter__(self):
        """배치 인덱스 생성"""
        # 그룹 순서 셔플
        group_indices = list(range(self.num_groups))
        if self.shuffle:
            random.shuffle(group_indices)

        # 배치 생성
        for batch_idx in range(self.num_batches):
            batch_groups = group_indices[
                batch_idx * self.batch_groups : (batch_idx + 1) * self.batch_groups
            ]

            # 배치에 포함될 샘플 인덱스
            batch_indices = []
            for group_idx in batch_groups:
                batch_indices.extend(self.groups[group_idx])

            yield batch_indices

    def __len__(self):
        return self.num_batches


class ImageTextPairDataset(Dataset):
    """
    이미지-텍스트 페어 데이터셋
    """
    def __init__(self, hf_dataset, transform=None, tokenizer=None, text_max_length=77):
        self.dataset = hf_dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 이미지 처리
        image = item['image']
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')

        if self.transform:
            image = self.transform(image=image)['image']

        # 텍스트 처리
        text = item.get('caption', item.get('text', ''))

        # 라벨
        label = item.get('label', item.get('image_group_id', 0))

        return {
            'image': image,
            'text': text,
            'label': label
        }


class ImageTextDataModule(pl.LightningDataModule):
    """
    이미지-텍스트 멀티모달 데이터 모듈 (단일 GPU 최적화)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Colab 메모리를 고려한 설정
        self.batch_groups = config.get('batch_groups', 5)  # 기존 10 → 5
        self.images_per_group = config.get('images_per_group', 4)  # 기존 10 → 4
        self.batch_size = self.batch_groups * self.images_per_group  # 20

        self.image_size = config.get('image_size', 384)
        self.text_max_length = config.get('text_max_length', 77)
        self.num_workers = config.get('num_workers', 2)  # Colab에서는 2로 제한

        # 이미지 전처리
        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])

    def setup(self, stage=None):
        """데이터셋 로드"""
        # 기존 데이터셋 사용 (일단 이미지만 있는 데이터셋)
        # 나중에 이미지-텍스트 페어 데이터셋으로 교체
        dataset = load_dataset(
            self.config.get('dataset_name', 'hyunlord/query_image_anchor_positive_large'),
            cache_dir=self.config.get('cache_dir', None)
        )

        # 데이터 분할 (Colab 메모리 고려하여 작게)
        train_size = self.config.get('train_size', 2000)  # 기존 4000 → 2000
        val_size = self.config.get('val_size', 200)  # 기존 400 → 200
        test_size = self.config.get('test_size', 200)

        # 전체 데이터셋을 train/val/test로 분할
        full_dataset = dataset['train']

        self.train_dataset = ImageTextPairDataset(
            full_dataset.select(range(train_size)),
            transform=self.transform,
            tokenizer=self.tokenizer,
            text_max_length=self.text_max_length
        )

        self.val_dataset = ImageTextPairDataset(
            full_dataset.select(range(train_size, train_size + val_size)),
            transform=self.transform,
            tokenizer=self.tokenizer,
            text_max_length=self.text_max_length
        )

        self.test_dataset = ImageTextPairDataset(
            full_dataset.select(range(train_size + val_size,
                                     train_size + val_size + test_size)),
            transform=self.transform,
            tokenizer=self.tokenizer,
            text_max_length=self.text_max_length
        )

    def collate_fn(self, batch):
        """배치 생성"""
        images = torch.stack([item['image'] for item in batch])
        texts = [item['text'] for item in batch]
        labels = torch.tensor([item['label'] for item in batch])

        # 텍스트가 없는 경우 라벨을 텍스트로 변환 (임시)
        if not texts[0]:
            texts = [f"image group {label}" for label in labels.tolist()]

        # 텍스트 토크나이징
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors='pt'
        )

        return {
            'images': images,
            'texts': text_inputs,
            'labels': labels
        }

    def train_dataloader(self):
        # SimpleBatchSampler 사용
        sampler = SimpleBatchSampler(
            self.train_dataset,
            batch_groups=self.batch_groups,
            images_per_group=self.images_per_group,
            shuffle=True
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
