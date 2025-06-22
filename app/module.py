import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import pytorch_lightning as pl
from transformers import AutoImageProcessor, AutoModel
from torchmetrics.retrieval import RetrievalMAP


class DeepHashingModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        backbone = AutoModel.from_pretrained(self.hparams.model_name)
        self.vision_model = backbone.vision_model
        self.hash_head = nn.Linear(self.vision_model.config.hidden_size, self.hparams.hash_dim)
        self.bn = nn.BatchNorm1d(self.hparams.hash_dim)

        self.retrieval_map = RetrievalMAP()
        self.validation_step_outputs = []
        self.latest_hash_match_acc = 0.0

        crop_transforms = [
            A.RandomResizedCrop(size=(384, 384), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.5),
            A.RandomResizedCrop(size=(384, 384), scale=(0.3, 0.7), ratio=(0.75, 1.33), p=0.5),
            A.RandomResizedCrop(size=(384, 384), scale=(0.4, 0.8), ratio=(1.5, 2.5), p=0.5),
            A.RandomResizedCrop(size=(384, 384), scale=(0.4, 0.8), ratio=(0.4, 0.66), p=0.5),
        ]

        self.transformation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.OneOf(crop_transforms, p=0.3),
            A.Sequential([
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.ColorJitter(p=0.5),
            ], p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ImageCompression(quality_range=[70, 100], p=0.3),
            A.Resize(height=384, width=384),
            ToTensorV2()
        ])

        if self.hparams.get("freeze_backbone_epochs", 0) > 0:
            self.freeze_backbone()

    def freeze_backbone(self):
        print("Freezing vision model backbone.")
        for param in self.vision_model.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing vision model backbone.")
        for param in self.vision_model.parameters():
            param.requires_grad = True

    def on_train_epoch_start(self):
        freeze_epochs = self.hparams.get("freeze_backbone_epochs", 0)
        if freeze_epochs > 0 and self.current_epoch == freeze_epochs:
            self.unfreeze_backbone()

    def on_train_epoch_end(self):
        v_num = getattr(self.logger, "version", "unknown_version")
        if isinstance(v_num, int):
            v_num = f"version_{v_num}"

        save_dir = os.path.join(
            self.hparams.get("checkpoint_dir", "checkpoints"),
            v_num,
            f"epoch_{self.current_epoch:03d}_HMA_{self.latest_hash_match_acc:.4f}"
        )
        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = os.path.join(save_dir, "model.ckpt")
        self.trainer.save_checkpoint(ckpt_path)

    def forward(self, images):
        features = self.vision_model(images).pooler_output
        v = self.hash_head(features)
        v = self.bn(v)
        v = F.normalize(v, p=2, dim=1)
        return v

    def _mine_hard_triplets(self, embeddings, labels):
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        dist_matrix.fill_diagonal_(float('inf'))

        anchors, positives, negatives = [], [], []
        for i in range(len(labels)):
            anchor_label = labels[i]
            pos_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
            pos_indices = pos_indices[pos_indices != i]
            neg_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
            if len(pos_indices) >= 1 and len(neg_indices) > 0:
                pos_dists = dist_matrix[i, pos_indices]
                neg_dists = dist_matrix[i, neg_indices]

                hardest_positive_idx = pos_indices[pos_dists.argmax()]
                hardest_negative_idx = neg_indices[neg_dists.argmin()]

                anchors.append(embeddings[i])
                positives.append(embeddings[hardest_positive_idx])
                negatives.append(embeddings[hardest_negative_idx])
        if not anchors:
            return None, None, None
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)

    def _ortho_hash_loss(self, embeddings):
        B = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(B, B.T)
        batch_size = B.size(0)
        identity = torch.eye(batch_size, device=self.device)
        loss = torch.norm(similarity_matrix - identity, p='fro') ** 2
        return loss / (batch_size * batch_size)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        with torch.no_grad():
            weak_aug = torch.stack(
                [self.transformation_pipeline(image=img.cpu().numpy().transpose(1, 2, 0))['image'] for img in images]
            ).to(self.device)
        strong_aug = images

        v_anchor = self(strong_aug)
        v_aug = self(weak_aug)

        anchors, positives, negatives = self._mine_hard_triplets(v_anchor, labels)
        if anchors is None:
            dummy_loss = torch.tensor(1e-6, requires_grad=True, device=self.device)
            self.log("train/skipped_batch", 1.0, on_step=True, logger=True, sync_dist=True)
            return dummy_loss
        triplet_loss = F.triplet_margin_loss(anchors, positives, negatives, margin=self.hparams.margin)
        ortho_loss = self._ortho_hash_loss(v_anchor)
        distill_loss = F.mse_loss(v_anchor, v_aug.detach())
        total_loss = triplet_loss + self.hparams.lambda_ortho * ortho_loss + self.hparams.lambda_distill * distill_loss

        self.log_dict({
            'train/total_loss': total_loss,
            'train/triplet_loss': triplet_loss,
            'train/ortho_loss': ortho_loss,
            'train/distill_loss': distill_loss
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)

        self.validation_step_outputs.append({'embeddings': embeddings.detach(), 'labels': labels.detach()})
        with torch.no_grad():
            images_aug = torch.stack(
                [self.transformation_pipeline(image=img.cpu().numpy().transpose(1, 2, 0))['image'] for img in images]
            ).to(self.device)
            embeddings_aug = self(images_aug)

            original_hash = torch.sign(embeddings)
            augmented_hash = torch.sign(embeddings_aug)

            matches = (torch.sum(original_hash == augmented_hash, dim=1) == self.hparams.hash_dim)
            accuracy = torch.mean(matches.float())
            self.latest_hash_match_acc = accuracy.item()
            self.log('val/hash_match_acc', self.latest_hash_match_acc,
                     on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def all_gather_tensor(self, tensor):
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/hash_match_acc"}}
