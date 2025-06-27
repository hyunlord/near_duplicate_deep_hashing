import os

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
        self.hash_head = nn.Sequential(
            nn.Linear(self.vision_model.config.hidden_size, self.hparams.hash_hidden_dim),
            nn.GELU(),
            nn.Linear(self.hparams.hash_hidden_dim, self.hparams.hash_dim)
        )
        self.layer_norm = nn.LayerNorm(self.hparams.hash_dim)
        self.class_centers = nn.Parameter(computed_centers.clone().to(self.device))

        self.retrieval_map = RetrievalMAP()
        self.validation_step_outputs = []
        self.latest_hash_match_acc = 0.0
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
        features = self.vision_model(images).last_hidden_state.mean(dim=1)
        v = self.hash_head(features)
        v = self.layer_norm(v)
        v = F.normalize(v, p=2, dim=1)
        return v

    def hash_supervision_loss(self, embeddings):
        # [B, D], 각 해시 비트를 -1 또는 1로 만들기 위한 항
        return torch.mean((torch.abs(embeddings) - 1) ** 2)

    def binarization_regularizer(self, embeddings):
        return torch.mean(torch.abs(embeddings))

    def bit_balance_loss(self, embeddings):
        # 각 비트 위치마다 평균값의 제곱이 작을수록 분포가 고르게 됨
        bit_means = torch.mean(embeddings, dim=0)
        return torch.mean(bit_means ** 2)

    def ortho_hash_loss(self, embeddings):
        # 각 임베딩 간 cosine similarity 계산
        B = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(B, B.T)
        batch_size = B.size(0)

        # 미리 diagonal 제거
        identity = torch.eye(batch_size, device=self.device)
        off_diagonal = similarity_matrix * (1 - identity)

        # 정확한 off-diagonal element 개수로 나눔
        loss = torch.sum(off_diagonal ** 2) / (batch_size * (batch_size - 1))
        return loss

    def class_aware_ortho_hash_loss(self, embeddings, labels):
        # 각 임베딩 간 cosine similarity 계산
        B = F.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(B, B.T)
        batch_size = B.size(0)

        # 같은 클래스는 유사하게, 다른 클래스는 직교하게
        label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        identity = torch.eye(batch_size, device=labels.device)
        same_class_mask = label_eq * (1 - identity)
        diff_class_mask = 1 - label_eq

        # 같은 클래스는 cosine sim이 1에 가깝도록, 다른 클래스는 0에 가깝도록
        loss_same = ((1 - sim) ** 2 * same_class_mask).sum()
        loss_diff = (sim ** 2 * diff_class_mask).sum()

        n_same = same_class_mask.sum().clamp(min=1.0)
        n_diff = diff_class_mask.sum().clamp(min=1.0)
        return loss_same / n_same + loss_diff / n_diff

    def codebook_loss(self, embeddings, labels):
        centers = F.normalize(self.class_centers, p=2, dim=1)  # [C, D]
        embs = F.normalize(embeddings, p=2, dim=1)  # [B, D]
        target_centers = centers[labels]  # [B, D]
        return F.mse_loss(embs, target_centers)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        unique_labels = labels.unique()
        if unique_labels.numel() < 2:
            dummy_loss = torch.tensor(1e-6, requires_grad=True, device=self.device)
            self.log("train/skipped_batch", 1.0, on_step=True, logger=True, sync_dist=True)
            return dummy_loss

        # 중복 계산을 방지하기 위해 전체 배치 이미지를 한 번에 임베딩 게산
        images_embeds = self(images)
        anchors, hard_positives, hard_negatives, weak_augments = [], [], [], []
        anchor_labels = []
        for label in unique_labels:
            # 해당 라벨을 가진 인덱스 리스트
            label_mask = (labels == label)
            label_indices = label_mask.nonzero(as_tuple=True)[0]
            if len(label_indices) < 2:
                continue  # anchor + positive 구성 불가

            # anchor로 사용할 라벨 별 첫 번째 이미지 선택
            anchor_idx = label_indices[0]
            anchor_embed = images_embeds[anchor_idx].unsqueeze(0)
            anchor_labels.append(labels[anchor_idx].unsqueeze(0))

            # hard positive: 같은 라벨 중 anchor에서 가장 멀리 떨어진 같은 이미지
            pos_indices = label_indices[label_indices != anchor_idx]
            pos_embeds = images_embeds[pos_indices]
            dists_pos = F.pairwise_distance(anchor_embed.expand(pos_embeds.size(0), -1), pos_embeds)
            hard_pos_idx = pos_indices[dists_pos.argmax()]
            hard_pos_embed = images_embeds[hard_pos_idx].unsqueeze(0)

            # hard negative: 다른 라벨 중 anchor에서 가장 가까운 것
            neg_indices = (labels != label).nonzero(as_tuple=True)[0]
            neg_embeds = images_embeds[neg_indices]
            dists_neg = F.pairwise_distance(anchor_embed.expand(neg_embeds.size(0), -1), neg_embeds)
            hard_neg_idx = neg_indices[dists_neg.argmin()]
            hard_neg_embed = images_embeds[hard_neg_idx].unsqueeze(0)

            # weak positive: 같은 라벨 중 랜덤으로 하나 → augmentation
            weak_pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]
            weak_pos_embed = images_embeds[weak_pos_idx].unsqueeze(0)

            anchors.append(anchor_embed)
            hard_positives.append(hard_pos_embed)
            hard_negatives.append(hard_neg_embed)
            weak_augments.append(weak_pos_embed)
        anchors = torch.cat(anchors, dim=0)
        hard_pos = torch.cat(hard_positives, dim=0)
        hard_neg = torch.cat(hard_negatives, dim=0)
        weak_augments = torch.cat(weak_augments, dim=0)

        # ortho loss 입력용
        anchor_labels = torch.cat(anchor_labels, dim=0)
        all_embeds = torch.cat([anchors, hard_pos, hard_neg], dim=0)
        all_labels = anchor_labels.repeat(3)

        # loss 계산
        triplet_loss = F.triplet_margin_loss(anchors, hard_pos, hard_neg, margin=self.hparams.margin)
        distill_loss = F.mse_loss(anchors, weak_augments.detach())
        ortho_loss = self.class_aware_ortho_hash_loss(all_embeds, all_labels)
        codebook_loss = self.codebook_loss(anchors, anchor_labels)
        total_loss = (
                triplet_loss +
                self.hparams.lambda_ortho * ortho_loss +
                self.hparams.lambda_distill * distill_loss +
                self.hparams.lambda_codebook * codebook_loss
        )
        self.log_dict({
                'train/total_loss': total_loss,
                'train/triplet_loss': triplet_loss,
                'train/ortho_loss': ortho_loss,
                'train/distill_loss': distill_loss,
                'train/codebook_loss': codebook_loss
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
