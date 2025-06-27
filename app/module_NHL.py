import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AutoModel


class DeepHashingModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        backbone = AutoModel.from_pretrained(self.hparams.model_name)
        backbone.config.gradient_checkpointing = True
        backbone.gradient_checkpointing_enable()
        self.vision_model = backbone.vision_model

        max_bit = max(self.hparams.bit_list)
        self.hash_head = nn.Sequential(
            nn.Linear(self.vision_model.config.hidden_size, self.hparams.hash_hidden_dim),
            nn.GELU(),
            nn.Linear(self.hparams.hash_hidden_dim, max_bit)
        )
        self.layer_norm = nn.LayerNorm(self.hparams.hash_dim)

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
        outputs = self.vision_model(images)
        features = outputs.last_hidden_state.mean(dim=1)

        raw = self.hash_head(features)  # (B, max_bit)
        outs = []
        for bit in self.hparams.bit_list:
            head = raw[:, :bit]  # (B, bit)
            normed = self.layer_norm(head)
            outs.append(F.normalize(normed, p=2, dim=1))
        return outs

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

    def _sample_hard_triplets(self, embeddings, labels):
        anchors, positives, negatives = [], [], []
        anchor_labels, pos_labels, neg_labels = [], [], []
        for label in labels.unique():
            idxs = (labels == label).nonzero(as_tuple=True)[0]
            if idxs.numel() < 2:
                continue

            # anchor
            a_idx = idxs[0]
            anchor = embeddings[a_idx].unsqueeze(0)

            # hard positive: 같은 클래스 중 가장 멀리 있는 것
            pos_idxs = idxs[idxs != a_idx]
            pos_embs = embeddings[pos_idxs]
            dists_pos = F.pairwise_distance(anchor.expand(pos_embs.size(0), -1), pos_embs)
            hp_idx = pos_idxs[dists_pos.argmax()]
            hard_pos = embeddings[hp_idx].unsqueeze(0)

            # hard negative: 다른 클래스 중 가장 가까운 것
            neg_idxs = (labels != label).nonzero(as_tuple=True)[0]
            if neg_idxs.numel() == 0:
                continue
            neg_embs = embeddings[neg_idxs]
            dists_neg = F.pairwise_distance(anchor.expand(neg_embs.size(0), -1), neg_embs)
            hn_idx = neg_idxs[dists_neg.argmin()]
            hard_neg = embeddings[hn_idx].unsqueeze(0)

            anchors.append(anchor)
            positives.append(hard_pos)
            negatives.append(hard_neg)

            anchor_labels.append(labels[a_idx].unsqueeze(0))
            pos_labels.append(labels[hp_idx].unsqueeze(0))
            neg_labels.append(labels[hn_idx].unsqueeze(0))
        if not anchors:
            return None, None, None, None, None, None
        return (
            torch.cat(anchors, dim=0),
            torch.cat(positives, dim=0),
            torch.cat(negatives, dim=0),
            torch.cat(anchor_labels, dim=0),
            torch.cat(pos_labels, dim=0),
            torch.cat(neg_labels, dim=0),
        )

    def training_step(self, batch, batch_idx):
        images, labels = batch
        unique_labels = labels.unique()
        if unique_labels.numel() < 2:
            dummy_loss = torch.tensor(1e-6, requires_grad=True, device=self.device)
            self.log("train/skipped_batch", 1.0, on_step=True, logger=True, sync_dist=True)
            return dummy_loss

        # 중복 계산을 방지하기 위해 전체 배치 이미지를 한 번에 임베딩 게산
        images_embeds_list = self(images)

        triplet_losses, ortho_losses, distill_losses, total_losses = [], [], [], []
        for images_embeds in images_embeds_list:
            anchors, pos, neg, a_lbl, p_lbl, n_lbl = self._sample_hard_triplets(images_embeds, labels)
            if anchors is None:
                dummy_loss = torch.tensor(1e-6, requires_grad=True, device=self.device)
                self.log("train/skipped_batch", 1.0, on_step=True, logger=True, sync_dist=True)
                return dummy_loss

            # Triplet loss
            triplet_loss = F.triplet_margin_loss(anchors, pos, neg, margin=self.hparams.margin)

            # Ortho loss
            all_embeds = torch.cat([anchors, pos, neg], dim=0)
            all_labels = torch.cat([a_lbl, p_lbl, n_lbl], dim=0)
            ortho_loss = self.class_aware_ortho_hash_loss(all_embeds, all_labels)
            total_loss = triplet_loss + self.hparams.lambda_ortho * ortho_loss

            triplet_losses.append(triplet_loss)
            ortho_losses.append(ortho_loss)
            #distill_losses.append(distill_loss)
            total_losses.append(total_loss)
        triplet_loss = sum(triplet_losses)
        ortho_loss = sum(ortho_losses)
        #distill_loss = sum(distill_losses)
        total_loss = sum(total_losses)
        self.log("train/triplet_losses", triplet_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        self.log("train/ortho_loss", ortho_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        #self.log("train/distill_loss", distill_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images, labels = batch
            images_embeds_list = self(images)

            triplet_losses, ortho_losses, distill_losses, total_losses = [], [], [], []
            for images_embeds, bit in zip(images_embeds_list, self.hparams.bit_list):
                anchors, pos, neg, a_lbl, p_lbl, n_lbl = self._sample_hard_triplets(images_embeds, labels)
                if anchors is None:
                    dummy_loss = torch.tensor(1e-6, requires_grad=True, device=self.device)
                    return dummy_loss

                # Triplet loss
                triplet_loss = F.triplet_margin_loss(anchors, pos, neg, margin=self.hparams.margin)

                # 평균 positive/negative cosine 유사도
                cos = nn.CosineSimilarity(dim=1)
                pos_sim = cos(anchors, pos).mean()
                neg_sim = cos(anchors, neg).mean()

                # Ortho loss
                all_embeds = torch.cat([anchors, pos, neg], dim=0)
                all_labels = torch.cat([a_lbl, p_lbl, n_lbl], dim=0)
                ortho_loss = self.class_aware_ortho_hash_loss(all_embeds, all_labels)

                total_loss = (
                        triplet_loss +
                        self.hparams.lambda_ortho * ortho_loss
                )
                triplet_losses.append(triplet_loss)
                ortho_losses.append(ortho_loss)
                total_losses.append(total_loss)

                #    Anchor vs. Positive 해시 일치율 계산
                #    embedding → sign 해시로 변환
                hash_anchor = torch.sign(anchors)
                hash_pos = torch.sign(pos)
                #    비트 단위 완전 일치 비율
                matches = (hash_anchor == hash_pos).all(dim=1).float().mean()
                self.latest_hash_match_acc = matches.item()

                self.log(f"val/{bit}_pos_sim", pos_sim, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f"val/{bit}_neg_sim", neg_sim, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f"val/{bit}_pos_hash_acc", self.latest_hash_match_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            triplet_loss = sum(triplet_losses)
            ortho_loss = sum(ortho_losses)
            total_loss = sum(total_losses)

            # 로깅
            self.log("val/triplet_loss", triplet_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val/ortho_loss", ortho_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,  # peak LR
            total_steps=total_steps,  # 전체 스텝 수
            pct_start=0.3,  # warm-up 비율 (예: 30%)
            anneal_strategy='cos',  # cosine annealing
            cycle_momentum=False  # AdamW는 모멘텀 개념이 없으므로 False
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 'step'마다 lr 업데이트
                "frequency": 1
            }
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)