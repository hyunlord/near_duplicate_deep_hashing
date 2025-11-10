"""
이미지-텍스트 멀티모달 해싱 모델 (단일 GPU 최적화)
1-bit binary hash + XOR 검색 지원
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel


class NestedHashLayer(nn.Module):
    """다중 해시 길이를 위한 Nested Hash Layer"""
    def __init__(self, feature_dim: int, hidden_size, bit_list: list[int]):
        super().__init__()
        self.bit_list = sorted(bit_list)
        self.max_bit = self.bit_list[-1]

        self.hash_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.max_bit)
        )
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(bit) for bit in self.bit_list])

    def forward(self, x):
        full_output = self.hash_head(x)
        outputs_bits = [full_output[:, :length] for length in self.bit_list]
        outputs = [F.normalize(bn(output), p=2, dim=1) for output, bn in zip(outputs_bits, self.batch_norms)]
        return outputs


class MultiModalHashingModel(pl.LightningModule):
    """
    이미지-텍스트 멀티모달 해싱 모델
    단일 GPU Colab 환경 최적화
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        # Backbone 모델 로드 (SigLIP은 vision + text 모두 포함)
        backbone = AutoModel.from_pretrained(self.hparams.model_name)

        # Gradient checkpointing (메모리 절약)
        backbone.config.gradient_checkpointing = True
        backbone.gradient_checkpointing_enable()

        self.vision_model = backbone.vision_model
        self.text_model = backbone.text_model

        # 각 모달리티별 Hash Layer
        vision_hidden = self.vision_model.config.hidden_size
        text_hidden = self.text_model.config.hidden_size

        self.vision_hash = NestedHashLayer(
            vision_hidden,
            self.hparams.hash_hidden_dim,
            self.hparams.bit_list
        )
        self.text_hash = NestedHashLayer(
            text_hidden,
            self.hparams.hash_hidden_dim,
            self.hparams.bit_list
        )

        # EMA for quantization
        self.bit_importance_ema_dict = dict()
        self.ema_decay = 0.99

    def forward(self, images=None, texts=None):
        """
        Args:
            images: (B, 3, 384, 384) or None
            texts: dict with input_ids, attention_mask or None

        Returns:
            dict with 'vision' and/or 'text' keys containing list of embeddings
        """
        outputs = {}

        if images is not None:
            vision_features = self.vision_model(images).pooler_output
            outputs['vision'] = self.vision_hash(vision_features)

        if texts is not None:
            text_features = self.text_model(**texts).pooler_output
            outputs['text'] = self.text_hash(text_features)

        return outputs

    def class_aware_ortho_hash_loss(self, embeddings, labels):
        """같은 클래스는 유사하게, 다른 클래스는 직교하게"""
        B = F.normalize(embeddings, p=2, dim=1)
        sim = torch.matmul(B, B.T)
        batch_size = B.size(0)

        label_eq = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        identity = torch.eye(batch_size, device=labels.device)
        same_class_mask = label_eq * (1 - identity)
        diff_class_mask = 1 - label_eq

        loss_same = ((1 - sim) ** 2 * same_class_mask).sum()
        loss_diff = (sim ** 2 * diff_class_mask).sum()

        n_same = same_class_mask.sum().clamp(min=1.0)
        n_diff = diff_class_mask.sum().clamp(min=1.0)
        return loss_same / n_same + loss_diff / n_diff

    def vectorized_sample_hard_triplets(self, embeddings, labels):
        """Hard triplet mining (vectorized)"""
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        batch_size = embeddings.size(0)
        is_same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))
        is_not_self = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

        positive_mask = is_same_label & is_not_self
        negative_mask = ~is_same_label

        # Hard positive
        anchor_positive_dist = pairwise_dist.clone()
        anchor_positive_dist[~positive_mask] = -torch.inf
        hard_positive_indices = torch.argmax(anchor_positive_dist, dim=1)

        # Hard negative
        anchor_negative_dist = pairwise_dist.clone()
        anchor_negative_dist[~negative_mask] = torch.inf
        hard_negative_indices = torch.argmin(anchor_negative_dist, dim=1)

        # Valid triplets check
        valid_triplet_mask = (anchor_positive_dist != -torch.inf).any(dim=1) & \
                            (anchor_negative_dist != torch.inf).any(dim=1)

        if not valid_triplet_mask.any():
            return None, None, None, None, None, None

        valid_indices = torch.where(valid_triplet_mask)[0]
        anchors = embeddings[valid_indices]
        positives = embeddings[hard_positive_indices[valid_indices]]
        negatives = embeddings[hard_negative_indices[valid_indices]]

        return anchors, positives, negatives, valid_indices, hard_positive_indices, hard_negative_indices

    def contrastive_pair_loss(self, anchors, positives, negatives, margin):
        """Contrastive triplet loss"""
        pos_dist = F.pairwise_distance(anchors, positives, p=2)
        neg_dist = F.pairwise_distance(anchors, negatives, p=2)
        loss = F.relu(pos_dist - neg_dist + margin).mean()
        return loss

    def cross_modal_triplet_loss(self, image_embeds, text_embeds, labels, margin=0.5):
        """크로스 모달 triplet loss"""
        # Image-to-Text
        i2t_dist = torch.cdist(image_embeds, text_embeds, p=2)

        is_same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))

        # Hard positive
        pos_dist = i2t_dist.clone()
        pos_dist[~is_same_label] = -torch.inf

        if (pos_dist != -torch.inf).any():
            hard_pos_dist_i2t = pos_dist.max(dim=1)[0]
        else:
            return torch.tensor(0.0, device=image_embeds.device)

        # Hard negative
        neg_dist = i2t_dist.clone()
        neg_dist[is_same_label] = torch.inf

        if (neg_dist != torch.inf).any():
            hard_neg_dist_i2t = neg_dist.min(dim=1)[0]
        else:
            return torch.tensor(0.0, device=image_embeds.device)

        loss_i2t = F.relu(hard_pos_dist_i2t - hard_neg_dist_i2t + margin).mean()

        # Text-to-Image (대칭)
        t2i_dist = i2t_dist.T
        pos_dist_t2i = t2i_dist.clone()
        pos_dist_t2i[~is_same_label] = -torch.inf
        hard_pos_dist_t2i = pos_dist_t2i.max(dim=1)[0]

        neg_dist_t2i = t2i_dist.clone()
        neg_dist_t2i[is_same_label] = torch.inf
        hard_neg_dist_t2i = neg_dist_t2i.min(dim=1)[0]

        loss_t2i = F.relu(hard_pos_dist_t2i - hard_neg_dist_t2i + margin).mean()

        return (loss_i2t + loss_t2i) / 2

    def modality_alignment_loss(self, image_embeds, text_embeds):
        """모달리티 정렬 loss (같은 인덱스 = paired data)"""
        cos_sim = F.cosine_similarity(image_embeds, text_embeds, dim=1)
        loss = (1 - cos_sim).mean()
        return loss

    def cross_modal_quantization_loss(self, image_embeds, text_embeds):
        """이진화 후 크로스 모달 일치"""
        image_hash = torch.sign(image_embeds)
        text_hash = torch.sign(text_embeds)
        loss = F.mse_loss(image_hash, text_hash)
        return loss

    def long_short_cascade_loss(self, embeddings_list):
        """Long-Short Cascade: 긴 해시가 짧은 해시를 가르침"""
        lcs_losses = []
        for i in range(len(embeddings_list) - 1):
            short_embed = embeddings_list[i]
            long_embed = embeddings_list[i + 1]

            short_sim = torch.matmul(short_embed, short_embed.T)
            long_sim = torch.matmul(long_embed, long_embed.T)

            lcs_loss = F.mse_loss(short_sim, long_sim.detach())
            lcs_losses.append(lcs_loss)

        return sum(lcs_losses) / len(lcs_losses) if lcs_losses else torch.tensor(0.0, device=embeddings_list[0].device)

    def consistency_loss(self, anchors, positives):
        """Anchor-Positive 일관성"""
        return F.mse_loss(anchors, positives)

    def training_step(self, batch, batch_idx):
        images = batch['images']
        texts = batch['texts']
        labels = batch['labels']

        # Forward pass
        outputs = self(images=images, texts=texts)
        vision_embeds_list = outputs['vision']
        text_embeds_list = outputs['text']

        total_loss = 0
        num_valid_tasks = 0

        for vision_embeds, text_embeds, bit in zip(vision_embeds_list, text_embeds_list, self.hparams.bit_list):

            # === Single-modal losses ===

            # Vision
            v_triplets = self.vectorized_sample_hard_triplets(vision_embeds, labels)
            if v_triplets[0] is None:
                continue

            v_anchors, v_positives, v_negatives = v_triplets[:3]
            vision_triplet_loss = self.contrastive_pair_loss(v_anchors, v_positives, v_negatives, self.hparams.margin)
            vision_ortho_loss = self.class_aware_ortho_hash_loss(vision_embeds, labels)
            vision_base_loss = vision_triplet_loss + self.hparams.lambda_ortho * vision_ortho_loss

            # Text
            t_triplets = self.vectorized_sample_hard_triplets(text_embeds, labels)
            if t_triplets[0] is None:
                continue

            t_anchors, t_positives, t_negatives = t_triplets[:3]
            text_triplet_loss = self.contrastive_pair_loss(t_anchors, t_positives, t_negatives, self.hparams.margin)
            text_ortho_loss = self.class_aware_ortho_hash_loss(text_embeds, labels)
            text_base_loss = text_triplet_loss + self.hparams.lambda_ortho * text_ortho_loss

            # === Cross-modal losses ===

            cross_triplet_loss = self.cross_modal_triplet_loss(vision_embeds, text_embeds, labels, self.hparams.margin)
            alignment_loss = self.modality_alignment_loss(vision_embeds, text_embeds)
            quant_loss = self.cross_modal_quantization_loss(vision_embeds, text_embeds)

            # Combine
            base_loss = (vision_base_loss + text_base_loss) / 2
            cross_modal_loss = (
                self.hparams.lambda_cross * cross_triplet_loss +
                self.hparams.lambda_align * alignment_loss +
                self.hparams.lambda_quant * quant_loss
            )

            bit_loss = base_loss + cross_modal_loss

            total_loss += bit_loss
            num_valid_tasks += 1

            # Logging (단일 GPU: sync_dist 제거)
            self.log(f"train/{bit}_vision_loss", vision_base_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f"train/{bit}_text_loss", text_base_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.log(f"train/{bit}_cross_loss", cross_modal_loss, on_step=True, on_epoch=True, prog_bar=False)

        # Long-Short Cascade
        lcs_loss_vision = self.long_short_cascade_loss(vision_embeds_list)
        lcs_loss_text = self.long_short_cascade_loss(text_embeds_list)
        lcs_loss = (lcs_loss_vision + lcs_loss_text) / 2

        total_loss = total_loss / max(num_valid_tasks, 1) + self.hparams.lambda_lcs * lcs_loss

        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/lcs_loss", lcs_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def calculate_sim_acc(self, anchors, pos, neg, images_embeds):
        """유사도 및 정확도 계산"""
        cos = nn.CosineSimilarity(dim=1)
        pos_sim = cos(anchors, pos).mean()
        neg_sim = cos(anchors, neg).mean()

        hash_anchor = torch.sign(anchors)
        hash_pos = torch.sign(pos)
        hash_neg = torch.sign(neg)
        pos_hash_acc = (hash_anchor == hash_pos).all(dim=1).float().mean().item()
        neg_collision_rate = (hash_anchor == hash_neg).all(dim=1).float().mean().item()

        val_codes = torch.sign(images_embeds)
        mean_bit_variance = val_codes.float().var(dim=0).mean().item()

        return pos_sim, neg_sim, pos_hash_acc, neg_collision_rate, mean_bit_variance

    def calculate_retrieval_recall(self, query_hash, db_hash, labels, k=10):
        """Recall@K 계산"""
        # Hamming distance
        query_binary = ((query_hash + 1) / 2).long()
        db_binary = ((db_hash + 1) / 2).long()

        hamming_dist = (query_binary.unsqueeze(1) ^ db_binary.unsqueeze(0)).sum(dim=2)

        # Top-K
        _, top_k_indices = torch.topk(hamming_dist, k=min(k, db_hash.shape[0]), dim=1, largest=False)

        # Recall
        query_labels = labels.unsqueeze(1).expand(-1, min(k, db_hash.shape[0]))
        retrieved_labels = labels[top_k_indices]

        correct = (query_labels == retrieved_labels).any(dim=1).float()
        recall = correct.mean().item()

        return recall

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images = batch['images']
            texts = batch['texts']
            labels = batch['labels']

            outputs = self(images=images, texts=texts)
            vision_embeds_list = outputs['vision']
            text_embeds_list = outputs['text']

            for vision_embeds, text_embeds, bit in zip(vision_embeds_list, text_embeds_list, self.hparams.bit_list):

                # Vision metrics
                v_triplets = self.vectorized_sample_hard_triplets(vision_embeds, labels)
                if v_triplets[0] is not None:
                    v_anchors, v_positives, v_negatives = v_triplets[:3]
                    v_pos_sim, v_neg_sim, v_pos_acc, v_collision, v_variance = \
                        self.calculate_sim_acc(v_anchors, v_positives, v_negatives, vision_embeds)

                    self.log(f"val/{bit}_vision_pos_acc", v_pos_acc, on_epoch=True, prog_bar=True)
                    self.log(f"val/{bit}_vision_variance", v_variance, on_epoch=True)

                # Text metrics
                t_triplets = self.vectorized_sample_hard_triplets(text_embeds, labels)
                if t_triplets[0] is not None:
                    t_anchors, t_positives, t_negatives = t_triplets[:3]
                    t_pos_sim, t_neg_sim, t_pos_acc, t_collision, t_variance = \
                        self.calculate_sim_acc(t_anchors, t_positives, t_negatives, text_embeds)

                    self.log(f"val/{bit}_text_pos_acc", t_pos_acc, on_epoch=True, prog_bar=True)
                    self.log(f"val/{bit}_text_variance", t_variance, on_epoch=True)

                # Cross-modal metrics
                cos_sim = F.cosine_similarity(vision_embeds, text_embeds, dim=1).mean().item()
                self.log(f"val/{bit}_cross_modal_sim", cos_sim, on_epoch=True, prog_bar=True)

                # Hash agreement
                vision_hash = torch.sign(vision_embeds)
                text_hash = torch.sign(text_embeds)
                hash_agreement = (vision_hash == text_hash).float().mean().item()
                self.log(f"val/{bit}_hash_agreement", hash_agreement, on_epoch=True, prog_bar=True)

                # Retrieval metrics
                i2t_recall = self.calculate_retrieval_recall(vision_hash, text_hash, labels, k=10)
                t2i_recall = self.calculate_retrieval_recall(text_hash, vision_hash, labels, k=10)

                self.log(f"val/{bit}_i2t_recall@10", i2t_recall, on_epoch=True, prog_bar=True)
                self.log(f"val/{bit}_t2i_recall@10", t2i_recall, on_epoch=True, prog_bar=True)

                # Final score
                if v_triplets[0] is not None and t_triplets[0] is not None:
                    final_score = (
                        0.25 * v_pos_acc +
                        0.25 * t_pos_acc +
                        0.2 * hash_agreement +
                        0.15 * i2t_recall +
                        0.15 * t2i_recall
                    )
                    self.log(f"val/{bit}_final_score", final_score, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """옵티마이저 설정"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.get('weight_decay', 0.01)
        )

        # OneCycleLR 스케줄러
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos'
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
