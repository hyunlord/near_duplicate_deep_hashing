import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_optimizer import PCGrad

import pytorch_lightning as pl

from transformers import AutoModel


class NestedHashLayer(nn.Module):
    def __init__(self, feature_dim: int, hidden_size, bit_list: list[int]):
        super().__init__()
        self.bit_list = sorted(bit_list)
        self.max_bit = self.bit_list[-1]

        self.hash_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.max_bit)
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(bit) for bit in self.bit_list])

    def forward(self, x):
        full_output = self.hash_head(x)
        # 최대 비트 길이의 출력을 각 해시 길이에 맞게 앞에서 부터 슬라이싱
        # 짧은 비트의 파라미터가 긴 비트의 파라미터의 일부가 되는 구조를 만듬
        outputs_bits = [full_output[:, :length] for length in self.bit_list]
        # LayerNorm & L2 Normalization
        outputs = [F.normalize(ln(output), p=2, dim=1) for output, ln in zip(outputs_bits, self.layer_norms)]
        # 여러 길이의 해시 코드에 해당하는 출력을 반환
        return outputs


class DeepHashingModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        # [수정 1] 수동 최적화 모드 활성화
        self.automatic_optimization = False

        backbone = AutoModel.from_pretrained(self.hparams.model_name)
        backbone.config.gradient_checkpointing = True
        backbone.gradient_checkpointing_enable()
        self.vision_model = backbone.vision_model
        self.nhl = NestedHashLayer(self.vision_model.config.hidden_size, self.hparams.hash_hidden_dim,
                                   self.hparams.bit_list)

    def forward(self, images):
        features = self.vision_model(images).last_hidden_state.mean(dim=1)
        outputs = self.nhl(features)
        return outputs

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

    def vectorized_sample_hard_triplets(self, embeddings, labels):
        # 1. 배치 내 모든 샘플 간의 유클리드 거리 행렬을 한 번에 계산합니다.
        # torch.cdist는 모든 쌍(pair) 간의 거리를 계산해주는 효율적인 함수입니다.
        # pairwise_dist의 shape: (batch_size, batch_size)
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        # 2. 마스크(Mask) 생성
        batch_size = embeddings.size(0)
        # 2-1. 라벨이 같은지 여부를 나타내는 마스크 (i, j가 같은 라벨이면 True)
        # shape: (batch_size, batch_size)
        is_same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))
        # 2-2. 자기 자신을 제외하는 마스크 (대각선이 False)
        is_not_self = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

        # 2-3. Positive 쌍 마스크: 라벨이 같으면서 자기 자신이 아닌 경우
        positive_mask = is_same_label & is_not_self
        # 2-4. Negative 쌍 마스크: 라벨이 다른 경우
        negative_mask = ~is_same_label

        # 3. 각 Anchor에 대해 Hard Positive 찾기
        # Positive가 아닌 쌍의 거리를 음의 무한대로 만들어 argmax 계산에서 제외
        anchor_positive_dist = pairwise_dist.clone()
        anchor_positive_dist[~positive_mask] = -torch.inf
        # 각 행(anchor)에서 거리가 가장 먼(max) 샘플의 인덱스를 찾음
        hard_positive_indices = torch.argmax(anchor_positive_dist, dim=1)

        # 4. 각 Anchor에 대해 Hard Negative 찾기
        # Negative가 아닌 쌍의 거리를 양의 무한대로 만들어 argmin 계산에서 제외
        anchor_negative_dist = pairwise_dist.clone()
        anchor_negative_dist[~negative_mask] = torch.inf
        # 각 행(anchor)에서 거리가 가장 가까운(min) 샘플의 인덱스를 찾음
        hard_negative_indices = torch.argmin(anchor_negative_dist, dim=1)

        # 최종적으로 인덱스를 사용하여 임베딩과 라벨을 가져옴
        anchors = embeddings
        positives = embeddings[hard_positive_indices]
        negatives = embeddings[hard_negative_indices]

        anchor_labels = labels
        pos_labels = labels[hard_positive_indices]
        neg_labels = labels[hard_negative_indices]
        return anchors, positives, negatives, anchor_labels, pos_labels, neg_labels

    def contrastive_loss(self, anchor, other, label, margin=1.0):
        # label: 1 (positive), 0 (negative)
        euclidean_dist = F.pairwise_distance(anchor, other, keepdim=True)
        loss = label * (euclidean_dist ** 2) + (1 - label) * (F.relu(margin - euclidean_dist) ** 2)
        return loss.mean()

    def calculate_base_loss(self, images_embeds_list, labels, loss_type):
        triplet_losses, ortho_losses, total_losses = [], [], []
        for images_embeds in images_embeds_list:
            anchors, positives, negatives, anchor_labels, pos_labels, neg_labels = \
                self.vectorized_sample_hard_triplets(images_embeds, labels)
            if anchors is None:
                self.log("train/skipped_batch", 1.0, on_step=True, logger=True, sync_dist=True)
                zero_loss = sum(torch.sum(embed) for embed in images_embeds_list) * 0.0

                triplet_losses.append(zero_loss)
                ortho_losses.append(zero_loss)
                total_losses.append(zero_loss)
                continue

            # Triplet loss
            # triplet_loss = F.triplet_margin_loss(anchors, positives, negatives, margin=self.hparams.margin)

            # contrastive_pair_loss
            positive_loss = self.contrastive_loss(anchors, positives, torch.ones_like(anchor_labels),
                                                  margin=self.hparams.margin)
            negative_loss = self.contrastive_loss(anchors, negatives, torch.zeros_like(anchor_labels),
                                                  margin=self.hparams.margin)
            triplet_loss = (positive_loss + negative_loss) / 2.0

            # Ortho loss
            all_embeds = torch.cat([anchors, positives, negatives], dim=0)
            all_labels = torch.cat([anchor_labels, pos_labels, neg_labels], dim=0)
            ortho_loss = self.class_aware_ortho_hash_loss(all_embeds, all_labels)
            total_loss = triplet_loss + self.hparams.lambda_ortho * ortho_loss

            triplet_losses.append(triplet_loss)
            ortho_losses.append(ortho_loss)
            total_losses.append(total_loss)
        triplet_loss = sum(triplet_losses)
        ortho_loss = sum(ortho_losses)
        total_loss = sum(total_losses)
        self.log(f"{loss_type}/triplet_loss", triplet_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=images_embeds_list[0].size(0))
        self.log(f"{loss_type}/ortho_loss", ortho_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=images_embeds_list[0].size(0))
        self.log(f"{loss_type}/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=images_embeds_list[0].size(0))
        return total_losses

    # Dominance-Award Dynamic Weightning Strategy
    # 최종 linear layer에서만 gradient 충돌을 해결
    def calculate_alpha(self, total_losses, bit_list_length):
        alphas = [1.0] * bit_list_length

        final_linear_weight = self.nhl.hash_head[2].weight
        W_nested = [final_linear_weight[:b, :] for b in self.hparams.bit_list]
        for k in range(bit_list_length):
            # Dominant Gradient(g_k^k)를 계산
            g_k_k = torch.autograd.grad(total_losses[k], W_nested[k], retain_graph=True, allow_unused=True)[0]
            if g_k_k is None:
                g_k_k = torch.zeros_like(W_nested[k])
            for i in range(k + 1, bit_list_length):
                # 다른 목적 함수가 W_k에 가하는 gradient(g_i^k)를 계산
                g_i_k = torch.autograd.grad(total_losses[i], W_nested[k], retain_graph=True, allow_unused=True)[0]
                if g_i_k is None:
                    g_i_k = torch.zeros_like(W_nested[k])

                # 내적을 통해 Anti-Domination 상태인지확인
                inner_product = torch.sum(g_i_k * g_k_k)
                if inner_product < 0:
                    # 충돌 시 가중치 후보 alpha_i_k를 계산 후 alpha_i를 업데이트
                    alpha_i_k = (alphas[k] / (k - bit_list_length)) * (torch.sum(g_k_k ** 2) / inner_product)
                    alphas[i] = min(alphas[i], alpha_i_k.item())

        # 가중치의 합이 비트 리스트의 길이가 되도록 정규화
        alpha_sum = sum(alphas)
        alphas = [(alpha / alpha_sum) * len(alphas) for alpha in alphas]
        return alphas

    # Long-short Cascade Self-distillation loss를 계산하는 함수
    def calculate_lcs_loss(self, hash_codes):
        lcs_losses = []
        bit_length = len(self.hparams.bit_list)
        for i in range(bit_length - 1):
            # long hash code에서 short hash code로의 단방향 학습을 위해 deatch를 통해 그래디언트 전파를 막음
            teacher_code = hash_codes[i + 1].detach()
            student_code = hash_codes[i]

            # 배치 내 샘플 간의 관계를 나타내는 유사도 행렬을 계산
            sim_teacher = F.normalize(teacher_code @ teacher_code.T)
            sim_student = F.normalize(student_code @ student_code.T)

            # 두 행렬 간의 차이를 loss로 정의하여, 짧은 코드가 긴 코드의 관계를 배우도록 함
            loss = F.mse_loss(sim_student, sim_teacher)
            lcs_losses.append(loss)
        return lcs_losses

    def consistency_loss(self, anchors, positives):
        return F.mse_loss(anchors, positives)

    def quantization_loss(self, embeddings):
        return torch.mean((embeddings.abs() - 1) ** 2)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        images, labels = batch
        unique_labels = labels.unique()
        if unique_labels.numel() < 2:
            dummy_loss = torch.tensor(1e-6, requires_grad=True, device=self.device)
            self.log("train/skipped_batch", 1.0, on_step=True, logger=True, sync_dist=True)
            return dummy_loss
        # 중복 계산을 방지하기 위해 전체 배치 이미지를 한 번에 임베딩 게산
        images_embeds_list = self(images)
        base_losses = self.calculate_base_loss(images_embeds_list, labels, loss_type='train')

        # --- LCS 손실 계산 ---
        hash_codes = [torch.sign(out) for out in images_embeds_list]
        lcs_losses = self.calculate_lcs_loss(hash_codes)

        # --- Consistency Loss 계산 ---
        cons_losses = []
        for embeds in images_embeds_list:
            anchors, positives, _, _, _, _ = self.vectorized_sample_hard_triplets(embeds, labels)
            cons_loss = self.consistency_loss(anchors, positives)
            cons_losses.append(cons_loss)

        # --- Quantization Loss 계산 ---
        quant_losses = [self.quantization_loss(embed) for embed in images_embeds_list]

        # 3. PCGrad에 전달할 최종 손실 리스트 구성
        # 각 bit 길이의 base_loss에 보조 손실들을 더해 하나의 태스크로 묶습니다.
        pcgrad_tasks = []
        bit_list_length = len(self.hparams.bit_list)
        for k in range(bit_list_length):
            task_loss = base_losses[k]
            task_loss += self.hparams.lambda_quant * quant_losses[k]
            task_loss += self.hparams.lambda_cons * cons_losses[k]
            if k < bit_list_length - 1:
                task_loss += self.hparams.lambda_lcs * lcs_losses[k]
            pcgrad_tasks.append(task_loss)
        # 4. 수동으로 그래디언트 계산 및 파라미터 업데이트
        optimizer.zero_grad()
        optimizer.pc_backward(pcgrad_tasks)  # PCGrad의 핵심!
        optimizer.step()

        final_loss = sum(pcgrad_tasks)  # 로깅을 위한 전체 손실 합
        self.log("train/quant_loss", sum(quant_losses), on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=images.size(0))
        self.log("train/lcs_loss", sum(lcs_losses), on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=images.size(0))
        self.log("train/cons_loss", sum(cons_losses), on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=images.size(0))
        self.log("train/final_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 sync_dist=True, batch_size=images.size(0))
        return total_loss

    def calculate_sim_acc(self, anchors, pos, neg, images_embeds):
        # 평균 positive/negative cosine 유사도
        cos = nn.CosineSimilarity(dim=1)
        pos_sim = cos(anchors, pos).mean()
        neg_sim = cos(anchors, neg).mean()

        hash_anchor = torch.sign(anchors)
        hash_pos = torch.sign(pos)
        hash_neg = torch.sign(neg)
        pos_hash_acc = (hash_anchor == hash_pos).all(dim=1).float().mean().item()
        # 네거티브 쌍에서 해시가 충돌(일치)하는 비율
        neg_collision_rate = (hash_anchor == hash_neg).all(dim=1).float().mean().item()

        # val_embeds: 검증 세트 전체 임베딩
        val_codes = torch.sign(images_embeds)
        # 각 비트(열)의 분산을 구하고 평균. 붕괴 시 0에 가까워짐
        # 값이 클수록 좋으므로, (1 - variance)를 페널티로 사용 가능
        mean_bit_variance = val_codes.float().var(dim=0).mean().item()
        return pos_sim, neg_sim, pos_hash_acc, neg_collision_rate, mean_bit_variance

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            images, labels = batch
            images_embeds_list = self(images)
            for images_embeds, bit in zip(images_embeds_list, self.hparams.bit_list):
                anchors, positives, negatives, _, _, _ = self.vectorized_sample_hard_triplets(images_embeds, labels)
                if anchors is None:
                    dummy_loss = torch.tensor(1e-6, requires_grad=True, device=self.device)
                    return dummy_loss
                pos_sim, neg_sim, pos_hash_acc, neg_collision_rate, mean_bit_variance = \
                    self.calculate_sim_acc(anchors, positives, negatives, images_embeds)
                self.log(f"val/{bit}_pos_sim", pos_sim,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f"val/{bit}_neg_sim", neg_sim,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f"val/{bit}_pos_hash_acc", pos_hash_acc,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f"val/{bit}_neg_collision_rate", neg_collision_rate,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                self.log(f"val/{bit}_mean_bit_variance", mean_bit_variance,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
                w_coll = 0.8  # 충돌률 페널티 가중치
                w_var = 0.4  # 분산 보너스 가중치
                final_score = pos_hash_acc - w_coll * neg_collision_rate + w_var * mean_bit_variance
                self.log(f"val/{bit}_final_score", final_score,
                         on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        base_optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = PCGrad(base_optimizer)
        optimizer.param_groups = base_optimizer.param_groups

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