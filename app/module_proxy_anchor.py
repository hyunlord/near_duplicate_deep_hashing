import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import AutoModel


# 파일 상단에 다른 import 문들과 함께 추가
class ProxyAnchorLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim, margin=0.5, alpha=32):
        super(ProxyAnchorLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.alpha = alpha

        # 각 클래스를 대표하는 프록시를 학습 가능한 파라미터로 선언
        self.proxies = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def forward(self, embeddings, labels):
        # 임베딩과 프록시 L2 정규화
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        proxies_norm = F.normalize(self.proxies, p=2, dim=1)

        # 코사인 유사도 계산
        # cos_sim: (batch_size, num_classes)
        cos_sim = torch.matmul(embeddings_norm, proxies_norm.t())

        # Positive / Negative 프록시 구분
        # one_hot: (batch_size, num_classes)
        one_hot = torch.zeros_like(cos_sim)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        positive_proxies = cos_sim[one_hot.bool()]
        negative_proxies = cos_sim[~one_hot.bool()]

        # Proxy-Anchor Loss 계산 (논문의 수식 기반)
        # F.logsigmoid(x) = log(1 / (1 + exp(-x))) -> log(1 + exp(x))와 유사한 효과 + 수치적 안정성
        pos_loss = torch.mean(F.logsigmoid(-(positive_proxies - self.margin) * self.alpha))
        neg_loss = torch.mean(F.logsigmoid((negative_proxies + self.margin) * self.alpha))

        loss = -pos_loss - neg_loss
        return loss


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

        backbone = AutoModel.from_pretrained(self.hparams.model_name)
        backbone.config.gradient_checkpointing = True
        backbone.gradient_checkpointing_enable()
        self.vision_model = backbone.vision_model
        self.nhl = NestedHashLayer(self.vision_model.config.hidden_size, self.hparams.hash_hidden_dim, self.hparams.bit_list)

        self.validation_step_outputs_mAP = {bit: [] for bit in self.hparams.bit_list}
        self.validation_step_outputs_acc = {bit: [] for bit in self.hparams.bit_list}

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

    def calculate_base_loss(self, images_embeds_list, labels, loss_type):
        triplet_losses, ortho_losses, total_losses = [], [], []
        for images_embeds in images_embeds_list:
            anchors, positives, negatives, anchor_labels, pos_labels, neg_labels = self.vectorized_sample_hard_triplets(images_embeds, labels)
            if anchors is None:
                self.log("train/skipped_batch", 1.0, on_step=True, logger=True, sync_dist=True)
                zero_loss = sum(torch.sum(embed) for embed in images_embeds_list) * 0.0

                triplet_losses.append(zero_loss)
                ortho_losses.append(zero_loss)
                total_losses.append(zero_loss)
                continue

            # Triplet loss
            triplet_loss = F.triplet_margin_loss(anchors, positives, negatives, margin=self.hparams.margin)

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
                    alpha_i_k = (alphas[k] / (k - bit_list_length)) * (torch.sum(g_k_k**2) / inner_product)
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

    def training_step(self, batch, batch_idx):
        images, labels = batch
        # 중복 계산을 방지하기 위해 전체 배치 이미지를 한 번에 임베딩 게산
        images_embeds_list = self(images)
        total_losses = self.calculate_base_loss(images_embeds_list, labels, loss_type='train')

        bit_list_length = len(self.hparams.bit_list)
        alphas = self.calculate_alpha(total_losses, bit_list_length)

        # --- 장단기 계단식 자기 증류 손실 계산 ---
        hash_codes = [torch.sign(out) for out in images_embeds_list]
        lcs_losses = self.calculate_lcs_loss(hash_codes)

        # --- 최종 목표 함수 계산 ---
        total_loss = 0
        for k in range(bit_list_length - 1):
            total_loss += alphas[k] * (total_losses[k] + self.hparams.lambda_lcs * lcs_losses[k])
        total_loss += alphas[bit_list_length - 1] * total_losses[bit_list_length - 1]
        self.log("train/lcs_loss", sum(lcs_losses),
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        self.log("train/final_loss", total_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        return total_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        with torch.no_grad():
            images_embeds_list = self(images)
        total_losses = self.calculate_base_loss(images_embeds_list, labels, loss_type='val')
        final_val_loss = sum(total_losses)
        self.log("val/final_loss", final_val_loss,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=images.size(0))
        for images_embeds, bit in zip(images_embeds_list, self.hparams.bit_list):
            self.validation_step_outputs_mAP[bit].append((images_embeds.detach(), labels.detach()))
            anchors, positives, negatives, _, _, _ = self.vectorized_sample_hard_triplets(images_embeds, labels)
            if anchors is not None:
                self.validation_step_outputs_acc[bit].append((anchors.detach(), positives.detach(), negatives.detach()))

    def calculate_and_log_map(self):
        for bit in self.hparams.bit_list:
            outputs = self.validation_step_outputs_mAP[bit]
            if not outputs:
                continue
            embeds_list = [out[0] for out in outputs]
            labels_list = [out[1] for out in outputs]
            embeds = torch.cat(embeds_list, dim=0).cpu()
            labels = torch.cat(labels_list, dim=0).cpu().numpy()

            codes = (torch.sign(embeds) > 0).byte().numpy()
            query_labels = (labels[:, None] == labels[None, :]).astype(np.uint8)
            hamm_dist = cdist(codes, codes, metric='hamming') * bit

            aps = []
            for i in range(codes.shape[0]):
                y_true = query_labels[i]
                # Hamming 거리가 가까울수록 점수가 높아야 하므로 음수를 취합니다.
                y_scores = -hamm_dist[i]
                aps.append(average_precision_score(y_true, y_scores))
            if not aps:
                mean_ap = 0.0
                print(f"Could not compute mAP for {bit}-bit (all classes have <= 1 sample).")
            else:
                mean_ap = float(np.mean(aps))
            self.log(f"val/{bit}_mAP", mean_ap, prog_bar=True, sync_dist=False)

    def calculate_and_log_acc_sim(self):
        cos = nn.CosineSimilarity(dim=1)
        for bit in self.hparams.bit_list:
            outputs = self.validation_step_outputs_acc[bit]
            if not outputs:
                print(f"Skipping Acc/Sim calculation for {bit}-bit: no validation outputs.")
                continue
            # 모든 스텝의 (anchor, pos, neg) 튜플을 각각의 리스트로 분리
            # outputs는 [(A1, P1, N1), (A2, P2, N2), ...] 형태
            anchors_list, pos_list, neg_list = zip(*outputs)
            # 전체 앵커, 포지티브, 네거티브를 하나의 큰 텐서로 합침
            all_anchors = torch.cat(anchors_list, dim=0)
            all_positives = torch.cat(pos_list, dim=0)
            all_negatives = torch.cat(neg_list, dim=0)
            # 1. 유사도 계산
            pos_sim = cos(all_anchors, all_positives).mean().item()
            neg_sim = cos(all_anchors, all_negatives).mean().item()
            # 2. 해시 정확도 계산
            hash_anchor = torch.sign(all_anchors)
            hash_pos = torch.sign(all_positives)
            # (전체 정답 쌍 개수) / (전체 쌍 개수)
            pos_hash_acc = (hash_anchor == hash_pos).all(dim=1).float().mean().item()
            self.log(f"val/{bit}_pos_sim", pos_sim, prog_bar=True, sync_dist=False)
            self.log(f"val/{bit}_neg_sim", neg_sim, prog_bar=True, sync_dist=False)
            self.log(f"val/{bit}_pos_hash_acc", pos_hash_acc, prog_bar=True, sync_dist=False)

    def calculate_and_log_recall_at_k(self):
        for bit in self.hparams.bit_list:
            outputs = self.validation_step_outputs_mAP[bit]
            if not outputs:
                continue
            # 1. 데이터 취합 및 해시 코드 생성
            embeds = torch.cat([out[0] for out in outputs], dim=0).cpu()
            labels = torch.cat([out[1] for out in outputs], dim=0).cpu().numpy()
            codes = (torch.sign(embeds) > 0).numpy().astype(np.uint8)
            num_data = codes.shape[0]
            # 2. 모든 쌍 간의 Hamming 거리 계산
            hamm_dist_matrix = cdist(codes, codes, metric='hamming') * bit
            # 3. 각 쿼리에 대한 전체 정답 개수 미리 계산
            # is_relevant[i, j] = True if label[i] == label[j]
            is_relevant = labels[:, None] == labels[None, :]
            np.fill_diagonal(is_relevant, False)
            total_relevant_per_query = is_relevant.sum(axis=1)
            # 4. 각 K 값에 대해 Recall 계산
            for k in self.hparams.recall_k_values:
                recalls_for_this_k = []
                for i in range(num_data):
                    # i번째 쿼리의 전체 정답 개수
                    total_relevant = total_relevant_per_query[i]
                    # i번째 쿼리와 다른 모든 데이터 간의 거리
                    dists = hamm_dist_matrix[i]
                    # 거리를 기준으로 인덱스 정렬 (가장 가까운 순)
                    sorted_indices = np.argsort(dists)
                    # 상위 K개의 이웃 선택 (자기 자신인 첫 번째 인덱스 제외)
                    retrieved_indices = sorted_indices[1:k + 1]
                    # 상위 K개 이웃이 정답인지 확인
                    # is_relevant[i, retrieved_indices] -> 쿼리 i에 대해, 뽑힌 애들이 정답인지 아닌지 bool 배열
                    num_retrieved_relevant = is_relevant[i, retrieved_indices].sum()
                    # Recall 계산
                    recall_for_query = num_retrieved_relevant / total_relevant
                    recalls_for_this_k.append(recall_for_query)
                # 모든 쿼리에 대한 Recall@K의 평균 계산
                if recalls_for_this_k:
                    mean_recall_at_k = np.mean(recalls_for_this_k)
                    self.log(f"val/{bit}_Recall@{k}", mean_recall_at_k, prog_bar=True, sync_dist=False)

    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return
        # --- 1. mAP 계산 ---
        self.calculate_and_log_map()
        # --- 2. Accuracy 및 Similarity 계산 ---
        self.calculate_and_log_acc_sim()
        # --- 3. Recall@K 계산 ---
        self.calculate_and_log_recall_at_k()
        # --- 4. 다음 에폭을 위해 저장된 출력 리스트 비우기 ---
        for bit in self.hparams.bit_list:
            self.validation_step_outputs_mAP[bit].clear()
            self.validation_step_outputs_acc[bit].clear()

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
