"""
XOR 기반 이미지-텍스트 검색 시스템
1-bit binary hash를 이용한 고속 검색
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple, Union
import numpy as np


class BinaryHashXOR:
    """1-bit binary hash의 XOR 기반 유사도 계산"""

    @staticmethod
    def sign_to_binary(continuous_codes):
        """
        연속 임베딩 → 1-bit binary hash (0/1)

        Args:
            continuous_codes: (batch, hash_dim) with float values

        Returns:
            binary: (batch, hash_dim) with values in {0, 1}
        """
        # torch.sign()은 -1/+1을 반환
        # (sign + 1) / 2 = 0 또는 1
        binary = ((torch.sign(continuous_codes) + 1) / 2).long()
        return binary

    @staticmethod
    def hamming_distance(codes1, codes2):
        """
        XOR 기반 Hamming distance (배치 단위)

        Args:
            codes1: (N, hash_dim) binary codes
            codes2: (M, hash_dim) binary codes

        Returns:
            hamming_dist: (N, M) hamming distances
        """
        # XOR 연산: codes1[:, None, :] ^ codes2[None, :, :]
        # Broadcasting으로 모든 쌍 계산
        xor_result = codes1.unsqueeze(1) ^ codes2.unsqueeze(0)  # (N, M, hash_dim)

        # XOR 결과에서 1의 개수 = Hamming distance
        hamming_dist = xor_result.sum(dim=2)  # (N, M)

        return hamming_dist

    @staticmethod
    def hamming_similarity(codes1, codes2):
        """
        Hamming similarity (1 - normalized Hamming distance)

        Args:
            codes1: (N, hash_dim)
            codes2: (M, hash_dim)

        Returns:
            similarity: (N, M) similarities in [0, 1]
        """
        hash_dim = codes1.shape[1]
        hamming_dist = BinaryHashXOR.hamming_distance(codes1, codes2)

        # 유사도로 변환: 1 - (distance / hash_dim)
        similarity = 1 - (hamming_dist.float() / hash_dim)

        return similarity

    @staticmethod
    def xor_retrieval(query_codes, database_codes, top_k=10):
        """
        XOR 기반 검색

        Args:
            query_codes: (N_query, hash_dim) binary
            database_codes: (N_db, hash_dim) binary
            top_k: 반환할 상위 결과 수

        Returns:
            indices: (N_query, top_k) - 가장 가까운 인덱스
            distances: (N_query, top_k) - Hamming distances
        """
        hamming_dist = BinaryHashXOR.hamming_distance(query_codes, database_codes)

        # 거리가 가장 작은 top_k 선택
        actual_k = min(top_k, database_codes.shape[0])
        top_k_distances, top_k_indices = torch.topk(
            hamming_dist,
            k=actual_k,
            dim=1,
            largest=False,  # 거리가 작을수록 유사
            sorted=True
        )

        return top_k_indices, top_k_distances


class MultiModalRetrieval:
    """
    이미지-텍스트 XOR 기반 검색 시스템
    """

    def __init__(self, model, tokenizer, bit_length=64, device='cuda'):
        """
        Args:
            model: MultiModalHashingModel
            tokenizer: Transformers tokenizer
            bit_length: 사용할 해시 비트 길이 (8, 16, 32, 48, 64, 128)
            device: 'cuda' or 'cpu'
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.bit_length = bit_length
        self.device = device

        # bit_length에 해당하는 인덱스 찾기
        if bit_length in model.hparams.bit_list:
            self.bit_index = model.hparams.bit_list.index(bit_length)
        else:
            raise ValueError(f"bit_length {bit_length} not in model's bit_list {model.hparams.bit_list}")

        self.model.to(device)

    @torch.no_grad()
    def encode_images(self, images):
        """
        이미지 → 1-bit binary hash (0/1)

        Args:
            images: (N, 3, H, W) tensor or list of PIL images

        Returns:
            binary_codes: (N, hash_dim) with {0, 1}
        """
        if not isinstance(images, torch.Tensor):
            # PIL Image list 처리
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            images = torch.stack([transform(img) for img in images])

        images = images.to(self.device)

        outputs = self.model(images=images)
        embeds = outputs['vision'][self.bit_index]  # (N, hash_dim)

        # 이진화: -1/+1 → 0/1
        binary_codes = BinaryHashXOR.sign_to_binary(embeds)

        return binary_codes

    @torch.no_grad()
    def encode_texts(self, texts):
        """
        텍스트 → 1-bit binary hash (0/1)

        Args:
            texts: List of strings

        Returns:
            binary_codes: (N, hash_dim) with {0, 1}
        """
        # 토크나이징
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors='pt'
        ).to(self.device)

        outputs = self.model(texts=text_inputs)
        embeds = outputs['text'][self.bit_index]

        binary_codes = BinaryHashXOR.sign_to_binary(embeds)

        return binary_codes

    def search_image_by_text(self, query_text: str, image_database, top_k=10):
        """
        텍스트 쿼리로 이미지 검색

        Args:
            query_text: str - 검색할 텍스트
            image_database: Tensor (N, 3, H, W) or list of PIL images
            top_k: int - 반환할 상위 결과 수

        Returns:
            indices: (top_k,) - 가장 유사한 이미지 인덱스
            distances: (top_k,) - Hamming distances
        """
        # 1. 텍스트 → 해시 코드
        query_codes = self.encode_texts([query_text])  # (1, hash_dim)

        # 2. 이미지 DB → 해시 코드
        db_codes = self.encode_images(image_database)  # (N, hash_dim)

        # 3. XOR 기반 검색
        indices, distances = BinaryHashXOR.xor_retrieval(
            query_codes, db_codes, top_k=top_k
        )

        return indices[0], distances[0]

    def search_text_by_image(self, query_image, text_database: List[str], top_k=10):
        """
        이미지 쿼리로 텍스트 검색

        Args:
            query_image: Tensor (3, H, W) or PIL Image
            text_database: List of strings
            top_k: int

        Returns:
            indices: (top_k,) - 가장 유사한 텍스트 인덱스
            distances: (top_k,) - Hamming distances
        """
        # 1. 이미지 → 해시 코드
        if not isinstance(query_image, torch.Tensor):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            query_image = transform(query_image)

        query_codes = self.encode_images(query_image.unsqueeze(0))  # (1, hash_dim)

        # 2. 텍스트 DB → 해시 코드
        db_codes = self.encode_texts(text_database)  # (M, hash_dim)

        # 3. XOR 기반 검색
        indices, distances = BinaryHashXOR.xor_retrieval(
            query_codes, db_codes, top_k=top_k
        )

        return indices[0], distances[0]

    def batch_search_i2t(self, query_images, text_database: List[str], top_k=10):
        """
        배치 이미지 → 텍스트 검색

        Args:
            query_images: (N, 3, H, W)
            text_database: List of strings
            top_k: int

        Returns:
            indices: (N, top_k)
            distances: (N, top_k)
        """
        query_codes = self.encode_images(query_images)
        db_codes = self.encode_texts(text_database)

        return BinaryHashXOR.xor_retrieval(query_codes, db_codes, top_k)

    def batch_search_t2i(self, query_texts: List[str], image_database, top_k=10):
        """
        배치 텍스트 → 이미지 검색

        Args:
            query_texts: List of strings
            image_database: (M, 3, H, W)
            top_k: int

        Returns:
            indices: (N, top_k)
            distances: (N, top_k)
        """
        query_codes = self.encode_texts(query_texts)
        db_codes = self.encode_images(image_database)

        return BinaryHashXOR.xor_retrieval(query_codes, db_codes, top_k)

    def save_hash_database(self, images=None, texts=None, save_path='hash_database.pt'):
        """
        해시 코드 데이터베이스 저장 (사전 계산하여 재사용)

        Args:
            images: 이미지 데이터베이스
            texts: 텍스트 데이터베이스
            save_path: 저장 경로
        """
        database = {}

        if images is not None:
            database['image_codes'] = self.encode_images(images).cpu()

        if texts is not None:
            database['text_codes'] = self.encode_texts(texts).cpu()
            database['texts'] = texts

        torch.save(database, save_path)
        print(f"Hash database saved to {save_path}")

    def load_hash_database(self, load_path='hash_database.pt'):
        """
        사전 계산된 해시 코드 데이터베이스 로드

        Args:
            load_path: 로드 경로

        Returns:
            database: dict with 'image_codes', 'text_codes', 'texts'
        """
        database = torch.load(load_path, map_location=self.device)
        print(f"Hash database loaded from {load_path}")
        return database

    def fast_search_with_precomputed(self, query_code, precomputed_db_codes, top_k=10):
        """
        사전 계산된 DB 코드로 빠른 검색

        Args:
            query_code: (1, hash_dim) or (hash_dim,)
            precomputed_db_codes: (N, hash_dim) - 사전 계산된 DB 해시 코드
            top_k: int

        Returns:
            indices: (top_k,)
            distances: (top_k,)
        """
        if query_code.dim() == 1:
            query_code = query_code.unsqueeze(0)

        query_code = query_code.to(self.device)
        precomputed_db_codes = precomputed_db_codes.to(self.device)

        indices, distances = BinaryHashXOR.xor_retrieval(
            query_code, precomputed_db_codes, top_k=top_k
        )

        return indices[0], distances[0]


class HashCodeAnalyzer:
    """해시 코드 품질 분석 도구"""

    @staticmethod
    def analyze_hash_distribution(hash_codes):
        """
        해시 코드 분포 분석

        Args:
            hash_codes: (N, hash_dim) binary codes

        Returns:
            dict with analysis results
        """
        # 비트별 분포
        bit_means = hash_codes.float().mean(dim=0)  # 각 비트의 평균 (0.5에 가까울수록 좋음)
        bit_variance = hash_codes.float().var(dim=0)  # 각 비트의 분산 (높을수록 좋음)

        # 균형도 (0.5에 가까울수록 균형잡힘)
        balance = 1 - torch.abs(bit_means - 0.5) * 2  # [0, 1], 1이 가장 균형잡힘

        # 엔트로피
        entropy = -bit_means * torch.log2(bit_means + 1e-8) - \
                  (1 - bit_means) * torch.log2(1 - bit_means + 1e-8)

        return {
            'bit_means': bit_means.cpu().numpy(),
            'bit_variance': bit_variance.cpu().numpy(),
            'balance': balance.cpu().numpy(),
            'entropy': entropy.cpu().numpy(),
            'mean_balance': balance.mean().item(),
            'mean_variance': bit_variance.mean().item(),
            'mean_entropy': entropy.mean().item()
        }

    @staticmethod
    def calculate_collision_rate(hash_codes, labels):
        """
        충돌률 계산 (다른 라벨인데 같은 해시 코드를 가지는 비율)

        Args:
            hash_codes: (N, hash_dim) binary codes
            labels: (N,) labels

        Returns:
            collision_rate: float
        """
        N = hash_codes.shape[0]

        # 모든 쌍 비교
        is_same_hash = (BinaryHashXOR.hamming_distance(hash_codes, hash_codes) == 0)
        is_diff_label = (labels.unsqueeze(0) != labels.unsqueeze(1))

        # 자기 자신 제외
        not_self = ~torch.eye(N, dtype=torch.bool, device=hash_codes.device)

        # 다른 라벨인데 해시가 같은 경우
        collisions = (is_same_hash & is_diff_label & not_self).sum().item()
        total_diff_label_pairs = (is_diff_label & not_self).sum().item()

        collision_rate = collisions / max(total_diff_label_pairs, 1)

        return collision_rate

    @staticmethod
    def visualize_hamming_distance_distribution(hash_codes, labels, save_path=None):
        """
        Hamming distance 분포 시각화

        Args:
            hash_codes: (N, hash_dim)
            labels: (N,)
            save_path: 저장 경로 (None이면 표시만)
        """
        import matplotlib.pyplot as plt

        # Hamming distance 계산
        hamming_dist = BinaryHashXOR.hamming_distance(hash_codes, hash_codes)

        # Same label vs Different label
        is_same_label = (labels.unsqueeze(0) == labels.unsqueeze(1))
        not_self = ~torch.eye(len(labels), dtype=torch.bool, device=hash_codes.device)

        same_label_distances = hamming_dist[is_same_label & not_self].cpu().numpy()
        diff_label_distances = hamming_dist[~is_same_label].cpu().numpy()

        # 히스토그램
        plt.figure(figsize=(10, 6))
        plt.hist(same_label_distances, bins=50, alpha=0.5, label='Same Label', color='blue')
        plt.hist(diff_label_distances, bins=50, alpha=0.5, label='Different Label', color='red')
        plt.xlabel('Hamming Distance')
        plt.ylabel('Frequency')
        plt.title('Hamming Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
